"""Plot Schirmer vascular-territory statistics as an axial mosaic.

The atlas is displayed over a user-supplied anatomical or perfusion image.
Its outer footprint is clipped to the nonzero background mask, while signed
distance fields provide smooth internal ROI interfaces.  Significant and
non-significant territories share one continuous colour scale but use
different opacity and outline width.

The statistical CSV requires ``label_id`` and ``statistic_value``.  Add one
of ``significant``, ``q_value`` or ``p_value`` to encode significance.  A
direct significance flag takes precedence, followed by q-value and p-value.
If no significance column is present, every supplied ROI is significant.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ATLAS = (
    REPO_ROOT / "cvdproc/data/atlas/Schirmer_VT/mni_vascular_territories.nii.gz"
)
DEFAULT_LABELS = DEFAULT_ATLAS.with_suffix("").with_suffix(".csv")
DEFAULT_Z_SLICES = (-60, -50, -40, -30, -20, -10, 0, 10, 25, 40, 60, 70)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from surface_plotting_utils import (  # noqa: E402
    colour_limits,
    label_lookup,
    load_statistical_csv,
)


def parse_number_list(text):
    """Parse comma- or space-separated floating-point values."""
    if isinstance(text, (list, tuple)):
        return [float(value) for value in text]
    return [float(item) for item in str(text).replace(",", " ").split()]


def parse_integer_list(text):
    """Parse comma- or space-separated integer values."""
    return [int(value) for value in parse_number_list(text)]


def load_schirmer_names(path, excluded_labels=()):
    """Return ``label_id -> descriptive name`` from the Schirmer LUT."""
    excluded = set(int(label) for label in excluded_labels) | {0}
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label = int(row["id"])
            if label in excluded:
                continue
            hemisphere = row.get("hemisphere", "").strip()
            territory = row.get("label", "").strip()
            result[label] = " ".join(part for part in (hemisphere, territory) if part)
    return result


def load_in_reference(path, reference):
    """Load a NIfTI on the exact reference grid, without resampling."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    same_grid = image.shape[:3] == reference.shape[:3] and np.allclose(
        image.affine, reference.affine, rtol=0, atol=1e-4
    )
    if not same_grid:
        affine_difference = float(np.max(np.abs(image.affine - reference.affine)))
        raise ValueError(
            "Atlas and background must already be on the same voxel grid; "
            "automatic resampling is disabled. "
            f"Atlas shape={image.shape[:3]}, background shape={reference.shape[:3]}, "
            f"maximum affine difference={affine_difference:.6g}."
        )
    return np.asarray(image.get_fdata(dtype=np.float32))


def validate_inputs(atlas, statistics, names):
    """Validate CSV labels and report the territories selected for drawing."""
    atlas_labels = set(int(value) for value in np.unique(atlas)) - {0}
    supplied = set(int(key[-1] if isinstance(key, tuple) else key) for key in statistics)
    invalid = sorted(supplied - atlas_labels)
    if invalid:
        raise ValueError(f"CSV label_id values absent from the Schirmer atlas: {invalid}")
    unnamed = sorted(supplied - set(names))
    if unnamed:
        raise ValueError(f"CSV labels excluded from or absent in the Schirmer LUT: {unnamed}")
    missing = sorted(set(names) - supplied)
    if missing:
        print(f"Note: {len(missing)} Schirmer ROIs have no CSV row and will not be drawn.")
    for label in sorted(supplied):
        print(f"  {label:2d}: {names[label]}")


def statistic_normalizer(statistics, vmin=None, vmax=None):
    """Create a continuous, zero-centred normalizer when values diverge."""
    values = [float(value) for value in statistics.values()]
    lo, hi = colour_limits(values, vmin, vmax, symmetric_if_diverging=True)
    if lo == hi:
        padding = max(abs(lo) * 0.05, 1e-6)
        lo, hi = lo - padding, hi + padding
    if lo < 0 < hi:
        limit = max(abs(lo), abs(hi))
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), -limit, limit
    return Normalize(vmin=lo, vmax=hi), lo, hi


def background_limits(data, mask, percentiles, vmin=None, vmax=None):
    """Determine robust background display limits inside its valid mask."""
    sample = data[mask & np.isfinite(data)]
    if sample.size == 0:
        raise ValueError("The background contains no finite nonzero voxels")
    lo = float(np.percentile(sample, percentiles[0])) if vmin is None else float(vmin)
    hi = float(np.percentile(sample, percentiles[1])) if vmax is None else float(vmax)
    if not lo < hi:
        raise ValueError(f"Invalid background range: {lo:g} to {hi:g}")
    return lo, hi


def world_z_to_index(image, z_mm):
    voxel = np.linalg.inv(image.affine) @ np.array([0.0, 0.0, z_mm, 1.0])
    return int(np.rint(voxel[2]))


def orient_axial(array, display_left_is_left=True):
    """Orient an x-y voxel plane for neurological or radiological display."""
    # Canonical NIfTI data are RAS: the second voxel axis increases from
    # posterior to anterior.  ``rot90`` alone leaves anterior at the bottom
    # under ``imshow(origin='lower')``.  Flip vertically so axial mosaics use
    # the conventional anterior-at-top, posterior-at-bottom presentation.
    result = np.flipud(np.rot90(array))
    return np.fliplr(result) if display_left_is_left else result


def smooth_signed_distance(mask, affine, sigma_mm):
    """Return a sub-voxel signed-distance field, positive inside ``mask``."""
    spacing = nib.affines.voxel_sizes(affine)
    signed = distance_transform_edt(mask, sampling=spacing) - distance_transform_edt(
        ~mask, sampling=spacing
    )
    if sigma_mm > 0:
        signed = gaussian_filter(signed, sigma=np.asarray(sigma_mm) / spacing, mode="nearest")
    return signed


def prepare_sdf_planes(atlas, labels, reference, z_indices, sigma_mm):
    """Compute only requested planes from per-ROI and union distance fields."""
    roi_planes = []
    for label in labels:
        sdf = smooth_signed_distance(atlas == label, reference.affine, sigma_mm)
        roi_planes.append(
            np.stack([sdf[:, :, index] for index in z_indices]).astype(np.float32)
        )
    union_sdf = smooth_signed_distance(np.isin(atlas, labels), reference.affine, sigma_mm)
    union_planes = np.stack(
        [union_sdf[:, :, index] for index in z_indices]
    ).astype(np.float32)
    return np.stack(roi_planes), union_planes


def write_csv_template(path, names):
    """Write a statistical CSV template containing all supported territories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label_id", "roi_name", "statistic_value", "q_value"))
        for label, name in sorted(names.items()):
            writer.writerow((label, name, "", ""))
    print(f"Saved CSV template: {path}")


def plot_mosaic(
    reference,
    background,
    background_mask,
    atlas,
    labels,
    statistics,
    significance,
    output,
    *,
    z_slices,
    rows,
    images_per_row,
    sigma_mm,
    iso_level,
    cmap_name,
    norm,
    bg_vmin,
    bg_vmax,
    bg_gamma,
    significant_opacity,
    nonsignificant_opacity,
    significant_linewidth,
    nonsignificant_linewidth,
    display_left_is_left,
    dpi,
    colorbar_label,
    title,
    canvas_color,
    figure2_layout,
):
    """Render the clipped, smoothed Schirmer mosaic."""
    valid = []
    for z_mm in z_slices:
        index = world_z_to_index(reference, z_mm)
        if 0 <= index < background.shape[2]:
            valid.append((float(z_mm), index))
        else:
            print(f"Warning: z={z_mm:g} mm is outside the background and was skipped")
    expected = rows * images_per_row
    if len(valid) != expected:
        raise ValueError(
            f"The layout requires {expected} valid slices ({rows} rows x "
            f"{images_per_row}), but {len(valid)} were available"
        )

    z_indices = [index for _, index in valid]
    roi_planes, union_planes = prepare_sdf_planes(
        atlas, labels, reference, z_indices, sigma_mm
    )
    cmap = plt.get_cmap(cmap_name)
    background_cmap = plt.get_cmap("gray").copy()
    background_cmap.set_bad(canvas_color)
    canvas_rgba = to_rgba(canvas_color)
    marker_color = "black" if sum(canvas_rgba[:3]) / 3.0 > 0.5 else "white"
    colours = np.asarray([cmap(norm(float(statistics[label])))[:3] for label in labels])
    is_significant = np.asarray(
        [bool(label_lookup(significance, label, default=True)) for label in labels]
    )
    alphas = np.where(is_significant, significant_opacity, nonsignificant_opacity)

    if figure2_layout:
        # Measured from the six-slice CBF group in Figure_2.pptx
        # (group extent 2752914 x 2229607 EMU; width/height = 1.2347).
        # The source panel uses nearly touching columns and only a narrow row gap.
        figure_width = 8.8
        figure_height = figure_width / (2752914.0 / 2229607.0)
    else:
        figure_width = max(10.0, images_per_row * 3.1)
        figure_height = max(5.0, rows * 3.0 + 1.4)
    if figure2_layout:
        if (rows, images_per_row) != (2, 3):
            raise ValueError("--figure2-layout requires --rows 2 --images-per-row 3")
        fig = plt.figure(
            figsize=(figure_width, figure_height), facecolor=canvas_color
        )
        # Exact visible-brain geometry measured from the final Figure 2 CBF
        # panel (366 x 276 px), rather than from the larger PowerPoint picture
        # frames that still contain internal white margins. Visible gaps in
        # the reference are 7/5 px (top), 6/10 px (bottom) and 6/8/7 px
        # between rows. Each axis is additionally cropped to its brain mask
        # below, so these boxes describe the visible anatomy rather than an
        # uncropped source canvas.
        ppt_boxes_top_down = (
            (15 / 366, 17 / 276, 102 / 366, 114 / 276),
            (124 / 366, 10 / 276, 108 / 366, 126 / 276),
            (237 / 366, 6 / 276, 110 / 366, 136 / 276),
            (11 / 366, 137 / 276, 108 / 366, 137 / 276),
            (125 / 366, 144 / 276, 106 / 366, 130 / 276),
            (241 / 366, 149 / 276, 103 / 366, 117 / 276),
        )
        panel_bottom, panel_top = 0.155, 0.865
        panel_height = panel_top - panel_bottom
        panel_width = panel_height * (366 / 276) / (figure_width / figure_height)
        panel_left, panel_right = (1 - panel_width) / 2, (1 + panel_width) / 2
        axes_flat = []
        for x, y_from_top, width, height in ppt_boxes_top_down:
            axes_flat.append(fig.add_axes([
                panel_left + x * panel_width,
                panel_top - (y_from_top + height) * panel_height,
                width * panel_width,
                height * panel_height,
            ]))
        axes = np.asarray(axes_flat, dtype=object).reshape(rows, images_per_row)
    else:
        fig, axes = plt.subplots(
            rows, images_per_row, figsize=(figure_width, figure_height), squeeze=False,
            facecolor=canvas_color
        )
    for slice_number, (axis, (z_mm, index)) in enumerate(
        zip(axes.ravel(), valid)
    ):
        brain = orient_axial(
            background_mask[:, :, index], display_left_is_left
        ).astype(bool)
        bg = orient_axial(background[:, :, index], display_left_is_left)
        scaled_bg = np.clip((bg - bg_vmin) / (bg_vmax - bg_vmin), 0.0, 1.0)
        scaled_bg = np.power(np.nan_to_num(scaled_bg, nan=0.0), bg_gamma)

        fields = np.stack(
            [
                orient_axial(roi_planes[r, slice_number], display_left_is_left)
                for r in range(len(labels))
            ]
        )
        union = orient_axial(union_planes[slice_number], display_left_is_left)
        winner = np.argmax(fields, axis=0)
        inside = (union >= iso_level) & brain

        rgba = np.zeros((*inside.shape, 4), dtype=np.float32)
        rgba[..., :3] = colours[winner]
        rgba[..., 3] = inside.astype(np.float32) * alphas[winner]

        axis.set_facecolor(canvas_color)
        axis.imshow(
            np.ma.masked_where(~brain, scaled_bg), cmap=background_cmap, origin="lower",
            vmin=0, vmax=1, interpolation="bilinear"
        )
        axis.imshow(rgba, origin="lower", interpolation="bilinear")

        if figure2_layout and brain.any():
            # Remove the large slice-dependent white margins that remain when
            # a full axial array is drawn in each picture box. This is the key
            # distinction between the compact Figure 2 panel and a regular
            # uncropped subplot mosaic.
            yy, xx = np.where(brain)
            x_pad = max(1.0, 0.012 * (xx.max() - xx.min() + 1))
            y_pad = max(1.0, 0.012 * (yy.max() - yy.min() + 1))
            axis.set_xlim(xx.min() - x_pad, xx.max() + x_pad)
            axis.set_ylim(yy.min() - y_pad, yy.max() + y_pad)

        # Thin complete outlines first, followed by full thick significant
        # outlines.  Clipping ``region`` by ``inside`` makes the thick contour
        # include both shared ROI interfaces and the external background edge.
        for roi_index in range(len(labels)):
            region = inside & (winner == roi_index)
            if region.any() and (~region).any():
                axis.contour(
                    region.astype(float), levels=[0.5], colors="#555555",
                    linewidths=nonsignificant_linewidth, origin="lower"
                )
        for roi_index, significant in enumerate(is_significant):
            if not significant:
                continue
            region = inside & (winner == roi_index)
            if region.any() and (~region).any():
                axis.contour(
                    region.astype(float), levels=[0.5], colors="black",
                    linewidths=significant_linewidth, origin="lower"
                )

        if not figure2_layout:
            axis.set_title(f"z = {z_mm:+g} mm", fontsize=10, pad=4)
            left_marker, right_marker = ("L", "R") if display_left_is_left else ("R", "L")
            axis.text(
                0.03, 0.96, left_marker, transform=axis.transAxes, va="top",
                ha="left", fontsize=8, color=marker_color
            )
            axis.text(
                0.97, 0.96, right_marker, transform=axis.transAxes, va="top",
                ha="right", fontsize=8, color=marker_color
            )
        axis.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=15, y=0.985)
    if significance:
        legend_handles = [
            Line2D(
                [0], [0], marker="s", linestyle="none", markersize=9,
                markerfacecolor="#d6604d", alpha=significant_opacity,
                markeredgecolor="black", markeredgewidth=significant_linewidth,
                label="Significant"
            ),
            Line2D(
                [0], [0], marker="s", linestyle="none", markersize=9,
                markerfacecolor="#d6604d", alpha=nonsignificant_opacity,
                markeredgecolor="#555555", markeredgewidth=nonsignificant_linewidth,
                label="Not significant"
            ),
        ]
        fig.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
            ncol=2, frameon=False, fontsize=9
        )

    if figure2_layout:
        left_marker, right_marker = ("L", "R") if display_left_is_left else ("R", "L")
        fig.text(
            max(0.01, panel_left - 0.035), 0.51, left_marker, va="center", ha="left",
            fontsize=13, fontweight="bold", color=marker_color
        )
        fig.text(
            min(0.99, panel_right + 0.035), 0.51, right_marker, va="center", ha="right",
            fontsize=13, fontweight="bold", color=marker_color
        )
    else:
        fig.subplots_adjust(
            left=0.025, right=0.975, top=0.90 if significance else 0.93,
            bottom=0.13, wspace=0.06, hspace=0.16
        )
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    # Keep the horizontal colourbar compact: one third of the figure width.
    colorbar_axis = fig.add_axes([1.0 / 3.0, 0.055, 1.0 / 3.0, 0.022])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(colorbar_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, dpi=dpi,
        bbox_inches=None if figure2_layout else "tight",
        facecolor=canvas_color
    )
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, help="Background NIfTI image")
    parser.add_argument(
        "--stat-csv", type=Path,
        help="CSV: label_id,statistic_value[,significant|q_value|p_value]"
    )
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=Path("Schirmer_mosaic.png"))
    parser.add_argument("--write-csv-template", type=Path)
    parser.add_argument(
        "--exclude-labels", type=parse_integer_list, default=[19],
        help="Atlas labels to ignore (default: 19, the documented single-voxel artifact)"
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--significant-opacity", type=float, default=0.88)
    parser.add_argument("--nonsignificant-opacity", type=float, default=0.36)
    parser.add_argument("--significant-linewidth", type=float, default=1.75)
    parser.add_argument("--nonsignificant-linewidth", type=float, default=0.55)
    parser.add_argument("--smoothing-mm", type=float, default=1.0)
    parser.add_argument(
        "--iso-level", type=float, default=0.0,
        help="Signed-distance display isolevel in mm (default: 0)"
    )
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--z-slices", type=parse_number_list, default=list(DEFAULT_Z_SLICES))
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--images-per-row", type=int, default=4)
    parser.add_argument(
        "--background-mask-threshold", type=float, default=0.0,
        help="Valid background requires abs(value) greater than this threshold"
    )
    parser.add_argument("--bg-vmin", type=float)
    parser.add_argument("--bg-vmax", type=float)
    parser.add_argument("--bg-percentiles", type=parse_number_list, default=[2.0, 98.0])
    parser.add_argument("--bg-gamma", type=float, default=1.0)
    parser.add_argument("--radiological", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--colorbar-label", default="Statistic")
    parser.add_argument("--title", default="Schirmer vascular territories")
    parser.add_argument(
        "--figure2-layout", action="store_true",
        help=("Use the compact 2x3 Figure 2 CBF-panel proportions and spacing; "
              "hide z labels and show only one global L/R pair")
    )
    parser.add_argument(
        "--canvas-color", default="white",
        help="Colour outside the background brain mask (default: white)"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("significant_opacity", "nonsignificant_opacity"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.significant_linewidth <= 0 or args.nonsignificant_linewidth <= 0:
        raise ValueError("Outline widths must be positive")
    if args.smoothing_mm < 0:
        raise ValueError("--smoothing-mm cannot be negative")
    if args.rows < 1 or args.images_per_row < 1:
        raise ValueError("--rows and --images-per-row must be positive")
    if len(args.bg_percentiles) != 2 or not args.bg_percentiles[0] < args.bg_percentiles[1]:
        raise ValueError("--bg-percentiles requires two increasing numbers")
    if len(args.z_slices) != args.rows * args.images_per_row:
        raise ValueError(
            f"--z-slices supplied {len(args.z_slices)} coordinates, but --rows x "
            f"--images-per-row requires {args.rows * args.images_per_row}"
        )

    names = load_schirmer_names(args.labels_csv, args.exclude_labels)
    if args.write_csv_template:
        write_csv_template(args.write_csv_template, names)
        return 0
    if args.background is None:
        raise ValueError("--background is required for rendering")
    if args.stat_csv is None:
        raise ValueError("--stat-csv is required unless --write-csv-template is used")

    statistics, significance = load_statistical_csv(
        args.stat_csv, label_columns=("label_id",), alpha=args.alpha
    )
    reference = nib.as_closest_canonical(nib.load(str(args.background)))
    background = np.asarray(reference.get_fdata(dtype=np.float32))
    background_mask = np.isfinite(background) & (
        np.abs(background) > args.background_mask_threshold
    )
    atlas = np.rint(load_in_reference(args.atlas, reference)).astype(np.int16)
    for label in args.exclude_labels:
        atlas[atlas == label] = 0
    validate_inputs(atlas, statistics, names)

    labels = sorted(int(key[-1] if isinstance(key, tuple) else key) for key in statistics)
    statistics = {label: float(label_lookup(statistics, label)) for label in labels}
    significance = {
        label: bool(label_lookup(significance, label, default=True)) for label in labels
    } if significance else {}
    bg_vmin, bg_vmax = background_limits(
        background, background_mask, args.bg_percentiles, args.bg_vmin, args.bg_vmax
    )
    norm, stat_vmin, stat_vmax = statistic_normalizer(
        statistics, args.vmin, args.vmax
    )
    plot_mosaic(
        reference,
        background,
        background_mask,
        atlas,
        labels,
        statistics,
        significance,
        args.output,
        z_slices=args.z_slices,
        rows=args.rows,
        images_per_row=args.images_per_row,
        sigma_mm=args.smoothing_mm,
        iso_level=args.iso_level,
        cmap_name=args.cmap,
        norm=norm,
        bg_vmin=bg_vmin,
        bg_vmax=bg_vmax,
        bg_gamma=args.bg_gamma,
        significant_opacity=args.significant_opacity,
        nonsignificant_opacity=args.nonsignificant_opacity,
        significant_linewidth=args.significant_linewidth,
        nonsignificant_linewidth=args.nonsignificant_linewidth,
        display_left_is_left=not args.radiological,
        dpi=args.dpi,
        colorbar_label=args.colorbar_label,
        title=args.title,
        canvas_color=args.canvas_color,
        figure2_layout=args.figure2_layout,
    )
    print(f"Saved: {args.output.resolve()}")
    print(f"Statistic range: {stat_vmin:g} to {stat_vmax:g}")
    print(f"Background range: {bg_vmin:g} to {bg_vmax:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
