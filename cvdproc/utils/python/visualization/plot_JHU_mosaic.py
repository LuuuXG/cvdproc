"""Publication-style TBSS/JHU axial mosaic without MRIcroGL.

Layers, from back to front:
    1. user supplied anatomical/FA background (grayscale)
    2. JHU ROI statistic values (continuous colormap)
    3. user supplied TBSS skeleton (green)

The statistical CSV uses the same convention as the other cvdproc surface
plotters.  Required columns are ``label_id`` and ``statistic_value``.  Add
either ``significant`` (0/1, true/false) or ``p_value`` to make non-significant
ROIs translucent.  If neither column is supplied, every ROI is significant.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ATLAS = REPO_ROOT / "cvdproc/data/standard/JHU/JHU-ICBM-labels-1mm.nii.gz"
DEFAULT_LABELS = REPO_ROOT / "cvdproc/data/standard/JHU/JHU-labels.xml"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from surface_plotting_utils import (  # noqa: E402
    colour_limits,
    label_lookup,
    load_statistical_csv,
)


# A compact 12-slice range for the 2 x 6 default layout.  It spans the
# cerebellar/pontine tracts through superior corona radiata, rather than
# spending the final panel on high-z anatomy with essentially no JHU ROI.
DEFAULT_Z_SLICES = (-40, -34, -28, -22, -16, -10, -4, 2, 8, 14, 20, 26)


def parse_number_list(text):
    """Parse comma/space-separated numeric values."""
    if isinstance(text, (list, tuple)):
        return [float(value) for value in text]
    return [float(item) for item in str(text).replace(",", " ").split()]


def load_jhu_names(xml_path):
    """Return the JHU ``label_id -> tract name`` mapping."""
    root = ET.parse(xml_path).getroot()
    return {
        int(node.attrib["index"]): " ".join((node.text or "").split())
        for node in root.findall("./data/label")
    }


def load_in_reference(path, reference, *, order):
    """Load a NIfTI in canonical space and resample it to ``reference``."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    same_grid = image.shape[:3] == reference.shape[:3] and np.allclose(
        image.affine, reference.affine, rtol=0, atol=1e-4
    )
    if not same_grid:
        image = resample_from_to(image, reference, order=order)
    return np.asarray(image.get_fdata(dtype=np.float32))


def validate_inputs(atlas_data, statistics, names):
    atlas_labels = set(int(value) for value in np.unique(atlas_data)) - {0}
    supplied = set(int(key[-1] if isinstance(key, tuple) else key) for key in statistics)
    invalid = sorted(supplied - atlas_labels)
    if invalid:
        raise ValueError(f"CSV label_id values absent from the JHU atlas: {invalid}")
    missing = sorted(atlas_labels - supplied)
    if missing:
        print(f"Note: {len(missing)} atlas ROIs have no CSV row and will not be drawn.")
    for label in sorted(supplied):
        print(f"  {label:2d}: {names.get(label, 'Unknown JHU label')}")


def make_roi_volumes(atlas_data, statistics, significance, nonsig_opacity):
    """Expand ROI-level values and significance into voxel arrays."""
    values = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    opacity = np.zeros(atlas_data.shape, dtype=np.float32)
    for label in np.unique(atlas_data):
        label = int(label)
        if label == 0:
            continue
        value = label_lookup(statistics, label, default=None)
        if value is None:
            continue
        selected = atlas_data == label
        values[selected] = float(value)
        is_significant = bool(label_lookup(significance, label, default=True))
        opacity[selected] = 1.0 if is_significant else nonsig_opacity
    return values, opacity


def background_limits(data, low, high, vmin=None, vmax=None):
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        raise ValueError("The background contains no finite voxels")
    lo = float(np.percentile(sample, low)) if vmin is None else float(vmin)
    hi = float(np.percentile(sample, high)) if vmax is None else float(vmax)
    if not lo < hi:
        raise ValueError(f"Invalid background range: {lo} to {hi}")
    return lo, hi


def statistic_normalizer(statistics, vmin=None, vmax=None):
    values = [float(value) for value in statistics.values()]
    lo, hi = colour_limits(values, vmin, vmax, symmetric_if_diverging=True)
    if lo == hi:
        padding = max(abs(lo) * 0.05, 1e-6)
        lo, hi = lo - padding, hi + padding
    if lo < 0 < hi:
        limit = max(abs(lo), abs(hi))
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), -limit, limit
    return Normalize(vmin=lo, vmax=hi), lo, hi


def world_z_to_index(image, z_mm):
    voxel = np.linalg.inv(image.affine) @ np.array([0.0, 0.0, z_mm, 1.0])
    return int(np.rint(voxel[2]))


def orient_axial(array, display_left_is_left=True):
    result = np.rot90(array)
    return np.fliplr(result) if display_left_is_left else result


def crop_bounds(mask, pad):
    if not np.any(mask):
        return 0, mask.shape[0], 0, mask.shape[1]
    rows, columns = np.where(mask)
    return (
        max(int(rows.min()) - pad, 0),
        min(int(rows.max()) + pad + 1, mask.shape[0]),
        max(int(columns.min()) - pad, 0),
        min(int(columns.max()) + pad + 1, mask.shape[1]),
    )


def alpha_over(destination, source):
    """Composite an RGBA ``source`` over an equally sized destination."""
    source_alpha = source[..., 3:4]
    destination_alpha = destination[..., 3:4]
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    premultiplied = (
        source[..., :3] * source_alpha
        + destination[..., :3] * destination_alpha * (1.0 - source_alpha)
    )
    output = np.zeros_like(destination)
    output[..., :3] = np.divide(
        premultiplied,
        output_alpha,
        out=np.zeros_like(premultiplied),
        where=output_alpha > 0,
    )
    output[..., 3:4] = output_alpha
    return output


def centre_on_canvas(array, shape, fill=0):
    """Centre a 2-D or RGBA array on a fixed-size canvas."""
    result_shape = (*shape, array.shape[2]) if array.ndim == 3 else shape
    result = np.full(result_shape, fill, dtype=array.dtype)
    height, width = array.shape[:2]
    canvas_height, canvas_width = shape
    copy_height = min(height, canvas_height)
    copy_width = min(width, canvas_width)
    source_row = max((height - copy_height) // 2, 0)
    source_column = max((width - copy_width) // 2, 0)
    target_row = max((canvas_height - copy_height) // 2, 0)
    target_column = max((canvas_width - copy_width) // 2, 0)
    result[
        target_row : target_row + copy_height,
        target_column : target_column + copy_width,
        ...,
    ] = array[
        source_row : source_row + copy_height,
        source_column : source_column + copy_width,
        ...,
    ]
    return result


def make_slice_rgba(
    bg,
    values,
    roi_alpha,
    skeleton_mask,
    *,
    bg_vmin,
    bg_vmax,
    bg_gamma,
    background_cmap,
    statistic_cmap,
    norm,
    skeleton_alpha,
):
    """Create one transparent slice in background -> skeleton -> ROI order."""
    brain = np.isfinite(bg) & (bg > 0)
    scaled = np.clip((bg - bg_vmin) / (bg_vmax - bg_vmin), 0.0, 1.0)
    scaled = np.power(np.nan_to_num(scaled, nan=0.0), bg_gamma)
    patch = background_cmap(scaled).astype(np.float32)
    patch[..., 3] = brain.astype(np.float32)

    skeleton_rgba = np.zeros_like(patch)
    skeleton_rgba[..., :3] = (0.10, 0.82, 0.28)
    skeleton_rgba[..., 3] = skeleton_mask.astype(np.float32) * skeleton_alpha
    patch = alpha_over(patch, skeleton_rgba)

    roi_rgba = statistic_cmap(norm(np.nan_to_num(values, nan=norm.vmin))).astype(np.float32)
    roi_rgba[..., 3] = np.where(np.isfinite(values), roi_alpha, 0.0)
    return alpha_over(patch, roi_rgba)


def plot_mosaic(
    background_image,
    background,
    skeleton,
    roi_values,
    roi_opacity,
    output,
    *,
    z_slices,
    rows,
    images_per_row,
    slice_overlap,
    cmap_name,
    norm,
    bg_vmin,
    bg_vmax,
    bg_gamma,
    skeleton_threshold,
    skeleton_alpha,
    skeleton_width,
    crop_pad,
    display_left_is_left,
    dpi,
    colorbar_label,
    canvas_color,
    background_cmap_name,
    show_slice_labels,
    title=None,
):
    valid = []
    for z_mm in z_slices:
        index = world_z_to_index(background_image, z_mm)
        if 0 <= index < background.shape[2]:
            valid.append((float(z_mm), index))
        else:
            print(f"Warning: z={z_mm:g} mm is outside the background and was skipped")
    if not valid:
        raise ValueError("None of the requested MNI z slices intersects the background")

    expected = rows * images_per_row
    if len(valid) != expected:
        raise ValueError(
            f"The layout requires {expected} valid slices ({rows} rows x "
            f"{images_per_row}), but {len(valid)} were available"
        )
    cmap = plt.get_cmap(cmap_name)
    background_cmap = plt.get_cmap(background_cmap_name)
    prepared = []
    max_height = max_width = 0

    for z_mm, index in valid:
        bg = orient_axial(background[:, :, index], display_left_is_left)
        values = orient_axial(roi_values[:, :, index], display_left_is_left)
        alpha = orient_axial(roi_opacity[:, :, index], display_left_is_left)
        skel = orient_axial(skeleton[:, :, index], display_left_is_left)

        brain = np.isfinite(bg) & (bg > 0)
        roi_mask = np.isfinite(values) & (alpha > 0)
        skel_mask = np.isfinite(skel) & (skel > skeleton_threshold)
        visible = brain | roi_mask | skel_mask
        r0, r1, c0, c1 = crop_bounds(visible, crop_pad)
        bg = bg[r0:r1, c0:c1]
        values = values[r0:r1, c0:c1]
        alpha = alpha[r0:r1, c0:c1]
        skel_mask = skel_mask[r0:r1, c0:c1]
        if skeleton_width > 1:
            from scipy.ndimage import binary_dilation
            skel_mask = binary_dilation(skel_mask, iterations=skeleton_width - 1)
        patch = make_slice_rgba(
            bg,
            values,
            alpha,
            skel_mask,
            bg_vmin=bg_vmin,
            bg_vmax=bg_vmax,
            bg_gamma=bg_gamma,
            background_cmap=background_cmap,
            statistic_cmap=cmap,
            norm=norm,
            skeleton_alpha=skeleton_alpha,
        )
        prepared.append((z_mm, patch))
        max_height = max(max_height, patch.shape[0])
        max_width = max(max_width, patch.shape[1])

    step = max(int(round(max_width * (1.0 - slice_overlap))), 1)
    row_width = max_width + step * (images_per_row - 1)
    base_rgb = np.asarray(to_rgba(canvas_color), dtype=np.float32)
    row_canvases = []
    row_labels = []
    for row_index in range(rows):
        canvas = np.zeros((max_height, row_width, 4), dtype=np.float32)
        canvas[..., :3] = base_rgb[:3]
        canvas[..., 3] = base_rgb[3]
        labels = []
        for column in range(images_per_row):
            item_index = row_index * images_per_row + column
            z_mm, patch = prepared[item_index]
            patch = centre_on_canvas(patch, (max_height, max_width), fill=0.0)
            x0 = column * step
            canvas[:, x0 : x0 + max_width] = alpha_over(
                canvas[:, x0 : x0 + max_width], patch
            )
            labels.append((x0 + max_width / 2.0, z_mm))
        row_canvases.append(canvas)
        row_labels.append(labels)

    fig, axes = plt.subplots(rows, 1, figsize=(max(8.0, row_width / 55), 2.55 * rows), squeeze=False)
    axes = axes[:, 0]
    for axis, canvas, labels in zip(axes, row_canvases, row_labels):
        axis.imshow(canvas, interpolation="nearest")
        if show_slice_labels:
            for x, z_mm in labels:
                axis.text(
                    x,
                    max_height - 1,
                    f"z={z_mm:g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                    bbox={"facecolor": canvas_color, "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
                )
        axis.set_xlim(0, row_width)
        axis.set_ylim(max_height, 0)
        axis.axis("off")

    # Reserve a right-hand column for the shared vertical colourbar.  Calling
    # subplots_adjust afterwards would expand the mosaics back over the bar.
    fig.subplots_adjust(
        left=0.015,
        right=0.89,
        top=0.91 if title else 0.97,
        bottom=0.025,
        hspace=0.02,
    )
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=list(axes),
        orientation="vertical",
        fraction=0.035,
        pad=0.025,
        aspect=28,
    )
    colorbar.set_label(colorbar_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=13, y=0.975)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=canvas_color, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_csv_template(path, names):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label_id", "roi_name", "statistic_value", "p_value"))
        for label, name in sorted(names.items()):
            if label:
                writer.writerow((label, name, "", ""))
    print(f"Saved CSV template: {path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Draw a TBSS background + green skeleton + JHU ROI statistic mosaic"
    )
    parser.add_argument("--background", type=Path, help="Background NIfTI image")
    parser.add_argument("--skeleton", type=Path, help="TBSS skeleton NIfTI image")
    parser.add_argument("--stat-csv", type=Path, help="CSV: label_id,statistic_value[,significant|p_value]")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS, help="JHU label atlas NIfTI")
    parser.add_argument("--labels-xml", type=Path, default=DEFAULT_LABELS, help="JHU label-name XML")
    parser.add_argument("--output", type=Path, default=Path("JHU_TBSS_mosaic.png"))
    parser.add_argument("--write-csv-template", type=Path, help="Write a 50-ROI CSV template and exit")
    parser.add_argument("--alpha", type=float, default=0.05, help="p-value threshold (default: 0.05)")
    parser.add_argument("--nonsignificant-opacity", type=float, default=0.25)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument(
        "--z-slices",
        type=parse_number_list,
        help="Explicit MNI z coordinates; overrides --start-z/--slice-step",
    )
    parser.add_argument("--rows", type=int, default=2, help="Number of mosaic rows")
    parser.add_argument("--images-per-row", type=int, default=6, help="Slices in each row")
    parser.add_argument("--start-z", type=float, default=-40.0, help="First MNI z coordinate in mm")
    parser.add_argument("--slice-step", type=float, default=6.0, help="MNI z increment in mm")
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=0.25,
        help="Horizontal fractional overlap between adjacent slices (default: 0.25)",
    )
    parser.add_argument("--skeleton-threshold", type=float, default=1e-6)
    parser.add_argument("--skeleton-alpha", type=float, default=0.95)
    parser.add_argument("--skeleton-width", type=int, default=1, help="Display thickness in pixels")
    parser.add_argument("--bg-vmin", type=float)
    parser.add_argument("--bg-vmax", type=float)
    parser.add_argument("--bg-percentiles", type=parse_number_list, default=[2.0, 99.5])
    parser.add_argument("--bg-gamma", type=float, default=0.75)
    parser.add_argument(
        "--background-cmap",
        default="gray",
        help="Matplotlib colormap for background voxels (e.g. gray, gray_r, bone)",
    )
    parser.add_argument(
        "--canvas-color",
        "--background-color",
        "--outside-brain-color",
        dest="canvas_color",
        default="white",
        help="Zero/background voxel and output colour (e.g. white, black, #f7f7f7)",
    )
    parser.add_argument("--crop-pad", type=int, default=4)
    parser.add_argument("--show-slice-labels", action="store_true")
    parser.add_argument("--radiological", action="store_true", help="Display patient left on image right")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--colorbar-label", default="Statistic")
    parser.add_argument("--title")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 <= args.nonsignificant_opacity <= 1:
        raise ValueError("--nonsignificant-opacity must be between 0 and 1")
    if not 0 <= args.skeleton_alpha <= 1:
        raise ValueError("--skeleton-alpha must be between 0 and 1")
    if args.rows < 1 or args.images_per_row < 1 or args.skeleton_width < 1:
        raise ValueError("--rows, --images-per-row and --skeleton-width must be positive")
    if not 0 <= args.slice_overlap < 1:
        raise ValueError("--slice-overlap must be in [0, 1)")
    if len(args.bg_percentiles) != 2:
        raise ValueError("--bg-percentiles requires two numbers")

    names = load_jhu_names(args.labels_xml)
    if args.write_csv_template:
        write_csv_template(args.write_csv_template, names)
        return 0
    if args.background is None or args.skeleton is None:
        raise ValueError("--background and --skeleton are required for rendering")
    if args.stat_csv is None:
        raise ValueError("--stat-csv is required unless --write-csv-template is used")

    statistics, significance = load_statistical_csv(
        args.stat_csv, label_columns=("label_id",), alpha=args.alpha
    )
    background_image = nib.as_closest_canonical(nib.load(str(args.background)))
    background = np.asarray(background_image.get_fdata(dtype=np.float32))
    atlas = np.rint(load_in_reference(args.atlas, background_image, order=0)).astype(np.int16)
    skeleton = load_in_reference(args.skeleton, background_image, order=0)
    validate_inputs(atlas, statistics, names)
    roi_values, roi_opacity = make_roi_volumes(
        atlas, statistics, significance, args.nonsignificant_opacity
    )
    bg_vmin, bg_vmax = background_limits(
        background,
        args.bg_percentiles[0],
        args.bg_percentiles[1],
        args.bg_vmin,
        args.bg_vmax,
    )
    norm, stat_vmin, stat_vmax = statistic_normalizer(
        statistics, args.vmin, args.vmax
    )
    slice_count = args.rows * args.images_per_row
    z_slices = (
        args.z_slices
        if args.z_slices is not None
        else [args.start_z + index * args.slice_step for index in range(slice_count)]
    )
    if len(z_slices) != slice_count:
        raise ValueError(
            f"--z-slices supplied {len(z_slices)} coordinates, but --rows x "
            f"--images-per-row requires {slice_count}"
        )
    plot_mosaic(
        background_image,
        background,
        skeleton,
        roi_values,
        roi_opacity,
        args.output,
        z_slices=z_slices,
        rows=args.rows,
        images_per_row=args.images_per_row,
        slice_overlap=args.slice_overlap,
        cmap_name=args.cmap,
        norm=norm,
        bg_vmin=bg_vmin,
        bg_vmax=bg_vmax,
        bg_gamma=args.bg_gamma,
        skeleton_threshold=args.skeleton_threshold,
        skeleton_alpha=args.skeleton_alpha,
        skeleton_width=args.skeleton_width,
        crop_pad=args.crop_pad,
        display_left_is_left=not args.radiological,
        dpi=args.dpi,
        colorbar_label=args.colorbar_label,
        canvas_color=args.canvas_color,
        background_cmap_name=args.background_cmap,
        show_slice_labels=args.show_slice_labels,
        title=args.title,
    )
    print(f"Saved: {args.output.resolve()}")
    print(f"Statistic range: {stat_vmin:g} to {stat_vmax:g}")
    print(f"Background range: {bg_vmin:g} to {bg_vmax:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
