"""Combined AAL cortical and subcortical statistical plate.

The output layout follows a compact 2 x 3 publication plate::

    left lateral | left medial | left subcortex in glass brain
    right lateral| right medial| right subcortex in glass brain

One structure-name CSV drives both domains and one colour scale is shared by
all six panels.  PNG output has a true alpha channel (transparent background).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from PIL import Image
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cvdproc.utils.python.visualization.plot_AAL_surface_enigma import (  # noqa: E402
    AALSurfaceENIGMAPlotter,
)
from cvdproc.utils.python.visualization.plot_glass_brain_mesh_ver import (  # noqa: E402
    GlassBrainMeshPlotter,
)
from cvdproc.utils.python.visualization.surface_plotting_utils import (  # noqa: E402
    colour_limits,
    parse_significance,
)


AAL_DIR = REPO_ROOT / "cvdproc/data/atlas/AAL_v4"
DEFAULT_LH_SURFACE = REPO_ROOT / "cvdproc/data/standard/fs_LR_32k/fs_LR.32k.L.inflated.surf.gii"
DEFAULT_RH_SURFACE = REPO_ROOT / "cvdproc/data/standard/fs_LR_32k/fs_LR.32k.R.inflated.surf.gii"
DEFAULT_LH_LABEL = AAL_DIR / "AAL.32k.L.label.gii"
DEFAULT_RH_LABEL = AAL_DIR / "AAL.32k.R.label.gii"
DEFAULT_CORTEX_MAPPING = AAL_DIR / "AAL.32k.surface_label_mapping.csv"
DEFAULT_LH_SUBCORTEX = AAL_DIR / "AAL_subctx_L.surf.gii"
DEFAULT_RH_SUBCORTEX = AAL_DIR / "AAL_subctx_R.surf.gii"
DEFAULT_SUBCORTEX_MAPPING = AAL_DIR / "AAL_ENIGMA_subctx_mapping.csv"
DEFAULT_MNI_BRAIN = REPO_ROOT / "cvdproc/data/standard/MNI152/MNI152_T1_1mm_brain.nii.gz"

VALUE_COLUMNS = ("statistic_value", "value", "stat")
NAME_COLUMNS = ("structure", "aal_name", "roi_name", "name")
SIGNIFICANCE_COLUMNS = ("significant", "is_significant", "significance")
P_COLUMNS = ("p_value", "pvalue", "p_val", "pval", "p")


def first_present(row, candidates):
    return next((column for column in candidates if row.get(column, "") != ""), None)


def load_structure_csv(path, alpha=0.05):
    """Load ``structure -> value`` and optional significance mappings."""
    statistics = {}
    significance = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            row = {
                str(key).strip().lower(): str(value).strip()
                for key, value in raw.items()
                if key is not None
            }
            name_column = first_present(row, NAME_COLUMNS)
            value_column = first_present(row, VALUE_COLUMNS)
            if name_column is None or value_column is None:
                raise ValueError(
                    "CSV requires a structure/name column and a statistic_value/value/stat column"
                )
            name = row[name_column]
            if not name:
                continue
            if name in statistics:
                raise ValueError(f"Duplicate structure '{name}' at CSV line {line_number}")
            statistics[name] = float(row[value_column])
            significance_column = first_present(row, SIGNIFICANCE_COLUMNS)
            p_column = first_present(row, P_COLUMNS)
            if significance_column is not None:
                significance[name] = parse_significance(row[significance_column])
            elif p_column is not None:
                significance[name] = float(row[p_column]) <= alpha
    if not statistics:
        raise ValueError(f"No usable statistic rows in '{path}'")
    return statistics, significance


def load_mappings(cortex_mapping, subcortex_mapping):
    cortex = []
    with Path(cortex_mapping).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label = int(row["surface_label_id"])
            if label:
                cortex.append((row["aal_name"].strip(), row["hemisphere"].strip().upper(), label))
    subcortex = []
    with Path(subcortex_mapping).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            subcortex.append(
                (row["structure"].strip(), row["hemisphere"].strip().upper(), int(row["label_id"]))
            )
    return cortex, subcortex


def write_template(path, cortex_mapping, subcortex_mapping):
    cortex, subcortex = load_mappings(cortex_mapping, subcortex_mapping)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("structure", "domain", "hemisphere", "statistic_value", "p_value"))
        for name, hemisphere, _ in cortex:
            writer.writerow((name, "cortex", hemisphere, "", ""))
        for name, hemisphere, _ in subcortex:
            writer.writerow((name, "subcortex", hemisphere, "", ""))
    print(f"Saved CSV template: {path.resolve()}")


def map_by_domain(statistics, significance, cortex_mapping, subcortex_mapping):
    cortex_rows, subcortex_rows = load_mappings(cortex_mapping, subcortex_mapping)
    known = {name for name, _, _ in cortex_rows + subcortex_rows}
    unknown = sorted(set(statistics) - known)
    if unknown:
        raise ValueError(f"Structures absent from the AAL plotting mappings: {unknown}")

    cortex_stats, cortex_sig = {}, {}
    for name, hemisphere, label in cortex_rows:
        if name in statistics:
            key = (hemisphere, label)
            cortex_stats[key] = statistics[name]
            if name in significance:
                cortex_sig[key] = significance[name]

    subcortex_stats = {"L": {}, "R": {}}
    subcortex_sig = {"L": {}, "R": {}}
    for name, hemisphere, label in subcortex_rows:
        if name in statistics:
            subcortex_stats[hemisphere][label] = statistics[name]
            if name in significance:
                subcortex_sig[hemisphere][label] = significance[name]
    return cortex_stats, cortex_sig, subcortex_stats, subcortex_sig


def combined_limits(statistics, vmin, vmax):
    return colour_limits(
        statistics.values(), vmin, vmax, symmetric_if_diverging=True
    )


def crop_alpha_panel(panel, padding=4):
    """Crop a transparent RGBA panel to its rendered non-empty bounds."""
    alpha = panel[..., 3]
    rows, columns = np.where(alpha > 2)
    if rows.size == 0:
        return panel
    row0 = max(int(rows.min()) - padding, 0)
    row1 = min(int(rows.max()) + padding + 1, panel.shape[0])
    column0 = max(int(columns.min()) - padding, 0)
    column1 = min(int(columns.max()) + padding + 1, panel.shape[1])
    return panel[row0:row1, column0:column1]


def compact_panel_plate(rgba, rows=2, columns=3, gap=0, margin=8):
    """Crop each PyVista viewport and repack it with nearly zero whitespace."""
    source_height, source_width = rgba.shape[:2]
    cell_height = source_height // rows
    cell_width = source_width // columns
    panels = []
    for row in range(rows):
        panel_row = []
        for column in range(columns):
            y0 = row * cell_height
            y1 = source_height if row == rows - 1 else (row + 1) * cell_height
            x0 = column * cell_width
            x1 = source_width if column == columns - 1 else (column + 1) * cell_width
            panel_row.append(crop_alpha_panel(rgba[y0:y1, x0:x1]))
        panels.append(panel_row)

    # Within each column, normalize both hemispheres to the same visible
    # height.  Mirrored L/R panels therefore align without reintroducing the
    # large fixed viewport margins that PyVista uses.
    target_height = max(panel.shape[0] for row in panels for panel in row)
    resized = []
    for row in panels:
        resized_row = []
        for panel in row:
            scale = target_height / panel.shape[0]
            width = max(1, int(round(panel.shape[1] * scale)))
            image = Image.fromarray(panel).resize(
                (width, target_height), Image.Resampling.LANCZOS
            )
            resized_row.append(np.asarray(image))
        resized.append(resized_row)

    column_widths = [
        max(resized[row][column].shape[1] for row in range(rows))
        for column in range(columns)
    ]
    plate_width = 2 * margin + sum(column_widths) + gap * (columns - 1)
    plate_height = 2 * margin + rows * target_height + gap * (rows - 1)
    plate = np.full((plate_height, plate_width, 4), 255, dtype=np.uint8)
    plate[..., 3] = 255
    for row in range(rows):
        x = margin
        y = margin + row * (target_height + gap)
        for column in range(columns):
            panel = resized[row][column]
            offset = (column_widths[column] - panel.shape[1]) // 2
            x0 = x + offset
            alpha = panel[..., 3:4].astype(np.float32) / 255.0
            target = plate[y : y + target_height, x0 : x0 + panel.shape[1], :3]
            target[:] = np.rint(
                panel[..., :3] * alpha + target * (1.0 - alpha)
            ).astype(np.uint8)
            x += column_widths[column] + gap
    return plate


def render_combined(args):
    statistics, significance = load_structure_csv(args.stat_csv, alpha=args.alpha)
    cortex_stats, cortex_sig, sub_stats, sub_sig = map_by_domain(
        statistics,
        significance,
        args.cortex_mapping,
        args.subcortex_mapping,
    )
    if not cortex_stats:
        raise ValueError("The CSV contains no mapped cortical AAL structures")
    if not any(sub_stats.values()):
        raise ValueError("The CSV contains no mapped subcortical AAL structures")
    shared_vmin, shared_vmax = combined_limits(statistics, args.vmin, args.vmax)

    cortex = AALSurfaceENIGMAPlotter(
        colormap=args.cmap,
        bg_color="white",
        nonsignificant_material=args.cortex_nonsignificant_material,
        frosted_opacity=args.frosted_opacity,
    )
    cortex.load_data(
        args.lh_surface,
        args.rh_surface,
        args.lh_label,
        args.rh_label,
        args.cortex_mapping,
        cortex_stats,
        vmin=shared_vmin,
        vmax=shared_vmax,
        significance=cortex_sig,
        nonsignificant_opacity=args.cortex_nonsignificant_opacity,
    )

    sub_loader = GlassBrainMeshPlotter(colormap=args.cmap)
    built_subcortex = {}
    for hemisphere, mesh_path in (("L", args.lh_subcortex), ("R", args.rh_subcortex)):
        config = {
            "mesh_path": str(mesh_path),
            "stat_values": sub_stats[hemisphere],
            "significance": sub_sig[hemisphere],
            "nonsignificant_opacity": args.subcortex_nonsignificant_opacity,
            "vmin": shared_vmin,
            "vmax": shared_vmax,
            "cmap": cortex._display_colormap(),
        }
        built_subcortex[hemisphere] = sub_loader._build_scalar_mesh(config)[0]

    glass = GlassBrainMeshPlotter._load_background_mesh(
        args.glass_brain, brain_iso_value=args.brain_iso_value
    )
    bounds = np.asarray(glass.bounds, dtype=float)
    lower = bounds[[0, 2, 4]]
    upper = bounds[[1, 3, 5]]
    glass_focal = (lower + upper) / 2.0
    glass_extent = float(np.max(upper - lower))

    plotter = pv.Plotter(
        off_screen=True,
        shape=(2, 3),
        border=False,
        window_size=(args.width, args.height),
    )
    plotter.set_background("white")
    plotter.enable_anti_aliasing()
    plotter.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.0)
    last_actor = None

    for row, hemisphere in enumerate(("L", "R")):
        base = cortex._surface_meshes[hemisphere]
        overlay = cortex._overlay_meshes[hemisphere]
        for column, view in enumerate(("lateral", "medial")):
            plotter.subplot(row, column)
            cortex._add_base_mesh(plotter, base)
            last_actor = cortex._add_colored_mesh(plotter, overlay, show_scalar_bar=False)
            cortex._add_nonsignificant_veil(
                plotter, cortex._nonsignificant_veil_meshes[hemisphere]
            )
            cortex._add_medial_wall_mesh(plotter, cortex._medial_wall_meshes[hemisphere])
            plotter.camera_position = cortex._camera_for_view(
                hemisphere, view, base.center
            )
            plotter.reset_camera_clipping_range()
            plotter.camera.zoom(args.cortex_zoom)

        plotter.subplot(row, 2)
        plotter.add_mesh(
            glass,
            color=args.glass_color,
            opacity=args.glass_opacity,
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.65,
            specular=0.08,
            specular_power=8,
            show_scalar_bar=False,
        )
        submesh = built_subcortex[hemisphere]
        last_actor = plotter.add_mesh(
            submesh,
            scalars="stat_value",
            opacity=submesh["plot_opacity"],
            cmap=cortex._display_colormap(),
            clim=(shared_vmin, shared_vmax),
            smooth_shading=True,
            ambient=0.20,
            diffuse=0.78,
            specular=0.22,
            specular_power=18,
            show_scalar_bar=False,
        )
        direction = np.array((-1.0, 0.0, 0.0) if hemisphere == "L" else (1.0, 0.0, 0.0))
        camera = glass_focal + direction * glass_extent * 3.2
        plotter.camera_position = [tuple(camera), tuple(glass_focal), (0, 0, 1)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = glass_extent * args.glass_scale
        plotter.reset_camera_clipping_range()

    rgba = plotter.screenshot(return_img=True, transparent_background=True)
    plotter.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plate = compact_panel_plate(
        rgba, gap=args.panel_gap, margin=args.panel_margin
    )

    # Compose on an opaque white figure and reserve a dedicated right-hand
    # band for the colourbar.  It can never overlap the subcortical panels.
    final_width = plate.shape[1] + args.colorbar_space
    final_height = plate.shape[0]
    figure = plt.figure(
        figsize=(final_width / args.dpi, final_height / args.dpi),
        dpi=args.dpi,
        facecolor="white",
    )
    image_fraction = plate.shape[1] / final_width
    image_axis = figure.add_axes((0.0, 0.0, image_fraction, 1.0))
    image_axis.imshow(plate)
    image_axis.axis("off")
    colorbar_x = image_fraction + (1.0 - image_fraction) * 0.22
    colorbar_width = (1.0 - image_fraction) * 0.16
    colorbar_axis = figure.add_axes((colorbar_x, 0.17, colorbar_width, 0.66))
    scalar = ScalarMappable(
        norm=Normalize(vmin=shared_vmin, vmax=shared_vmax),
        cmap=cortex._display_colormap(),
    )
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, cax=colorbar_axis, orientation="vertical")
    colorbar.ax.set_title(args.colorbar_label, fontsize=12, pad=8)
    colorbar.ax.tick_params(labelsize=10, width=0.8, length=4)
    colorbar.outline.set_linewidth(0.8)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        transparent=False,
        facecolor="white",
        edgecolor="white",
    )
    plt.close(figure)
    print(f"Saved: {args.output.resolve()}")
    print(f"Shared statistic range: {shared_vmin:g} to {shared_vmax:g}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Combined AAL cortex + subcortex 2x3 statistical figure"
    )
    parser.add_argument("--stat-csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("AAL_combined.png"))
    parser.add_argument("--write-csv-template", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--colorbar-label", default="Statistic")
    parser.add_argument("--cortex-nonsignificant-opacity", type=float, default=0.08)
    parser.add_argument("--subcortex-nonsignificant-opacity", type=float, default=0.25)
    parser.add_argument("--cortex-nonsignificant-material", choices=("frosted", "flat"), default="frosted")
    parser.add_argument("--frosted-opacity", type=float, default=0.32)
    parser.add_argument(
        "--glass-opacity",
        type=float,
        default=0.18,
        help="Opacity of the anatomical glass-brain shell (default: 0.18)",
    )
    parser.add_argument(
        "--glass-color",
        default="#c9ced3",
        help="Glass-brain shell colour (default: a visible neutral blue-grey)",
    )
    parser.add_argument("--brain-iso-value", type=float, default=4750.0)
    parser.add_argument(
        "--cortex-zoom", type=float, default=1.62,
        help="Cortical panel magnification; larger values reduce inter-panel whitespace",
    )
    parser.add_argument(
        "--glass-scale", type=float, default=0.43,
        help="Glass-brain parallel camera scale; smaller values enlarge the panel",
    )
    parser.add_argument("--width", type=int, default=4800)
    parser.add_argument("--height", type=int, default=2200)
    parser.add_argument(
        "--panel-gap", type=int, default=0,
        help="Pixel gap between tightly cropped panels (default: 0)",
    )
    parser.add_argument("--panel-margin", type=int, default=6)
    parser.add_argument(
        "--colorbar-space", type=int, default=320,
        help="Dedicated white pixel width reserved to the right of all panels",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--lh-surface", type=Path, default=DEFAULT_LH_SURFACE)
    parser.add_argument("--rh-surface", type=Path, default=DEFAULT_RH_SURFACE)
    parser.add_argument("--lh-label", type=Path, default=DEFAULT_LH_LABEL)
    parser.add_argument("--rh-label", type=Path, default=DEFAULT_RH_LABEL)
    parser.add_argument("--cortex-mapping", type=Path, default=DEFAULT_CORTEX_MAPPING)
    parser.add_argument("--lh-subcortex", type=Path, default=DEFAULT_LH_SUBCORTEX)
    parser.add_argument("--rh-subcortex", type=Path, default=DEFAULT_RH_SUBCORTEX)
    parser.add_argument("--subcortex-mapping", type=Path, default=DEFAULT_SUBCORTEX_MAPPING)
    parser.add_argument("--glass-brain", type=Path, default=DEFAULT_MNI_BRAIN)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.write_csv_template:
        write_template(
            args.write_csv_template, args.cortex_mapping, args.subcortex_mapping
        )
        return 0
    if args.stat_csv is None:
        raise ValueError("--stat-csv is required unless --write-csv-template is used")
    for name in (
        "cortex_nonsignificant_opacity",
        "subcortex_nonsignificant_opacity",
        "frosted_opacity",
        "glass_opacity",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.panel_gap < 0 or args.panel_margin < 0 or args.colorbar_space < 1:
        raise ValueError(
            "--panel-gap and --panel-margin must be non-negative; "
            "--colorbar-space must be positive"
        )
    render_combined(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
