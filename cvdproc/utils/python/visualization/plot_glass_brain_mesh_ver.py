import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cvdproc.utils.python.visualization.surface_plotting_utils import (  # noqa: E402
    colour_limits,
    load_gifti_surface,
    load_statistical_csv,
    parse_label_value_pairs,
    parse_significance,
    pyvista_faces,
    vertex_statistic_and_opacity,
)
from cvdproc.utils.python.visualization.plot_JHU_glass_brain import (  # noqa: E402
    brain_surface,
    smooth_brain_poly,
)

DEFAULT_MNI_BRAIN = (
    REPO_ROOT / "cvdproc" / "data" / "standard" / "MNI152"
    / "MNI152_T1_1mm_brain.nii.gz"
)


class GlassBrainMeshPlotter:
    """Glass-brain visualization using surface meshes with per-ROI statistic overlay.

    Renders an optional background mesh (e.g. inflated brain) as semi-transparent
    context, then overlays a subcortical mesh whose per-vertex labels are mapped to
    user-supplied ROI statistic values and colored via a matplotlib/pyvista colormap
    with a scalar bar. Outputs 6 standard views.

    Parameters
    ----------
    background_mesh : str or None
        Path to background GIFTI mesh (e.g. inflated surface).
    background_opacity : float
        Opacity of the background mesh.
    background_color : str or tuple
        Color for the background mesh.
    colormap : str
        Matplotlib / PyVista colormap name.
    output_dir : str
        Directory for output PNGs.
    window_size : tuple
        PyVista render window size (width, height).
    """

    VIEW_PRESETS = {
        "anterior":  ((0, 1, 0),  (0, 0, 1)),
        "posterior": ((0, -1, 0), (0, 0, 1)),
        "right":     ((1, 0, 0),  (0, 0, 1)),
        "left":      ((-1, 0, 0), (0, 0, 1)),
        "superior":  ((0, 0, 1),  (0, 1, 0)),
        "inferior":  ((0, 0, -1), (0, 1, 0)),
    }
    VIEW_ORDER = ("anterior", "posterior", "left", "right", "superior", "inferior")

    def __init__(
        self,
        background_mesh=DEFAULT_MNI_BRAIN,
        background_opacity=0.12,
        background_color=(0.827, 0.827, 0.827),
        colormap="coolwarm",
        output_dir=None,
        window_size=(4800, 3000),
        dpi=600,
    ):
        self.background_mesh = background_mesh
        self.background_opacity = background_opacity
        self.background_color = background_color
        self.colormap = colormap
        self.output_dir = output_dir or os.getcwd()
        self.window_size = window_size
        self.dpi = dpi
        self.subctx_configs = []

    @staticmethod
    def _load_gii_mesh(path):
        """Load a GIFTI surface and return PyVista PolyData."""
        verts, faces_full, _ = load_gifti_surface(path)
        mesh = pv.PolyData(verts, pyvista_faces(faces_full))
        mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
        return mesh

    @staticmethod
    def _load_background_mesh(path, brain_iso_value=4750.0):
        """Load a GIFTI mesh or JHU-matched MNI152 NIfTI glass brain."""
        suffixes = "".join(Path(path).suffixes).lower()
        if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
            import nibabel as nib

            # These are the same MNI152 isosurface and smoothing operations
            # used by plot_JHU_glass_brain.py.
            points, faces, _ = brain_surface(nib.load(path), iso_value=brain_iso_value)
            return pv.wrap(smooth_brain_poly(points, faces))
        return GlassBrainMeshPlotter._load_gii_mesh(path)

    def add_subctx(
        self,
        mesh_path,
        stat_values,
        vmin=None,
        vmax=None,
        cmap=None,
        significance=None,
        nonsignificant_opacity=0.35,
    ):
        """Register a subcortical mesh with per‑ROI statistic values.

        Parameters
        ----------
        mesh_path : str
            Path to a GIFTI mesh whose label data array maps each vertex
            to an integer region ID.
        stat_values : dict
            Mapping from label ID (int) to statistic value (float).
            Example: {1: 1.2, 3: 1.4, 5: 1.2, 7: 2.1}
        vmin, vmax : float or None
            Clamp the colormap range. If None, inferred from stat_values.
        cmap : str or None
            Colormap override for this subctx layer.
        significance : dict or None
            Optional label-to-boolean mapping. Omitted labels are significant.
        nonsignificant_opacity : float
            Opacity assigned to non-significant regions.
        """
        self.subctx_configs.append({
            "mesh_path": mesh_path,
            "stat_values": stat_values,
            "significance": significance,
            "nonsignificant_opacity": nonsignificant_opacity,
            "vmin": vmin,
            "vmax": vmax,
            "cmap": cmap or self.colormap,
        })

    def _build_scalar_mesh(self, config):
        """Load labeled GIFTI and produce a mesh with per‑vertex statistic scalars."""
        verts, faces_full, label_data = load_gifti_surface(
            config["mesh_path"], require_labels=True
        )
        faces_pv = pyvista_faces(faces_full)

        stat_values = config["stat_values"]
        vmin, vmax = colour_limits(stat_values.values(), config["vmin"], config["vmax"])
        scalar_data, opacity_data = vertex_statistic_and_opacity(
            label_data,
            stat_values,
            config["significance"],
            nonsignificant_opacity=config["nonsignificant_opacity"],
        )

        mesh = pv.PolyData(verts, faces_pv)
        mesh["stat_value"] = scalar_data
        mesh["plot_opacity"] = opacity_data
        mesh.set_active_scalars("stat_value")
        mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

        return mesh, vmin, vmax

    def render_views(self, views=None, prefix="view", scalar_bar_label="Statistic"):
        """Render the requested views into one 2 x 3 PNG plate.

        Parameters
        ----------
        views : list of str or None
            Subset of VIEW_PRESETS keys. If None, renders all 6.
        prefix : str
            Filename prefix for saved PNGs.
        scalar_bar_label : str
            Title shown on the scalar bar.
        """
        if views is None:
            views = list(self.VIEW_ORDER)
        invalid = [view for view in views if view not in self.VIEW_PRESETS]
        if invalid:
            raise ValueError(f"Unknown views: {invalid}")
        if len(views) != 6:
            raise ValueError("The combined plate requires exactly six views")
        if not self.subctx_configs:
            raise ValueError("No subcortical mesh has been added")

        os.makedirs(self.output_dir, exist_ok=True)
        built = [
            (config, *self._build_scalar_mesh(config))
            for config in self.subctx_configs
        ]
        plotted_meshes = [item[1] for item in built]
        shared_vmin = min(item[2] for item in built)
        shared_vmax = max(item[3] for item in built)
        bg_mesh = None
        if self.background_mesh and os.path.exists(self.background_mesh):
            bg_mesh = self._load_background_mesh(self.background_mesh)
        # Camera framing must include the glass-brain shell, not merely the
        # small subcortical overlay.  Otherwise the viewing point can end up
        # inside the translucent background mesh.
        framing_meshes = plotted_meshes + ([bg_mesh] if bg_mesh is not None else [])
        bounds = np.asarray([mesh.bounds for mesh in framing_meshes], dtype=float)
        lower = bounds[:, [0, 2, 4]].min(axis=0)
        upper = bounds[:, [1, 3, 5]].max(axis=0)
        focal_point = (lower + upper) / 2.0
        extent = float(np.max(upper - lower))

        plotter = pv.Plotter(
            off_screen=True,
            window_size=self.window_size,
            shape=(2, 3),
            border=False,
        )
        plotter.set_background("white")
        plotter.enable_anti_aliasing()
        plotter.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.0)
        last_actor = None

        for index, view_name in enumerate(views):
            plotter.subplot(index // 3, index % 3)
            view_direction, view_up = self.VIEW_PRESETS[view_name]
            if bg_mesh is not None:
                plotter.add_mesh(
                    bg_mesh,
                    color=self.background_color,
                    opacity=self.background_opacity,
                    smooth_shading=True,
                    specular=0.1,
                    specular_power=10,
                )
            for config, labeled_mesh, _, _ in built:
                last_actor = plotter.add_mesh(
                    labeled_mesh,
                    scalars="stat_value",
                    opacity=labeled_mesh["plot_opacity"],
                    cmap=config["cmap"],
                    clim=[shared_vmin, shared_vmax],
                    smooth_shading=True,
                    specular=0.35,
                    specular_power=20,
                    show_scalar_bar=False,
                )
            camera_pos = (
                focal_point + np.asarray(view_direction, dtype=float) * max(extent * 3.2, 1.0)
            )
            plotter.reset_camera()
            plotter.camera_position = [tuple(camera_pos), tuple(focal_point), view_up]
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = extent * 0.58
            plotter.reset_camera_clipping_range()

        plotter.add_scalar_bar(
            title=scalar_bar_label,
            mapper=last_actor.mapper,
            vertical=True,
            position_x=0.94,
            position_y=0.16,
            width=0.025,
            height=0.62,
            title_font_size=20,
            label_font_size=14,
            fmt="%.2f",
            color="black",
        )
        image = plotter.screenshot(return_img=True)
        out_path = os.path.join(self.output_dir, f"{prefix}_6views.png")
        Image.fromarray(image).save(out_path, dpi=(self.dpi, self.dpi))
        plotter.close()
        print(f"Saved: {out_path}")


def load_stat_values_from_csv(mesh_path, csv_path):
    """Load stat values from a mapping CSV.

    The CSV is expected to have columns: mesh_file, label_id, statistic_value.
    Returns a dict {label_id: statistic_value}.
    """
    stat_values, _ = load_statistical_csv(csv_path, mesh_name=mesh_path)
    return stat_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glass brain mesh plotter with per-ROI statistic overlay")
    parser.add_argument(
        "--background", type=str, default=str(DEFAULT_MNI_BRAIN),
        help="Background GIFTI mesh or MNI152 T1 NIfTI (JHU-matched glass brain)",
    )
    parser.add_argument(
        "--subctx",
        type=str,
        action="append",
        required=True,
        help="Subcortical GIFTI mesh path with label data; repeat to overlay multiple meshes",
    )
    parser.add_argument("--stat_values", type=str, default=None,
                        help="Comma-separated label:value pairs, e.g. '1:1.2,3:1.4,5:1.2,7:2.1'")
    parser.add_argument("--stat_csv", type=str, default=None,
                        help="CSV with columns: mesh_file,label_id,statistic_value")
    parser.add_argument(
        "--significance_values",
        type=str,
        default=None,
        help="Optional label_id:flag pairs; flags accept 1/0, true/false, yes/no",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--nonsignificant_opacity", type=float, default=0.35)
    parser.add_argument("--colormap", type=str, default="coolwarm")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--bg_opacity", type=float, default=0.12)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--prefix", type=str, default="subctx_glass")
    parser.add_argument("--bar_label", type=str, default="Statistic")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    plotter = GlassBrainMeshPlotter(
        background_mesh=args.background,
        background_opacity=args.bg_opacity,
        colormap=args.colormap,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    for subctx_path in args.subctx:
        # Each mesh loads its own labels and statistics.  This makes the
        # command-line interface consistent with the class API and permits
        # bilateral AAL subcortical figures in a single render.
        if args.stat_values:
            stat_values = parse_label_value_pairs(args.stat_values, float)
            significance = {}
        elif args.stat_csv and os.path.exists(args.stat_csv):
            stat_values, significance = load_statistical_csv(
                args.stat_csv, mesh_name=subctx_path, alpha=args.alpha
            )
        else:
            stat_values = {1: 1.2, 3: 1.4, 5: 1.2, 7: 2.1}
            significance = {}
            print(f"Using default stat values: {stat_values}")

        if args.significance_values:
            significance.update(
                parse_label_value_pairs(args.significance_values, parse_significance)
            )

        plotter.add_subctx(
            mesh_path=subctx_path,
            stat_values=stat_values,
            significance=significance,
            nonsignificant_opacity=args.nonsignificant_opacity,
            vmin=args.vmin, vmax=args.vmax,
        )

    plotter.render_views(prefix=args.prefix, scalar_bar_label=args.bar_label)
    print("Done.")
