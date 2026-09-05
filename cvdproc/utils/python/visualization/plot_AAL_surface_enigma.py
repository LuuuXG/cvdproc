"""
ENIGMA-style cortical surface plotter for AAL atlas on fs_LR 32k space.

Loads fs_LR 32k inflated surfaces and AAL 32k label files, maps per-ROI
statistic values to vertex-level scalars, and renders publication-quality
views with ENIGMA-like matte directional lighting and medial wall masked in
white.  It produces one 2 x 3 (six-view) plate per hemisphere.

Usage
-----
python plot_AAL_surface_enigma.py \
    --lh_inflated cvdproc/data/standard/fs_LR_32k/fs_LR.32k.L.inflated.surf.gii \
    --rh_inflated cvdproc/data/standard/fs_LR_32k/fs_LR.32k.R.inflated.surf.gii \
    --lh_label cvdproc/data/atlas/AAL_v4/AAL.32k.L.label.gii \
    --rh_label cvdproc/data/atlas/AAL_v4/AAL.32k.R.label.gii \
    --mapping_csv cvdproc/data/atlas/AAL_v4/AAL.32k.surface_label_mapping.csv \
    --stat_values "1:0.8,2:1.2,3:-0.5,4:1.5,..." \
    --significance_values "1:1,2:0,3:1,4:0,..." \
    --colormap RdBu_r --output_dir /mnt/e/Neuroimage/workdir/AAL_mesh
"""

import os
import csv
import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from matplotlib import colormaps, colors

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cvdproc.utils.python.visualization.surface_plotting_utils import (  # noqa: E402
    colour_limits,
    load_gifti_labels,
    load_gifti_surface,
    load_statistical_csv,
    parse_label_value_pairs,
    parse_significance,
    pyvista_faces,
    vertex_statistic_and_opacity,
)

# Six standard views shown as a 2 x 3 plate for each hemisphere.
VIEW_ORDER = ("lateral", "medial", "anterior", "posterior", "superior", "inferior")


class AALSurfaceENIGMAPlotter:
    """ENIGMA-style cortical surface plotter.

    Parameters
    ----------
    colormap : str
        Matplotlib / PyVista colormap name.
    bg_color : str
        Background color for the render window.
    nan_color : str
        Color for medial wall (NaN) vertices. Typically white.
    specular : float
        Specular intensity. A low value avoids a plastic / creamy appearance.
    specular_power : float
        Specular power (shininess).
    ambient : float
        Ambient light level.
    output_dir : str
        Directory for output PNGs.
    window_size : tuple
        Render window dimensions.
    """

    def __init__(
        self,
        colormap="RdBu_r",
        bg_color="white",
        nan_color="white",
        base_color=(0.965, 0.955, 0.925),
        color_saturation=0.82,
        color_lightness=0.06,
        nonsignificant_material="frosted",
        frosted_color=(0.94, 0.975, 1.0),
        frosted_opacity=0.32,
        specular=0.08,
        specular_power=10,
        ambient=0.16,
        output_dir=None,
        window_size=(4800, 3000),
        dpi=600,
    ):
        self.colormap = colormap
        self.bg_color = bg_color
        self.nan_color = nan_color
        self.base_color = base_color
        self.color_saturation = color_saturation
        self.color_lightness = color_lightness
        self.nonsignificant_material = nonsignificant_material
        self.frosted_color = frosted_color
        self.frosted_opacity = frosted_opacity
        self.specular = specular
        self.specular_power = specular_power
        self.ambient = ambient
        self.output_dir = output_dir or os.getcwd()
        self.window_size = window_size
        self.dpi = dpi

        self._surface_meshes = {}
        self._overlay_meshes = {}
        self._nonsignificant_veil_meshes = {}
        self._medial_wall_meshes = {}
        self._vmin = None
        self._vmax = None

    def _display_colormap(self):
        """Return a slightly softened red-blue map for cortical rendering."""
        source = colormaps.get_cmap(self.colormap).resampled(256)
        rgba = source(np.linspace(0.0, 1.0, 256))
        rgb = rgba[:, :3]
        luminance = np.sum(rgb * np.array([0.2126, 0.7152, 0.0722]), axis=1, keepdims=True)
        rgb = luminance + self.color_saturation * (rgb - luminance)
        rgb = rgb + self.color_lightness * (1.0 - rgb)
        rgba[:, :3] = np.clip(rgb, 0.0, 1.0)
        return colors.ListedColormap(rgba, name=f"{self.colormap}_soft")

    def load_data(
        self,
        lh_inflated_path,
        rh_inflated_path,
        lh_label_path,
        rh_label_path,
        mapping_csv,
        stat_values,
        vmin=None,
        vmax=None,
        significance=None,
        nonsignificant_opacity=0.15,
    ):
        """Load surfaces and labels, build per-vertex scalar arrays.

        Parameters
        ----------
        lh_inflated_path, rh_inflated_path : str
            Paths to L/R inflated surface GIFTI files.
        lh_label_path, rh_label_path : str
            Paths to L/R AAL 32k label GIFTI files (label 0 = MedialWall).
        mapping_csv : str
            CSV with columns: hemisphere, surface_label_id, aal_name.
        stat_values : dict
            Mapping from ``surface_label_id`` to a value shared across both
            hemispheres, or from ``(hemisphere, surface_label_id)`` to a
            hemisphere-specific value. MedialWall (0) is handled automatically.
        vmin, vmax : float or None
            Colormap range. If None, inferred from stat_values.
        significance : dict or None
            Optional label-to-boolean mapping. Missing labels and ``None`` are
            treated as significant.
        nonsignificant_opacity : float
            Opacity assigned to regions marked non-significant.
        """
        # Build surface-label-id -> stat-value lookup from stat_values
        # stat_values keys match surface_label_id (1-41)
        self._vmin, self._vmax = colour_limits(stat_values.values(), vmin, vmax)

        for hemi, inflated_path, label_path in [
            ("L", lh_inflated_path, lh_label_path),
            ("R", rh_inflated_path, rh_label_path),
        ]:
            # Load inflated surface
            verts, faces_full, _ = load_gifti_surface(inflated_path)
            faces_pv = pyvista_faces(faces_full)

            # Load AAL label data
            label_data = load_gifti_labels(label_path)
            if len(label_data) != len(verts):
                raise ValueError(
                    f"Surface and label vertex counts differ for hemisphere {hemi}: "
                    f"{len(verts)} != {len(label_data)}"
                )

            # Per-vertex scalar: NaN for medial wall, stat value for cortical regions
            scalar_data, opacity_data = vertex_statistic_and_opacity(
                label_data,
                stat_values,
                significance,
                hemisphere=hemi,
                nonsignificant_opacity=nonsignificant_opacity,
            )

            mesh = pv.PolyData(verts, faces_pv)
            mesh["stat_value"] = scalar_data
            mesh.set_active_scalars("stat_value")
            # Make neighboring normals consistent before smooth shading.  Do
            # not auto-orient: a cortical hemisphere has an open medial-wall
            # boundary, for which global auto-orientation is not reliable.
            mesh.compute_normals(
                cell_normals=False,
                point_normals=True,
                consistent_normals=True,
                non_manifold_traversal=False,
                inplace=True,
            )
            self._surface_meshes[hemi] = mesh

            # The statistical overlay may be translucent, but the cortical
            # geometry itself remains opaque.  This prevents the far side of
            # an inflated hemisphere from showing through a non-significant
            # parcel.
            overlay = mesh.copy(deep=True)
            overlay_opacity = opacity_data.copy()
            overlay_opacity[label_data == 0] = 0.0
            overlay["plot_opacity"] = overlay_opacity
            normals = np.asarray(overlay.point_normals)
            normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.divide(normals, normal_length, out=np.zeros_like(normals), where=normal_length > 0)
            overlay.points = overlay.points + normals * 0.12
            self._overlay_meshes[hemi] = overlay

            # A low-alpha colour painted directly over an opaque white cortex
            # merely looks desaturated.  For non-significant parcels, add a
            # second, slightly elevated translucent shell with a glossy/frosted
            # material.  The opaque cortex below still hides the far side of
            # the hemisphere, while the separated shell supplies highlights
            # and visual depth that read as translucency.
            nonsignificant = (opacity_data < 1.0) & (label_data != 0)
            veil_faces = faces_full[np.all(nonsignificant[faces_full], axis=1)]
            veil = pv.PolyData(verts, pyvista_faces(veil_faces))
            if veil.n_cells:
                veil.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    consistent_normals=True,
                    non_manifold_traversal=False,
                    inplace=True,
                )
                veil_normals = np.asarray(veil.point_normals)
                veil_length = np.linalg.norm(veil_normals, axis=1, keepdims=True)
                veil_normals = np.divide(
                    veil_normals,
                    veil_length,
                    out=np.zeros_like(veil_normals),
                    where=veil_length > 0,
                )
                veil.points = veil.points + veil_normals * 0.22
            self._nonsignificant_veil_meshes[hemi] = veil

            # Render the medial wall as its own opaque white mesh.  Relying
            # only on NaN colours is unreliable when a per-vertex opacity
            # array is also supplied to PyVista: the NaN wall may appear
            # translucent or inherit neighbouring parcel colours.
            wall_faces = faces_full[np.all(label_data[faces_full] == 0, axis=1)]
            medial_wall = pv.PolyData(verts, pyvista_faces(wall_faces))
            medial_wall.compute_normals(
                cell_normals=False,
                point_normals=True,
                consistent_normals=True,
                non_manifold_traversal=False,
                inplace=True,
            )
            self._medial_wall_meshes[hemi] = medial_wall

    @staticmethod
    def _camera_for_view(hemi, view_name, focal_point, distance=500):
        """Return a camera triple centred on one hemisphere."""
        lateral_sign = -1 if hemi == "L" else 1
        directions = {
            "lateral": (lateral_sign, 0, 0),
            "medial": (-lateral_sign, 0, 0),
            "anterior": (0, 1, 0),
            "posterior": (0, -1, 0),
            "superior": (0, 0, 1),
            "inferior": (0, 0, -1),
        }
        view_up = (0, 1, 0) if view_name in ("superior", "inferior") else (0, 0, 1)
        direction = np.asarray(directions[view_name], dtype=float)
        focal = np.asarray(focal_point, dtype=float)
        return [tuple(focal + distance * direction), tuple(focal), view_up]

    def _add_base_mesh(self, plotter, mesh):
        """Add the opaque cortical geometry below the statistical overlay."""
        plotter.add_mesh(
            mesh,
            color=self.base_color,
            opacity=1.0,
            smooth_shading=True,
            specular=0.015,
            specular_power=4,
            ambient=0.42,
            diffuse=0.66,
            show_scalar_bar=False,
        )

    def _add_colored_mesh(self, plotter, mesh, show_scalar_bar=False, scalar_bar_label="Effect size"):
        return plotter.add_mesh(
            mesh,
            scalars="stat_value",
            opacity=mesh["plot_opacity"],
            cmap=self._display_colormap(),
            clim=[self._vmin, self._vmax],
            nan_color=self.nan_color,
            smooth_shading=True,
            # Matte, directional shading is closer to the ENIGMA plates than
            # the previous bright glossy material.
            specular=self.specular,
            specular_power=self.specular_power,
            ambient=self.ambient,
            diffuse=0.88,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={
                "title": scalar_bar_label,
                "position_x": 0.92,
                "position_y": 0.16,
                "width": 0.025,
                "height": 0.62,
                "title_font_size": 20,
                "label_font_size": 14,
                "fmt": "%.2f",
                "color": "k",
            },
        )

    def _add_nonsignificant_veil(self, plotter, mesh):
        """Add a pearly translucent coat over non-significant parcels."""
        if self.nonsignificant_material != "frosted" or not mesh.n_cells:
            return None
        return plotter.add_mesh(
            mesh,
            color=self.frosted_color,
            opacity=self.frosted_opacity,
            smooth_shading=True,
            interpolation="phong",
            ambient=0.22,
            diffuse=0.46,
            specular=0.72,
            specular_power=42,
            show_scalar_bar=False,
        )

    def _add_medial_wall_mesh(self, plotter, mesh):
        """Add the label-0 medial wall as an opaque white surface."""
        plotter.add_mesh(
            mesh,
            color=self.nan_color,
            opacity=1.0,
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.82,
            specular=0.0,
            show_scalar_bar=False,
        )

    def render_hemisphere_grids(self, prefix="aal_surface", scalar_bar_label="Effect size"):
        """Save one 4 x 3 plate containing six views of both hemispheres."""
        os.makedirs(self.output_dir, exist_ok=True)
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(self.window_size[0], self.window_size[1] * 2),
            shape=(4, 3),
            border=False,
        )
        plotter.set_background(self.bg_color)
        plotter.enable_anti_aliasing()
        plotter.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.0)
        last_actor = None

        for hemi_index, hemi in enumerate(("L", "R")):
            base_mesh = self._surface_meshes[hemi]
            overlay_mesh = self._overlay_meshes[hemi]
            for index, view_name in enumerate(VIEW_ORDER):
                row = hemi_index * 2 + index // 3
                column = index % 3
                plotter.subplot(row, column)
                self._add_base_mesh(plotter, base_mesh)
                last_actor = self._add_colored_mesh(plotter, overlay_mesh, show_scalar_bar=False)
                self._add_nonsignificant_veil(
                    plotter, self._nonsignificant_veil_meshes[hemi]
                )
                self._add_medial_wall_mesh(plotter, self._medial_wall_meshes[hemi])
                plotter.camera_position = self._camera_for_view(hemi, view_name, base_mesh.center)
                plotter.reset_camera_clipping_range()
                plotter.camera.zoom(1.15)
                plotter.add_text(
                    f"{hemi} · {view_name.title()}",
                    position="lower_edge",
                    font_size=10,
                    color="black",
                )

        # One colour scale is shared by all twelve panels.
        plotter.add_scalar_bar(
            title=scalar_bar_label,
            mapper=last_actor.mapper,
            vertical=True,
            position_x=0.94,
            position_y=0.20,
            width=0.022,
            height=0.60,
            title_font_size=20,
            label_font_size=14,
            fmt="%.2f",
            color="k",
        )
        image = plotter.screenshot(return_img=True)
        out_path = os.path.join(self.output_dir, f"{prefix}_LR_12views.png")
        Image.fromarray(image).save(out_path, dpi=(self.dpi, self.dpi))
        plotter.close()
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stat_values_from_csv(csv_path):
    """Read ``surface_label_id, statistic_value`` from a CSV.

    An optional ``hemisphere`` column (``L``/``R``) supplies separate values
    for the two sides; rows without it apply to both hemispheres.
    """
    stat_values, _ = load_statistical_csv(
        csv_path, label_columns=("surface_label_id", "label_id")
    )
    return stat_values


def generate_random_test_stats(mapping_csv, seed=42):
    """Generate random statistic values for testing.

    Reads the surface_label_mapping CSV to discover valid label IDs (excluding
    MedialWall=0), then assigns each a random value drawn from a normal
    distribution.
    """
    label_ids = set()
    with open(mapping_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = int(row["surface_label_id"].strip())
            if lid != 0:
                label_ids.add(lid)

    rng = np.random.default_rng(seed)
    stat_values = {}
    for lid in sorted(label_ids):
        stat_values[lid] = round(float(rng.normal(0.0, 1.0)), 3)
    return stat_values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ENIGMA-style AAL cortical surface plotter (fs_LR 32k)"
    )
    parser.add_argument("--lh_inflated", type=str, required=True, help="Left inflated surface GIFTI")
    parser.add_argument("--rh_inflated", type=str, required=True, help="Right inflated surface GIFTI")
    parser.add_argument("--lh_label", type=str, required=True, help="Left AAL 32k label GIFTI")
    parser.add_argument("--rh_label", type=str, required=True, help="Right AAL 32k label GIFTI")
    parser.add_argument("--mapping_csv", type=str, default=None, help="surface_label_mapping CSV")
    parser.add_argument("--stat_values", type=str, default=None,
                        help="Comma-separated label_id:value pairs, e.g. '1:0.8,2:1.2,...'")
    parser.add_argument("--stat_csv", type=str, default=None, help="CSV with surface_label_id,statistic_value")
    parser.add_argument(
        "--significance_values",
        type=str,
        default=None,
        help="Optional label_id:flag pairs; flags accept 1/0, true/false, yes/no",
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="p-value threshold when stat CSV contains p_value (default: 0.05)")
    parser.add_argument("--nonsignificant_opacity", type=float, default=0.15)
    parser.add_argument(
        "--nonsignificant_material",
        choices=("frosted", "flat"),
        default="frosted",
        help="Material for non-significant cortex: frosted adds a translucent pearly coat",
    )
    parser.add_argument(
        "--frosted_opacity",
        type=float,
        default=0.32,
        help="Opacity of the frosted highlight coat (default: 0.32)",
    )
    parser.add_argument("--colormap", type=str, default="RdBu_r")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--prefix", type=str, default="aal_surface")
    parser.add_argument("--bar_label", type=str, default="Effect size")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution metadata (default: 600)")
    parser.add_argument("--test_random", action="store_true",
                        help="Generate random test stats instead of providing --stat_values")
    args = parser.parse_args()

    # --- Build stat_values dict ---
    if args.test_random and args.mapping_csv:
        stat_values = generate_random_test_stats(args.mapping_csv)
        print(f"Generated random test stats for {len(stat_values)} regions")
    elif args.stat_values:
        stat_values = parse_label_value_pairs(args.stat_values, float)
        significance = {}
    elif args.stat_csv and os.path.exists(args.stat_csv):
        stat_values, significance = load_statistical_csv(
            args.stat_csv,
            label_columns=("surface_label_id", "label_id"),
            alpha=args.alpha,
        )
    else:
        if not args.mapping_csv:
            parser.error("One of --test_random, --stat_values, --stat_csv is required.")
        stat_values = generate_random_test_stats(args.mapping_csv)
        significance = {}
        print(f"Auto-generated random test stats for {len(stat_values)} regions")

    if args.test_random and args.mapping_csv:
        significance = {}
    if args.significance_values:
        significance.update(
            parse_label_value_pairs(args.significance_values, parse_significance)
        )

    plotter = AALSurfaceENIGMAPlotter(
        colormap=args.colormap,
        output_dir=args.output_dir,
        dpi=args.dpi,
        nonsignificant_material=args.nonsignificant_material,
        frosted_opacity=args.frosted_opacity,
    )

    plotter.load_data(
        lh_inflated_path=args.lh_inflated,
        rh_inflated_path=args.rh_inflated,
        lh_label_path=args.lh_label,
        rh_label_path=args.rh_label,
        mapping_csv=args.mapping_csv,
        stat_values=stat_values,
        significance=significance,
        nonsignificant_opacity=args.nonsignificant_opacity,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    plotter.render_hemisphere_grids(prefix=args.prefix, scalar_bar_label=args.bar_label)
    print("Done.")
