"""Plot labelled JHU mesh statistics inside an MNI152 glass brain.

Python example
--------------
from cvdproc.utils.python.visualization.plot_JHU_glass_brain import plot_jhu_glass_brain
plot_jhu_glass_brain(values=[...50 values...], output="jhu_stats.png")
plot_jhu_glass_brain(
    values=[...50 values...],
    significance=[...50 boolean flags...],
    output="jhu_stats.png",
)

CLI examples
------------
python plot_JHU_glass_brain.py --values 0.1 -0.2 ... --output jhu_stats.png
python plot_JHU_glass_brain.py --stat_csv values.csv --output jhu_stats.png
python plot_JHU_glass_brain.py --stat_values "1:0.2,..." \
    --significance_values "1:1,2:0,..." --output jhu_stats.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ATLAS_DIR = REPO_ROOT / "cvdproc" / "data" / "atlas" / "JHU"
STANDARD_MNI152_DIR = REPO_ROOT / "cvdproc" / "data" / "standard" / "MNI152"
DEFAULT_MESH = ATLAS_DIR / "JHU.49groove.surf.gii"
DEFAULT_BRAIN = STANDARD_MNI152_DIR / "MNI152_T1_1mm_brain.nii.gz"

from cvdproc.utils.python.visualization.surface_plotting_utils import (  # noqa: E402
    colour_limits as shared_colour_limits,
    load_gifti_surface,
    load_statistical_csv,
    normalize_label_mapping,
    normalize_significance_mapping,
    parse_label_value_pairs,
    parse_significance,
    vertex_statistic_and_opacity,
)


def vtk_poly(points, faces):
    """Convert NumPy point/triangle arrays to VTK PolyData."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points, np.float32), deep=True))
    poly.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    packed = np.c_[np.full(len(faces), 3, dtype=np.int64), faces].ravel()
    cells.SetCells(
        len(faces),
        numpy_to_vtk(packed, deep=True, array_type=vtk.VTK_ID_TYPE),
    )
    poly.SetPolys(cells)
    return poly


def brain_surface(image, iso_value=4750.0):
    """Extract the T1 intensity isosurface used by ``plot_glass_brain.py``."""
    from skimage.measure import marching_cubes

    data = np.asarray(image.dataobj, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("The MNI brain template contains no finite voxels")
    data_min, data_max = float(finite.min()), float(finite.max())
    if not data_min < iso_value < data_max:
        raise ValueError(
            f"brain_iso_value must lie inside the T1 range "
            f"({data_min:g}, {data_max:g}); got {iso_value:g}"
        )
    voxel_points, faces, _, _ = marching_cubes(data, level=iso_value)
    world_points = nib.affines.apply_affine(image.affine, voxel_points)
    return world_points.astype(np.float32), faces.astype(np.int32), data


def smooth_brain_poly(points, faces):
    """Apply the same Laplacian + Taubin smoothing as ``plot_glass_brain.py``."""
    import vtk

    laplacian = vtk.vtkSmoothPolyDataFilter()
    laplacian.SetInputData(vtk_poly(points, faces))
    laplacian.SetNumberOfIterations(400)
    laplacian.SetRelaxationFactor(0.03)
    laplacian.FeatureEdgeSmoothingOff()
    laplacian.BoundarySmoothingOn()

    taubin = vtk.vtkWindowedSincPolyDataFilter()
    taubin.SetInputConnection(laplacian.GetOutputPort())
    taubin.SetNumberOfIterations(200)
    taubin.SetPassBand(0.1)
    taubin.FeatureEdgeSmoothingOff()
    taubin.BoundarySmoothingOff()
    taubin.NonManifoldSmoothingOn()
    taubin.NormalizeCoordinatesOn()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(taubin.GetOutputPort())
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def render_view(renderer, direction, view_up, bounds, size=(900, 850)):
    """Render one orthographic view to a PIL image."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    centre = np.array(
        [(bounds[0] + bounds[1]) / 2,
         (bounds[2] + bounds[3]) / 2,
         (bounds[4] + bounds[5]) / 2],
        dtype=float,
    )
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    vector = np.asarray(direction, dtype=float)
    vector /= np.linalg.norm(vector)
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*centre)
    camera.SetPosition(*(centre + vector * extent * 3.2))
    camera.SetViewUp(*view_up)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(extent * 0.55)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(8)
    window.SetSize(*size)
    window.AddRenderer(renderer)
    renderer.ResetCameraClippingRange()
    window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGBA()
    capture.ReadFrontBufferOff()
    capture.Update()
    vtk_image = capture.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    pixels = vtk_to_numpy(vtk_image.GetPointData().GetScalars()).reshape(height, width, 4)
    image = Image.fromarray(np.flipud(pixels), "RGBA").convert("RGB")
    window.RemoveRenderer(renderer)
    window.Finalize()
    return image


def load_combined_mesh(path):
    return load_gifti_surface(path, require_labels=True)


def normalize_values(values, label_ids):
    return normalize_label_mapping(values, label_ids, name="JHU values")


def load_values_csv(path):
    values, _ = load_statistical_csv(path, label_columns=("label_id",))
    return values


def colour_limits(values, vmin, vmax):
    return shared_colour_limits(
        values, vmin, vmax, symmetric_if_diverging=True
    )


def build_renderer(
    brain_poly,
    points,
    faces,
    vertex_labels,
    values,
    significance,
    nonsignificant_opacity,
    vmin,
    vmax,
    colormap,
    brain_opacity,
):
    import vtk
    from matplotlib import cm
    from vtk.util.numpy_support import numpy_to_vtk

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(120)
    renderer.SetOcclusionRatio(0.05)

    brain_mapper = vtk.vtkPolyDataMapper()
    brain_mapper.SetInputData(brain_poly)
    brain_actor = vtk.vtkActor()
    brain_actor.SetMapper(brain_mapper)
    brain_property = brain_actor.GetProperty()
    brain_property.SetColor(0.827, 0.827, 0.827)
    brain_property.SetOpacity(brain_opacity)
    brain_property.SetInterpolationToPhong()
    brain_property.SetAmbient(0.0)
    brain_property.SetDiffuse(1.0)
    brain_property.SetSpecular(0.25)
    brain_property.SetSpecularPower(15)
    renderer.AddActor(brain_actor)

    cmap = cm.get_cmap(colormap)
    scalar, opacity = vertex_statistic_and_opacity(
        vertex_labels,
        values,
        significance,
        nonsignificant_opacity=nonsignificant_opacity,
    )
    # Encode significance as actual RGBA surface opacity.  An opaque neutral
    # copy underneath would hide the alpha difference and only desaturate the
    # non-significant tracts.
    fraction = np.clip((scalar - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    colours = np.asarray(cmap(fraction) * 255, dtype=np.uint8)
    colours[:, 3] = np.asarray(np.clip(opacity, 0, 1) * 255, dtype=np.uint8)
    poly = vtk_poly(points, faces)
    vtk_colours = numpy_to_vtk(colours, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_colours.SetName("StatisticRGB")
    vtk_colours.SetNumberOfComponents(4)
    poly.GetPointData().SetScalars(vtk_colours)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.ForceTranslucentOn()
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetAmbient(0.19)
    actor.GetProperty().SetDiffuse(0.76)
    actor.GetProperty().SetSpecular(0.12)
    actor.GetProperty().SetSpecularPower(24)
    renderer.AddActor(actor)
    return renderer


def compose(panels, output, vmin, vmax, colormap, title, bar_label, dpi):
    width, height = panels[0].size
    columns, rows = 3, 2
    top, bar_width = 90, 300
    canvas = Image.new(
        "RGB",
        (width * columns + bar_width, height * rows + top),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        font = ImageFont.truetype("arial.ttf", 27)
        small = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        title_font = font = small = ImageFont.load_default()
    draw.text((35, 24), title, fill=(35, 35, 35), font=title_font)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        panel_x = column * width
        panel_y = top + row * height
        canvas.paste(panel, (panel_x, panel_y))
    from matplotlib import cm
    cmap = cm.get_cmap(colormap)
    x0 = width * columns + 75
    y0 = top + int(height * 0.45)
    y1 = top + int(height * 1.55)
    for y in range(y0, y1):
        fraction = 1 - (y - y0) / max(y1 - y0 - 1, 1)
        colour = tuple(int(255 * channel) for channel in cmap(fraction)[:3])
        draw.line((x0, y, x0 + 52, y), fill=colour)
    draw.text((x0 - 3, y0 - 55), bar_label, fill=(35, 35, 35), font=font)
    draw.text((x0 + 68, y0 - 12), f"{vmax:g}", fill=(55, 55, 55), font=small)
    draw.text((x0 + 68, y1 - 14), f"{vmin:g}", fill=(55, 55, 55), font=small)
    if vmin < 0 < vmax:
        draw.text((x0 + 68, (y0 + y1) // 2 - 12), "0", fill=(55, 55, 55), font=small)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(dpi, dpi))


def plot_jhu_glass_brain(
    values,
    output,
    mesh_path=DEFAULT_MESH,
    brain_path=DEFAULT_BRAIN,
    vmin=None,
    vmax=None,
    colormap="RdBu_r",
    title="JHU white-matter statistics",
    bar_label="Statistic",
    brain_opacity=0.10,
    dpi=600,
    brain_iso_value=4750.0,
    significance=None,
    nonsignificant_opacity=0.35,
):
    points, faces, vertex_labels = load_combined_mesh(mesh_path)
    label_ids = np.unique(vertex_labels)
    values = normalize_values(values, label_ids)
    significance = normalize_significance_mapping(significance, label_ids)
    vmin, vmax = colour_limits(values.values(), vmin, vmax)
    brain_image = nib.load(brain_path)
    brain_points, brain_faces, _ = brain_surface(brain_image, brain_iso_value)
    brain_poly = smooth_brain_poly(brain_points, brain_faces)
    renderer = build_renderer(
        brain_poly,
        points,
        faces,
        vertex_labels,
        values,
        significance,
        nonsignificant_opacity,
        vmin,
        vmax,
        colormap,
        brain_opacity,
    )
    lower, upper = brain_points.min(axis=0), brain_points.max(axis=0)
    bounds = (lower[0], upper[0], lower[1], upper[1], lower[2], upper[2])
    views = [
        ((0, 1, 0), (0, 0, 1)),
        ((0, -1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 0, 1)),
        ((1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 0)),
        ((0, 0, -1), (0, 1, 0)),
    ]
    panels = [render_view(renderer, direction, up, bounds) for direction, up in views]
    compose(panels, output, vmin, vmax, colormap, title, bar_label, dpi)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--values", nargs="+", type=float,
                       help="50 ordered values corresponding to label IDs 1..50")
    group.add_argument("--stat_values",
                       help="Comma-separated label:value pairs for all labels")
    group.add_argument("--stat_csv", help="CSV columns: label_id,statistic_value")
    parser.add_argument(
        "--significance_values",
        help="Optional label_id:flag pairs; flags accept 1/0, true/false, yes/no",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--nonsignificant_opacity", type=float, default=0.35)
    parser.add_argument("--mesh", default=str(DEFAULT_MESH))
    parser.add_argument("--brain", default=str(DEFAULT_BRAIN))
    parser.add_argument("--output", required=True)
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--colormap", default="RdBu_r")
    parser.add_argument("--title", default="JHU white-matter statistics")
    parser.add_argument("--bar_label", default="Statistic")
    parser.add_argument("--brain_opacity", type=float, default=0.10)
    parser.add_argument(
        "--brain_iso_value",
        type=float,
        default=4750.0,
        help="T1 intensity isovalue for the glass-brain surface (default: 4750)",
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.values is not None:
        values = args.values
        significance = {}
    elif args.stat_values:
        values = parse_label_value_pairs(args.stat_values, float)
        significance = {}
    else:
        values, significance = load_statistical_csv(
            args.stat_csv, label_columns=("label_id",), alpha=args.alpha
        )
    if args.significance_values:
        significance.update(
            parse_label_value_pairs(args.significance_values, parse_significance)
        )
    output = plot_jhu_glass_brain(
        values=values,
        significance=significance,
        nonsignificant_opacity=args.nonsignificant_opacity,
        output=args.output,
        mesh_path=args.mesh,
        brain_path=args.brain,
        vmin=args.vmin,
        vmax=args.vmax,
        colormap=args.colormap,
        title=args.title,
        bar_label=args.bar_label,
        brain_opacity=args.brain_opacity,
        dpi=args.dpi,
        brain_iso_value=args.brain_iso_value,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
