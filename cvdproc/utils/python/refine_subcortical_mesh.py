"""Create a display-oriented, smoothed version of the AAL subcortical mesh.

The source surface is retained unchanged.  The output keeps the original
vertex labels and label table, so existing per-region plotting code continues
to work.  Each labelled structure is processed independently: this preserves
the visual separation of adjacent nuclei instead of smoothing them together.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _smooth_region(points: np.ndarray, faces: np.ndarray, pass_band: float, iterations: int) -> np.ndarray:
    """Windowed-sinc smooth a closed, single-label triangular surface."""
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToFloat()
    vtk_points.SetNumberOfPoints(len(points))
    for index, point in enumerate(points):
        vtk_points.SetPoint(index, *point)
    poly.SetPoints(vtk_points)

    cells = vtk.vtkCellArray()
    for face in faces:
        triangle = vtk.vtkTriangle()
        for corner, vertex in enumerate(face):
            triangle.GetPointIds().SetId(corner, int(vertex))
        cells.InsertNextCell(triangle)
    poly.SetPolys(cells)

    smoothing = vtk.vtkWindowedSincPolyDataFilter()
    smoothing.SetInputData(poly)
    smoothing.SetNumberOfIterations(iterations)
    smoothing.SetPassBand(pass_band)
    smoothing.BoundarySmoothingOff()
    smoothing.FeatureEdgeSmoothingOff()
    smoothing.NonManifoldSmoothingOn()
    smoothing.NormalizeCoordinatesOn()
    smoothing.Update()
    return vtk_to_numpy(smoothing.GetOutput().GetPoints().GetData()).astype(np.float32)


def refine_surface(source: Path, destination: Path, shrink: float, pass_band: float, iterations: int) -> None:
    image = nib.load(str(source))
    coordinates, triangles, labels = (array.data for array in image.darrays[:3])
    refined = coordinates.astype(np.float32, copy=True)

    for label in np.unique(labels):
        global_vertices = np.flatnonzero(labels == label)
        global_to_local = np.full(len(coordinates), -1, dtype=np.int32)
        global_to_local[global_vertices] = np.arange(len(global_vertices))
        region_faces = triangles[np.all(labels[triangles] == label, axis=1)]
        if len(region_faces) == 0:
            continue
        local_faces = global_to_local[region_faces]
        smooth = _smooth_region(coordinates[global_vertices], local_faces, pass_band, iterations)

        # A small contraction around the structure centroid opens reliable
        # display gaps while retaining its position and left/right symmetry.
        centroid = smooth.mean(axis=0, keepdims=True)
        refined[global_vertices] = centroid + shrink * (smooth - centroid)

    image.darrays[0].data = refined
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--shrink", type=float, default=0.96, help="Per-nucleus display scale (default: 0.96)")
    parser.add_argument("--pass-band", type=float, default=0.06, help="Windowed-sinc pass band; lower is smoother")
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    refine_surface(args.source, args.destination, args.shrink, args.pass_band, args.iterations)


if __name__ == "__main__":
    main()
