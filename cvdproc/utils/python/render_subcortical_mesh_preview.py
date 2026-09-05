"""Render a compact, four-view PNG preview of a labelled GIFTI surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import vtk


def actor_for_label(points, faces, colour):
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToFloat()
    for point in points:
        vtk_points.InsertNextPoint(*point)
    poly.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3, [int(face[0]), int(face[1]), int(face[2])])
    poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SetFeatureAngle(75)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*colour[:3])
    actor.GetProperty().SetAmbient(0.16)
    actor.GetProperty().SetDiffuse(0.74)
    actor.GetProperty().SetSpecular(0.18)
    actor.GetProperty().SetSpecularPower(18)
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image = nib.load(str(args.surface))
    points, triangles, labels = (array.data for array in image.darrays[:3])
    label_colours = {entry.key: entry.rgba for entry in image.labeltable.labels}

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1600, 480)
    cameras = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0.7, -1, 0.35)]
    for index, direction in enumerate(cameras):
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(index / 4, 0, (index + 1) / 4, 1)
        renderer.SetBackground(1, 1, 1)
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            lookup = np.full(len(points), -1, dtype=np.int32)
            lookup[indices] = np.arange(len(indices))
            faces = triangles[np.all(labels[triangles] == label, axis=1)]
            renderer.AddActor(actor_for_label(points[indices], lookup[faces], label_colours[int(label)]))
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        focal = camera.GetFocalPoint()
        distance = camera.GetDistance()
        vector = np.asarray(direction, dtype=float)
        vector /= np.linalg.norm(vector)
        camera.SetPosition(*(np.asarray(focal) + distance * vector))
        camera.SetViewUp(0, 0, 1)
        camera.OrthogonalizeViewUp()
        renderer.ResetCameraClippingRange()
        window.AddRenderer(renderer)
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(args.output))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()


if __name__ == "__main__":
    main()
