"""Reconstruct a high-resolution, display-oriented AAL subcortical surface.

Unlike mesh-only smoothing, this starts from the labelled volume and uses a
Gaussian-smoothed occupancy field.  The resulting isosurfaces do not retain
the voxel-grid ripples found in a direct label-to-mesh conversion.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes


LABEL_CODES = {1: 7001, 2: 7002, 3: 7011, 4: 7012, 5: 7021, 6: 7022, 7: 7101, 8: 7102}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dseg", type=Path)
    parser.add_argument("labelled_surface", type=Path, help="Source only for the GIFTI label table")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma", type=float, default=1.05, help="Gaussian width in native 1-mm voxels")
    parser.add_argument("--upsample", type=int, default=2, help="Continuous field resolution multiplier")
    parser.add_argument("--shrink", type=float, default=0.955, help="Per-nucleus display scale")
    args = parser.parse_args()

    volume = nib.load(str(args.dseg))
    segmentation = np.asanyarray(volume.dataobj)
    source = nib.load(str(args.labelled_surface))
    all_points, all_faces, all_labels = [], [], []
    offset = 0

    for display_label, atlas_code in LABEL_CODES.items():
        # Smooth the binary occupancy itself: the 0.5 contour is a continuous
        # anatomical envelope rather than the staircase of the source voxels.
        mask = segmentation == atlas_code
        occupied = np.argwhere(mask)
        margin = int(np.ceil(4 * args.sigma)) + 2
        lower = np.maximum(occupied.min(axis=0) - margin, 0)
        upper = np.minimum(occupied.max(axis=0) + margin + 1, segmentation.shape)
        cropped_mask = mask[tuple(slice(start, stop) for start, stop in zip(lower, upper))]
        field = gaussian_filter(cropped_mask.astype(np.float32), sigma=args.sigma)
        field = zoom(field, args.upsample, order=3, prefilter=True)
        vertices, faces, _, _ = marching_cubes(field, level=0.5, spacing=(1 / args.upsample,) * 3)
        vertices += lower
        world = nib.affines.apply_affine(volume.affine, vertices)
        centre = world.mean(axis=0, keepdims=True)
        world = centre + args.shrink * (world - centre)
        all_points.append(world.astype(np.float32))
        all_faces.append((faces + offset).astype(np.int32))
        all_labels.append(np.full(len(world), display_label, dtype=np.int32))
        offset += len(world)

    output = nib.gifti.GiftiImage()
    output.add_gifti_data_array(nib.gifti.GiftiDataArray(np.vstack(all_points), intent="NIFTI_INTENT_POINTSET"))
    output.add_gifti_data_array(nib.gifti.GiftiDataArray(np.vstack(all_faces), intent="NIFTI_INTENT_TRIANGLE"))
    output.add_gifti_data_array(nib.gifti.GiftiDataArray(np.concatenate(all_labels), intent="NIFTI_INTENT_LABEL"))
    output.labeltable = copy.deepcopy(source.labeltable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(args.output))


if __name__ == "__main__":
    main()
