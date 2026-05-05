#!/usr/bin/env python3
import argparse
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def nifti_warp_to_h5(input_nifti: str, output_h5: str, vector_image: str = None):
    """
    Convert a 4D NIfTI warp field (X, Y, Z, 3) to an ANTs-compatible HDF5 transform.

    Parameters
    ----------
    input_nifti : str
        Warp image defining the displacement-field domain.
    output_h5 : str
        Output ANTs-compatible .h5 transform.
    vector_image : str or None
        Image whose voxel basis is used to interpret the warp vectors.
        If None, use the warp image itself.
    """

    if not os.path.isfile(input_nifti):
        raise FileNotFoundError(f"Input warp not found: {input_nifti}")
    if vector_image is not None and not os.path.isfile(vector_image):
        raise FileNotFoundError(f"Vector reference image not found: {vector_image}")

    print(f"Loading warp: {input_nifti}")
    warp_img = nib.load(input_nifti)
    data = np.asarray(warp_img.get_fdata(dtype=np.float64))
    warp_affine = warp_img.affine

    if data.ndim != 4 or data.shape[3] != 3:
        raise ValueError(f"Input image must have shape (X, Y, Z, 3), but got {data.shape}")

    if vector_image is None:
        vector_affine = warp_affine
        print("Using warp image affine as vector basis.")
    else:
        vector_affine = nib.load(vector_image).affine
        print(f"Using vector basis from: {vector_image}")

    print(f"Warp shape: {data.shape}")

    # -----------------------------
    # 1) Domain geometry comes from warp image
    # -----------------------------
    A_domain_ras = warp_affine[:3, :3]
    origin_ras = warp_affine[:3, 3]

    # -----------------------------
    # 2) Vector basis comes from vector_image affine
    # -----------------------------
    A_vector_ras = vector_affine[:3, :3]

    # Convert voxel-space displacement vectors -> physical displacement in RAS
    disp_phys_ras = np.einsum("ij,xyzj->xyzi", A_vector_ras, data)

    # -----------------------------
    # 3) Convert RAS -> LPS for ITK/ANTs
    # -----------------------------
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])

    origin_lps = ras_to_lps @ origin_ras

    spacing = np.linalg.norm(A_domain_ras, axis=0)
    if np.any(spacing <= 0):
        raise ValueError(f"Invalid spacing derived from warp affine: {spacing}")

    direction_ras = A_domain_ras / spacing
    direction_lps = ras_to_lps @ direction_ras

    disp_phys_lps = np.einsum("ij,xyzj->xyzi", ras_to_lps, disp_phys_ras)

    # SimpleITK expects [Z, Y, X, C]
    disp_sitk_array = np.transpose(disp_phys_lps, (2, 1, 0, 3))

    print("Creating SimpleITK vector image...")
    disp_img = sitk.GetImageFromArray(disp_sitk_array, isVector=True)
    disp_img.SetSpacing(tuple(spacing.tolist()))
    disp_img.SetOrigin(tuple(origin_lps.tolist()))
    disp_img.SetDirection(tuple(direction_lps.reshape(-1).tolist()))

    print("Creating displacement transform...")
    tx = sitk.DisplacementFieldTransform(disp_img)

    print(f"Writing HDF5 transform: {output_h5}")
    sitk.WriteTransform(tx, output_h5)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 4D NIfTI warp (X,Y,Z,3) to an ANTs-compatible HDF5 transform."
    )
    parser.add_argument("--input", required=True, help="Input NIfTI warp file with shape (X, Y, Z, 3)")
    parser.add_argument("--output", required=True, help="Output HDF5 transform file (.h5)")
    parser.add_argument(
        "--vector_image",
        default=None,
        help="Image whose affine defines the voxel basis of warp vectors. Default: use warp image itself.",
    )
    args = parser.parse_args()

    nifti_warp_to_h5(
        input_nifti=args.input,
        output_h5=args.output,
        vector_image=args.vector_image,
    )


if __name__ == "__main__":
    main()