#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os
import tempfile
import argparse

def nifti_warp_to_h5(input_nifti: str, output_h5: str):
    """
    Convert a 4D NIfTI warp field (x,y,z,3) to an ANTs-compatible .h5 transform file.
    The function creates a temporary NIFTI_INTENT_DISPVECT displacement field,
    which is removed after the .h5 is written.
    """

    print(f"Loading warp: {input_nifti}")
    warped = nib.load(input_nifti)
    data = warped.get_fdata()
    affine = warped.affine

    shape = data.shape
    if len(shape) != 4 or shape[3] != 3:
        raise ValueError("Input image must be 4D with last dimension size 3 (x,y,z,3).")

    # Reshape to (X, Y, Z, 1, 3)
    data = np.reshape(data, (shape[0], shape[1], shape[2], 1, 3))

    # Convert voxel displacements to physical displacements
    reshaped = np.reshape(data, (3, -1))
    stacked = np.vstack([reshaped, np.zeros(reshaped.shape[1])])
    multiplied = np.matmul(affine, stacked)[:3,]
    last = np.reshape(multiplied, data.shape)

    # Save temporary displacement field with DISPVECT intent
    temp_disp = tempfile.NamedTemporaryFile(suffix="_displacement.nii.gz", delete=False).name
    new_img = nib.Nifti1Image(last, affine)
    new_img.header.set_intent(nib.nifti1.intent_codes['NIFTI_INTENT_DISPVECT'])
    nib.save(new_img, temp_disp)
    print(f"Saved temporary displacement field: {temp_disp}")

    # Convert to ANTs-compatible HDF5 transform
    print("Converting to ANTs HDF5 transform...")
    displacement_image = sitk.ReadImage(
        temp_disp,
        sitk.sitkVectorFloat64,
        imageIO="NiftiImageIO"
    )
    tx = sitk.DisplacementFieldTransform(displacement_image)
    sitk.WriteTransform(tx, output_h5)
    print(f"Saved ANTs-compatible transform: {output_h5}")

    # Remove temporary file
    try:
        os.remove(temp_disp)
        print("Temporary file removed.")
    except Exception as e:
        print(f"Warning: failed to remove temporary file {temp_disp}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NIfTI warp to ANTs-compatible H5 transform.")
    parser.add_argument("--input", required=True, help="Input NIfTI warp file (4D, last dimension=3)")
    parser.add_argument("--output", required=True, help="Output HDF5 transform file (.h5)")
    args = parser.parse_args()

    nifti_warp_to_h5(args.input, args.output)
