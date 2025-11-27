import os
import nibabel as nib
import numpy as np


def threshold_binarize_nifti(image_path, threshold, output_path=None):
    """
    Apply threshold-based binarization to a NIfTI image and save it to disk.

    Parameters:
    - image_path (str): Path to the input NIfTI file (.nii or .nii.gz).
    - threshold (float): Threshold used for binarization.
    - output_path (str): Optional path for the binarized image.
    """
    # Load the NIfTI image
    img = nib.load(image_path)
    img_data = img.get_fdata()

    # Apply thresholding
    binarized_data = (img_data > threshold).astype(np.uint8)

    # Create a new NIfTI image
    binarized_img = nib.Nifti1Image(binarized_data, img.affine)

    # Build the output file path
    if output_path is not None:
        output_path = output_path
    else:
        image_name = os.path.basename(image_path)
        folder = os.path.dirname(image_path)
        output_filename = f"binarized_thr{threshold}_{image_name}"
        output_path = os.path.join(folder, output_filename)

    # Save the binarized image
    nib.save(binarized_img, output_path)
    print(f"Binarized image saved to {output_path}")

    return output_path

if __name__ == "__main__":
    threshold_binarize_nifti('/mnt/e/Codes/cvdproc/cvdproc/data/standard/mni_icbm152_nlin_asym_09a_nifti/mni_icbm152_gm_tal_nlin_asym_09a.nii', 0.15)
