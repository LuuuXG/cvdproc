import nibabel as nib
import numpy as np

def convert_gii_to_mgh(
    gii_path: str,
    mgh_output_path: str,
    reference_mgh_path: str
):
    """
    Replace the data of an existing MGH image using value from a GIfTI file,
    while preserving all metadata (affine, header, qform, sform) from the reference.

    Parameters:
        gii_path (str): Path to GIfTI file containing surface values.
        mgh_output_path (str): Path to save the modified MGH file.
        reference_mgh_path (str): Path to reference MGH whose metadata will be preserved.
    """
    # Load reference MGH image
    ref_img = nib.load(reference_mgh_path)

    # Load value from .gii
    gii = nib.load(gii_path)
    values = np.squeeze(gii.darrays[-1].data).astype(np.float32)
    values = values[:, np.newaxis, np.newaxis]  # shape: (n_vertices, 1, 1)

    # Overwrite data in ref_img by creating a new object of the same class
    modified_img = ref_img.__class__(
        dataobj=values,
        affine=ref_img.affine,
        header=ref_img.header
    )

    nib.save(modified_img, mgh_output_path)
    print(f"Saved MGH with replaced data: {mgh_output_path}")


if __name__ == "__main__":
    gii_path = '/mnt/f/BIDS/SVD_BIDS/derivatives/nemo_mean_fsaverage/lh_smooth6mm_mean_masked.func.gii'
    mgh_output_path = '/mnt/f/BIDS/SVD_BIDS/derivatives/nemo_mean_fsaverage/lh_smooth6mm_mean_masked.mgh'
    reference_mgh_path = '/mnt/f/BIDS/SVD_BIDS/derivatives/dwi_pipeline/sub-SVD0003/ses-01/mirror_probtrackx_output/lh.HighConn_fsaverage.mgh'

    convert_gii_to_mgh(gii_path, mgh_output_path, reference_mgh_path)