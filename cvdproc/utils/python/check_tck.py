import nibabel as nib
import numpy as np
from nibabel.streamlines import load, Tractogram, save

def convert_tck_to_dsi_mm_space(tck_path, fa1_path, fa_dsi_path, output_path):
    # Load TCK streamlines (RASMM space)
    tck = load(tck_path)
    streamlines = tck.streamlines

    # Load FA images
    fa1_img = nib.load(fa1_path)
    fa1_affine = fa1_img.affine
    fa1_affine_inv = np.linalg.inv(fa1_affine)

    fa_dsi_img = nib.load(fa_dsi_path)
    fa_dsi_affine = fa_dsi_img.affine
    fa_dsi_header = fa_dsi_img.header
    fa_dsi_voxel_size = fa_dsi_header.get_zooms()[:3]

    # Compute voxel coordinates from RASMM
    streamlines_voxel = [nib.affines.apply_affine(fa1_affine_inv, s) for s in streamlines]

    # Convert to DSI mm space
    streamlines_dsi_mm = [nib.affines.apply_affine(fa_dsi_affine, s) for s in streamlines_voxel]

    # Fix Z scaling (DSI Studio assumes isotropic spacing)
    scale_factor = fa_dsi_voxel_size[2] / fa_dsi_voxel_size[0]  # e.g., 3.0 / 1.796875
    print(f"Z scaling factor applied: {scale_factor:.3f}")

    streamlines_fixed = [s * [1, 1, scale_factor] for s in streamlines_dsi_mm]

    # Save final TCK in DSI-compatible space
    tractogram = Tractogram(streamlines_fixed, affine_to_rasmm=np.eye(4))
    save(tractogram, output_path)
    print(f"Saved transformed TCK to: {output_path}")

# Example usage:
convert_tck_to_dsi_mm_space(
    tck_path=r"F:\BIDS\SVD_BIDS\derivatives\dwi_pipeline\sub-SVD0035\ses-02\tckgen_output\tracked.tck",
    fa1_path=r"F:\BIDS\SVD_BIDS\derivatives\dwi_pipeline\sub-SVD0035\ses-02\dti_FA.nii.gz",
    fa_dsi_path=r"F:\BIDS\SVD_BIDS\derivatives\dwi_pipeline\sub-SVD0035\ses-02\tckgen_output\dsistudio\sub-SVD0035_ses-02_acq-DTIb1000_dwi_fa.nii.gz",
    output_path=r"F:\BIDS\SVD_BIDS\derivatives\dwi_pipeline\sub-SVD0035\ses-02\tckgen_output\tracked_in_dsi_space.tck"
)
