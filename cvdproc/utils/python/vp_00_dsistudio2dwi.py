# Transform image exported from DSI Studio to original DWI space
# by replacing the affine and header information

import nibabel as nib

image_A_path = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0382/ses-baseline/visual_pathway_analysis/rh_OR_final.nii.gz"  # DSI Studio exported image
image_B_path = "/mnt/f/BIDS/WCH_AF_Project/derivatives/qsiprep/sub-HC0382/ses-baseline/dwi/sub-HC0382_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_dwiref.nii.gz"  # Original DWI image
output_path = image_A_path

img_A = nib.load(image_A_path)
img_B = nib.load(image_B_path)

new_img = nib.Nifti1Image(img_A.get_fdata(), img_B.affine, header=img_B.header)

nib.save(new_img, output_path)
