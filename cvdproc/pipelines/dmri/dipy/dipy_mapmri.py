from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst import mapmri
import numpy as np

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
fraw = "/mnt/f/BIDS/SVD_demo/derivatives/qsiprep/sub-SVDBBB001/ses-baseline/dwi/sub-SVDBBB001_ses-baseline_acq-DSIb3000_dir-PA_space-ACPC_desc-preproc_dwi.nii.gz"
fbval = "/mnt/f/BIDS/SVD_demo/derivatives/qsiprep/sub-SVDBBB001/ses-baseline/dwi/sub-SVDBBB001_ses-baseline_acq-DSIb3000_dir-PA_space-ACPC_desc-preproc_dwi.bval"
fbvec = "/mnt/f/BIDS/SVD_demo/derivatives/qsiprep/sub-SVDBBB001/ses-baseline/dwi/sub-SVDBBB001_ses-baseline_acq-DSIb3000_dir-PA_space-ACPC_desc-preproc_dwi.bvec"

mask_fname = "/mnt/f/BIDS/SVD_demo/derivatives/qsiprep/sub-SVDBBB001/ses-baseline/dwi/sub-SVDBBB001_ses-baseline_acq-DSIb3000_dir-PA_space-ACPC_desc-brain_mask.nii.gz"

out_dir = "/mnt/f/BIDS/SVD_demo/derivatives/qsiprep/sub-SVDBBB001/ses-baseline/dwi"

# --------------------------------------------------
# 2. Load data and mask
# --------------------------------------------------
data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

mask_data, _ = load_nifti(mask_fname)
mask = mask_data > 0  # boolean mask

print("DWI shape:", data.shape)
print("Mask shape:", mask.shape, "masked voxels:", mask.sum())

# --------------------------------------------------
# 3. Build gradient table (with or without big_delta / small_delta)
# --------------------------------------------------
big_delta = 0.0365  # seconds (example value)
small_delta = 0.0157  # seconds (example value)

gtab = gradient_table(
    bvals=bvals,
    bvecs=bvecs,
    big_delta=big_delta,
    small_delta=small_delta,
)

# --------------------------------------------------
# 4. Fit MAP-MRI model within mask
# --------------------------------------------------
radial_order = 6

map_model = mapmri.MapmriModel(
    gtab,
    radial_order=radial_order,
    laplacian_regularization=False,
    positivity_constraint=True,
)

# The mask argument ensures fitting is done only inside the mask
mapfit = map_model.fit(data, mask=mask)

# --------------------------------------------------
# 5. Compute MAP-MRI scalar maps
#    (choose what you need)
# --------------------------------------------------
rtop = mapfit.rtop()                   # Return-to-origin probability
rtap = mapfit.rtap()                   # Return-to-axis probability
rtpp = mapfit.rtpp()                   # Return-to-plane probability
msd = mapfit.msd()                     # Mean squared displacement
ng = mapfit.non_gaussianity()          # Non-gaussianity

# Outside the mask, you may want to set values to 0 or NaN
rtop[~mask] = 0
rtap[~mask] = 0
rtpp[~mask] = 0
msd[~mask] = 0
ng[~mask] = 0

# --------------------------------------------------
# 6. Save as NIfTI using the DWI affine
# --------------------------------------------------
save_nifti(f"{out_dir}/sub-SVDBBB001_ses-baseline_mapmri_rtop.nii.gz", rtop, affine)
save_nifti(f"{out_dir}/sub-SVDBBB001_ses-baseline_mapmri_rtap.nii.gz", rtap, affine)
save_nifti(f"{out_dir}/sub-SVDBBB001_ses-baseline_mapmri_rtpp.nii.gz", rtpp, affine)
save_nifti(f"{out_dir}/sub-SVDBBB001_ses-baseline_mapmri_msd.nii.gz", msd, affine)
save_nifti(f"{out_dir}/sub-SVDBBB001_ses-baseline_mapmri_ng.nii.gz", ng, affine)

print("Finished MAP-MRI fitting and NIfTI export.")
