import os
import nibabel as nib
import numpy as np
import scipy.io as sio

mag_4d_path = '/mnt/f/BIDS/demo_BIDS/derivatives/qsm_pipeline/sub-TAOHC0261/ses-baseline/sub-TAOHC0261_ses-baseline_part-mag_desc-smoothed_GRE.nii.gz'
r2star_path = '/mnt/f/BIDS/demo_BIDS/derivatives/qsm_pipeline/sub-TAOHC0261/ses-baseline/sepia_output/sub-TAOHC0261_ses-baseline_R2starmap.nii.gz'
s0_path = '/mnt/f/BIDS/demo_BIDS/derivatives/qsm_pipeline/sub-TAOHC0261/ses-baseline/sepia_output/sub-TAOHC0261_ses-baseline_S0map.nii.gz'
header_path = '/mnt/f/BIDS/demo_BIDS/derivatives/qsm_pipeline/sub-TAOHC0261/ses-baseline/sub-TAOHC0261_ses-baseline_desc-sepia_header.mat'

# only fectch first 5 echoes
mag_4d_img = nib.load(mag_4d_path)
mag_4d = mag_4d_img.get_fdata()
if mag_4d.shape[3] > 5:
    mag_4d = mag_4d[:,:,:,0:5]

# load header to get TE and delta_TE
te = sio.loadmat(header_path)['TE'].squeeze()  # in s
delta_te = sio.loadmat(header_path)['delta_TE'].squeeze()  # in s

# load S0 and R2* maps
r2s_img = nib.load(r2star_path)
r2s = r2s_img.get_fdata().astype(np.float32)   # 1/s
s0_img = nib.load(s0_path)
s0 = s0_img.get_fdata().astype(np.float32)

# TE handling (seconds)
te = np.asarray(te).squeeze().astype(np.float32)
if te.size < 5:
    raise ValueError("TE contains fewer than 5 values.")
te5 = te[:5]

# pick delta TE from header if valid, otherwise compute from te5
delta_te = np.asarray(delta_te).squeeze()
if delta_te.size >= 1 and float(np.atleast_1d(delta_te).ravel()[0]) > 0:
    dte = float(np.atleast_1d(delta_te).ravel()[0])
else:
    dte = float(np.mean(np.diff(te5)))

# build 3 extra TEs and the full 8-TE vector
te_extra = te5[-1] + dte * np.arange(1, 4, dtype=np.float32)  # TE6..TE8
te_full = np.concatenate([te5, te_extra], axis=0).astype(np.float32)  # length 8

# synthesize 3 echoes using S(TE) = S0 * exp(-R2* * TE)
synth_list = []
for te_val in te_extra:
    synth_list.append(s0 * np.exp(-r2s * float(te_val)))
synth3 = np.stack(synth_list, axis=-1).astype(np.float32)  # (X,Y,Z,3)

# combine measured (5) + synthetic (3)
mag_4d = mag_4d.astype(np.float32)
if mag_4d.shape[-1] != 5:
    raise ValueError("After slicing, mag_4d is not 5 echoes.")
mag8 = np.concatenate([mag_4d, synth3], axis=-1).astype(np.float32)  # (X,Y,Z,8)

# voxel-wise normalization by echo-1 (to match QQNet input convention)
eps = 1e-6
den = mag8[..., 0] + eps
mag8_norm = mag8 / den[..., None]
# explicitly set channel 0 to 1 inside foreground and 0 outside
ch0 = np.zeros_like(den, dtype=np.float32)
ch0[den > eps] = 1.0
mag8_norm[..., 0] = ch0

# save outputs
out_dir = os.path.join(os.path.dirname(mag_4d_path), "qqnet_input")
os.makedirs(out_dir, exist_ok=True)

mag8_raw_nii = nib.Nifti1Image(mag8, mag_4d_img.affine, mag_4d_img.header)
nib.save(mag8_raw_nii, os.path.join(out_dir, "sub-TAOHC0261_ses-baseline_mag8_raw.nii.gz"))

mag8_norm_nii = nib.Nifti1Image(mag8_norm, mag_4d_img.affine, mag_4d_img.header)
nib.save(mag8_norm_nii, os.path.join(out_dir, "sub-TAOHC0261_ses-baseline_mag8_norm.nii.gz"))

# also save TE list for record
sio.savemat(os.path.join(out_dir, "te_full.mat"), {"TE_full": te_full})

print("Done.")
print("mag8_raw shape:", mag8.shape, "mag8_norm shape:", mag8_norm.shape)
print("TE_full (s):", te_full)