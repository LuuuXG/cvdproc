#!/usr/bin/env python3
import numpy as np
import nibabel as nib
from skimage.filters import frangi
from scipy.ndimage import binary_dilation

# ==== 输入文件 ====
in_file = "/mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/T1w_filterNLM.nii"
synthseg_file = "/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-AFib0241/ses-baseline/synthseg/sub-AFib0241_ses-baseline_space-T1w_synthseg.nii.gz"
brainmask_file = "/mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/sub-AFib0241_ses-baseline_acq-highres_space-T1w_label-brain_mask.nii.gz"

out_frangi = "/mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/T1w_frangi_norm.nii.gz"

# ==== Step 1: Load data ====
img = nib.load(in_file)
data = img.get_fdata().astype(np.float32)

seg = nib.load(synthseg_file).get_fdata().astype(int)
brainmask = nib.load(brainmask_file).get_fdata().astype(bool)

# ==== Step 2: 构建 mask_keep ====
mask_keep = brainmask.copy()

# Cortical GM
mask_keep[seg > 1000] = False
# 去掉 CSF/脑室等（可扩展列表）
csf_labels = [4, 5, 43, 44, 14, 15, 24]
csf_mask = np.isin(seg, csf_labels)

# 膨胀 1 voxel（iterations=1 表示半径=1）
csf_mask_dilated = binary_dilation(csf_mask, iterations=1)

# 构建保留的 mask
mask_keep = np.ones_like(seg, dtype=bool)
mask_keep[csf_mask_dilated] = False

# cerebellum
cerebellum_labels = [6, 7, 8, 45, 46, 47] 
mask_keep[np.isin(seg, cerebellum_labels)] = False

# ==== Step 3: 先mask再frangi再mask ====
data_masked = np.zeros_like(data, dtype=np.float32)
data_masked[mask_keep] = data[mask_keep]

frangi_img = frangi(
    data_masked,
    sigmas=np.arange(0.1, 5.1, 0.5),
    alpha=0.5,
    beta=0.5,
    gamma=15
)

frangi_img[~mask_keep] = 0  # 再次mask
# in csf mask to 0
frangi_img[csf_mask_dilated] = 0

# ==== Step 4: Normalize to [0, 1] ====
if frangi_img.max() > frangi_img.min():  # 防止除零
    frangi_img = (frangi_img - frangi_img.min()) / (frangi_img.max() - frangi_img.min())
else:
    frangi_img = np.zeros_like(frangi_img, dtype=np.float32)

# ==== Step 5: Save result ====
nib.save(nib.Nifti1Image(frangi_img.astype(np.float32), img.affine, img.header), out_frangi)
print(f"[INFO] Frangi vesselness image saved (mask->frangi): {out_frangi}")
