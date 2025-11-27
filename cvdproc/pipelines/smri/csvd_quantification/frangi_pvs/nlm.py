#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import os
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

# ==== 输入输出 ====
in_file = "/mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/sub-AFib0241_ses-baseline_acq-highres_desc-brain_T1w.nii.gz"
out_file = "/mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/T1w_filterNLM2.nii.gz"

# ==== 加载数据 ====
img = nib.load(in_file)
data = img.get_fdata().astype(np.float32)

# ==== 估计噪声 (全图) ====
sigma = np.mean(estimate_sigma(data, N=1))

# ==== NLM 滤波 (QIT 默认 patch=1, search=2) ====
denoised = nlmeans(
    data,
    sigma=sigma,
    patch_radius=1,
    block_radius=2,
    rician=True
)

# ==== 保存结果 ====
den_img = nib.Nifti1Image(denoised, img.affine, img.header)
nib.save(den_img, out_file)

print(f"[INFO] Denoised image saved to {out_file}")
