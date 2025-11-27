#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from nilearn.image import resample_img

src_path = "/mnt/e/Neuroimage/workdir/sub-HC0380_ses-baseline_acq-highres_T1w.nii.gz"
ref_path = "/mnt/e/Neuroimage/workdir/sub-HC0380_ses-baseline_acq-highres_T1w_lps.nii.gz"
out_path = "/mnt/e/Neuroimage/workdir/sub-HC0380_ses-baseline_acq-highres_T1w_inLPSspace.nii.gz"

# 1) 读取源与参考，并规范 sform/qform
src = nib.load(src_path)
ref = nib.load(ref_path)

def normalize_xform(img):
    hdr = img.header.copy()
    hdr.set_qform(img.affine, code=1)
    hdr.set_sform(img.affine, code=1)
    return img.__class__(img.get_fdata(), img.affine, hdr)

src = normalize_xform(src)
ref = normalize_xform(ref)

# 2) 在世界坐标中把源图重采样到“参考LPS”的 affine + shape
#    这一步自动完成：方向匹配(RAS→LPS)、去斜切到正交网格、体素对齐
resampled = resample_img(
    src,
    target_affine=ref.affine,
    target_shape=ref.shape[:3],
    interpolation="continuous"   # 与 qsiprep/niworkflows 使用一致
)

# 3) 保存，并把 qform/sform 写成与 ref 一样（更保险）
hdr = resampled.header.copy()
hdr.set_qform(ref.affine, code=1)
hdr.set_sform(ref.affine, code=1)
out = resampled.__class__(resampled.get_fdata().astype(np.float32), ref.affine, hdr)
nib.save(out, out_path)

# 4) 快速校验
print("Saved:", out_path)
print("Ref axcodes:", nib.aff2axcodes(ref.affine), "shape:", ref.shape)
print("Out axcodes:", nib.aff2axcodes(out.affine), "shape:", out.shape)
print("Affine equal to ref?:", np.allclose(out.affine, ref.affine))
