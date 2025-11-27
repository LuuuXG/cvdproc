import nibabel as nib
import numpy as np
from skimage.filters import frangi

# 输入输出文件
in_file = "/mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/T1w_filterNLM2.nii.gz"
out_file = "//mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/T1w_frangi.nii.gz"

# 读取 NIfTI
img = nib.load(in_file)
data = img.get_fdata().astype(np.float32)

# Frangi 滤波
# 参数：scale_range (σ范围)、scale_step (σ步长)，越大能检出更粗管状结构
frangi_img = frangi(
    data,
    scale_range=(1, 3),  # 小管状结构 (1~3voxel)
    scale_step=1,
    alpha=0.5,           # 尺度归一化参数
    beta=0.5,            # 平滑与线性度平衡
    gamma=15             # 亮暗对比参数
)

# 保存结果
out_img = nib.Nifti1Image(frangi_img.astype(np.float32), img.affine, img.header)
nib.save(out_img, out_file)

print(f"[INFO] Frangi filter applied. Saved to {out_file}")
