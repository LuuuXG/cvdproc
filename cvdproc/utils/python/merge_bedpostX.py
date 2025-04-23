import os
import glob
import nibabel as nib
import numpy as np

folder1 = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0029/ses-03/bedpostX_input.bedpostX3"
folder2 = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0029/ses-03/bedpostX_input.bedpostX1"
output_folder = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0029/ses-03/bedpostX_input.bedpostX"

os.makedirs(output_folder, exist_ok=True)

# 获取 folder1 中所有 .nii.gz 文件（不含子目录）
files = sorted([
    os.path.basename(f)
    for f in glob.glob(os.path.join(folder1, "*.nii.gz"))
    if os.path.isfile(f)
])

for fname in files:
    f1 = os.path.join(folder1, fname)
    f2 = os.path.join(folder2, fname)
    out = os.path.join(output_folder, fname)

    if not os.path.exists(f1) or not os.path.exists(f2):
        print(f"Skipping {fname} (missing in one of the folders)")
        continue

    img1 = nib.load(f1)
    img2 = nib.load(f2)
    d1 = img1.get_fdata()
    d2 = img2.get_fdata()

    if d1.shape != d2.shape:
        print(f"Shape mismatch for {fname}, skipping.")
        continue

    merged = np.copy(d2)
    merged[:, :, 2:25] = d1[:, :, 2:25]

    nib.save(nib.Nifti1Image(merged, img1.affine, img1.header), out)
    print(f"Merged: {fname}")
