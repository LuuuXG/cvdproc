import nibabel as nib
import numpy as np
import os

def remove_region_from_mask(mask_path, dk_path, output_path, region_value=1):
    # Load the binary mask and DK atlas
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    dk_img = nib.load(dk_path)
    dk_data = dk_img.get_fdata()

    # Set mask values to 0 where DK atlas equals the region_value (e.g., 83 for brainstem)
    mask_data[dk_data == region_value] = 0

    # Save the modified mask
    new_mask_img = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
    nib.save(new_mask_img, output_path)

    print(f"Processed mask saved to {output_path}")

# 示例使用
mask_path = r'E:\Codes\Basic_Imaging_Process\data\mni_icbm152_gm_tal_nlin_asym_09a_binarized.nii'  # 二值mask文件路径
dk_path = r'E:\Codes\Basic_Imaging_Process\data\mni_icbm152_csf_tal_nlin_asym_09a_binarized.nii'  # DK模板文件路径
output_path = r'E:\Codes\Basic_Imaging_Process\data\mni_icbm152_gm_tal_nlin_asym_09a_binarized_removecsf.nii'  # 输出文件路径

remove_region_from_mask(mask_path, dk_path, output_path)
