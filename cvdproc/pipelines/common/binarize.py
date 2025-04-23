import os
import nibabel as nib
import numpy as np


def threshold_binarize_nifti(image_path, threshold, output_path=None):
    """
    对指定的 NIfTI 图像进行阈值二值化处理，并保存到源文件所在的文件夹。

    Parameters:
    - image_path (str): 输入 NIfTI 文件的路径 (.nii 或 .nii.gz)。
    - threshold (float): 用于二值化的阈值。
    - output_path (str): 可选参数，输出二值化图像的路径。
    """
    # 加载 NIfTI 图像
    img = nib.load(image_path)
    img_data = img.get_fdata()

    # 进行阈值二值化
    binarized_data = (img_data > threshold).astype(np.uint8)

    # 创建新的 NIfTI 图像
    binarized_img = nib.Nifti1Image(binarized_data, img.affine)

    # 构建输出文件路径
    if output_path is not None:
        output_path = output_path
    else:
        image_name = os.path.basename(image_path)
        folder = os.path.dirname(image_path)
        output_filename = f"binarized_thr{threshold}_{image_name}"
        output_path = os.path.join(folder, output_filename)

    # 保存二值化图像
    nib.save(binarized_img, output_path)
    print(f"Binarized image saved to {output_path}")

    return output_path

if __name__ == "__main__":
    threshold_binarize_nifti('/mnt/e/Codes/cvdproc/cvdproc/data/standard/mni_icbm152_nlin_asym_09a_nifti/mni_icbm152_gm_tal_nlin_asym_09a.nii', 0.15)