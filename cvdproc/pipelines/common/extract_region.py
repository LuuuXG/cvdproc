import nibabel as nib
import numpy as np

def extract_roi_from_nii(input_nii_path, roi_list, binarize, output_nii_path):
    """
    从输入的 nii 文件中提取指定的 ROI 区域，并保存为新的 nii 文件。

    参数：
        input_nii_path (str): 输入 nii 文件的路径。
        roi_list (list of int): 要提取的 ROI 区域值列表（例如 [1, 2, 3]）。
        binarize (bool): 是否将提取的区域二值化。若为 True，则选取的区域所有值设为 1，
                         否则保留原始的区域值。
        output_nii_path (str): 输出 nii 文件的路径。
    """
    # 加载输入的 nii 文件
    img = nib.load(input_nii_path)
    # 获取数据数组。注意：get_fdata() 返回浮点型数据，
    # 如果原始数据为整数但希望后续处理仍为整数，可以考虑在此处转换数据类型。
    data = img.get_fdata()

    # 生成 ROI 掩码：选取 data 中在 roi_list 内的所有 voxel
    roi_mask = np.isin(data, roi_list)

    # 初始化一个与原始数据同样形状的数组，所有 voxel 初始为 0
    new_data = np.zeros(data.shape, dtype=data.dtype)

    if binarize:
        # 二值化：将选取的区域全部设为 1
        new_data[roi_mask] = 1
    else:
        # 保留原始值
        new_data[roi_mask] = data[roi_mask]

    # 构造新的 nii 图像，保留原始的 affine 和 header 信息
    new_img = nib.Nifti1Image(new_data, affine=img.affine, header=img.header)

    # 保存新的 nii 文件到指定路径
    nib.save(new_img, output_nii_path)

    print(f"Extracted ROI saved to {output_nii_path}")

    return output_nii_path

if __name__ == "__main__":
    input_nii_path = r"F:\CSVD_revision\shiva_output_pvs\results\segmentations\pvs_segmentation\sub-YC_TI039\Brain_Seg_for_PVS.nii.gz"
    roi_list = [1, 6]
    binarize = True
    output_nii_path = r"F:\CSVD_revision\shiva_output_pvs\results\segmentations\pvs_segmentation\sub-YC_TI039\Brain_Seg_for_PVS_DM.nii.gz"

    extract_roi_from_nii(input_nii_path, roi_list, binarize, output_nii_path)