import nibabel as nib
import numpy as np

# 读取 lh.white.gii（网格数据）
gii_mesh = nib.load("E:/Neuroimage/lh.white.gii")  # 仅包含顶点和面

# 读取 lh.thickness.gii（包含顶点值）
gii_thickness = nib.load("E:/Neuroimage/lh.thickness.gii")

# 提取数据
vertices = gii_mesh.darrays[0].data  # 顶点坐标
faces = gii_mesh.darrays[1].data  # 面索引
thickness_values = gii_thickness.darrays[0].data  # 顶点值（可能有非 0 值）

# 确保顶点数匹配
num_vertices = vertices.shape[0]
assert thickness_values.shape[0] == num_vertices, "Error: Thickness and mesh vertex counts do not match!"

# 创建一个新的顶点值数组：
# - thickness_values > 0 的地方设为 1
# - 其他地方保留原始值
vertex_values = np.where(thickness_values > 0, 1, thickness_values).astype(np.float32)

# 创建新的 GIFTI 对象
new_gii = nib.gifti.GiftiImage()

# 添加顶点（pointset）
new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
    vertices, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
))

# 添加面（triangle）
new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
    faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
))

# 添加修正后的顶点值（NIFTI_INTENT_SHAPE）
new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
    vertex_values, intent=nib.nifti1.intent_codes['NIFTI_INTENT_SHAPE']
))

# 保存新的 GIFTI 文件
output_path = "E:/Neuroimage/lh.cortex_fixed.gii"
nib.save(new_gii, output_path)
print(f"Fixed GIFTI file saved as {output_path}")
