#!/bin/bash

# Usage: ./generic_registration.sh <structural_image> <structural_brain_flag> <reg_image> <reg_brain_flag> <target_image> <use_brain_flag> <direction_flag> [-m <mask_file>]

# 参数说明：
# structural_image: 结构像文件路径（如 T1）
# structural_brain_flag: 结构像是否已剥脑 (1 表示已剥脑，0 表示未剥脑)
# reg_image: 用于结构像配准的图像文件路径（如 fMRI 或 mag 图像）
# reg_brain_flag: 用于配准的图像是否已剥脑 (1 表示已剥脑，0 表示未剥脑)
# target_image: 待配准的目标图像文件路径（如 QSM 或 fMRI）
# use_brain_flag: 在配准过程中是否使用剥脑后的数据 (1 表示使用，0 表示不使用)
# direction_flag: 配准方向 (0 表示 T1 配准到 reg，1 表示 reg 配准到 T1)
# -m <mask_file>: 可选参数，指定用于第二个文件的掩膜文件路径

# 检查输入参数数量
if [ "$#" -lt 7 ]; then
    echo "Usage: $0 <structural_image> <structural_brain_flag> <reg_image> <reg_brain_flag> <target_image> <use_brain_flag> <direction_flag> [-m <mask_file>]"
    exit 1
fi

# 获取输入参数
structural_image=$1
structural_brain_flag=$2
reg_image=$3
reg_brain_flag=$4
target_image=$5
use_brain_flag=$6
direction_flag=$7
mask_file=""

# 检查是否指定了掩膜文件
if [ "$#" -eq 9 ] && [ "$8" == "-m" ]; then
    mask_file=$9
fi

# 定义输出路径
structural_dir=$(dirname "$structural_image")
reg_dir=$(dirname "$reg_image")
target_dir=$(dirname "$target_image")
target_basename=$(basename "$target_image" .nii.gz)  # 获取文件名，不含扩展名
target_basename=$(basename "$target_basename" .nii)  # 处理 .nii 情况

# 创建中间数据存储文件夹
intermediate_dir="$target_dir/${target_basename}_reg_intermediate_files"

# 如果已经存在文件夹，则输出提示并删除
if [ -d "$intermediate_dir" ]; then
    echo "Intermediate directory already exists, deleting..."
    rm -r "$intermediate_dir"
fi

mkdir -p "$intermediate_dir"

# Step 1: 剥脑处理（如果需要）
if [ "$structural_brain_flag" -eq 0 ]; then
    echo "Performing brain extraction on structural image..."
    bet2 "$structural_image" "$intermediate_dir/structural_brain" -f 0.2 -g 0 -m
    structural_brain="$intermediate_dir/structural_brain.nii.gz"
else
    structural_brain="$structural_image"
fi

if [ "$reg_brain_flag" -eq 0 ]; then
    echo "Performing brain extraction on registration image..."
    bet2 "$reg_image" "$intermediate_dir/reg_brain" -f 0.2 -g 0 -m
    reg_brain="$intermediate_dir/reg_brain.nii.gz"
else
    reg_brain="$reg_image"
fi

# Step 2: 配准步骤
if [ "$use_brain_flag" -eq 1 ]; then
    if [ "$direction_flag" -eq 0 ]; then
        echo "Registering structural (brain) to registration image (brain)..."
        flirt -in "$structural_brain" -ref "$reg_brain" -out "$intermediate_dir/structural_to_reg_brain.nii.gz" -omat "$intermediate_dir/structural_to_reg.mat"

        if [ -n "$mask_file" ]; then
            echo "Applying mask to structural_to_reg_brain image..."
            fslmaths "$intermediate_dir/structural_to_reg_brain.nii.gz" -mas "$mask_file" "$intermediate_dir/structural_to_reg_brain.nii.gz"
        fi

        echo "Registering structural (aligned with registration image) to MNI space..."
        flirt -in "$intermediate_dir/structural_to_reg_brain.nii.gz" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -out "$intermediate_dir/structural_to_MNI.nii.gz" -omat "$intermediate_dir/structural_to_MNI.mat" -dof 12
    elif [ "$direction_flag" -eq 1 ]; then
        if [ -n "$mask_file" ]; then
            echo "Applying mask to registration image..."
            fslmaths "$reg_brain" -mas "$mask_file" "$reg_brain"
        fi

        echo "Registering registration image (brain) to structural (brain)..."
        flirt -in "$reg_brain" -ref "$structural_brain" -out "$intermediate_dir/reg_to_structural_brain.nii.gz" -omat "$intermediate_dir/reg_to_structural.mat"
        flirt -in "$target_image" -ref "$structural_brain" -applyxfm -init "$intermediate_dir/reg_to_structural.mat" -out "$intermediate_dir/target_to_structural.nii.gz"

        echo "Registering structural to MNI space..."
        flirt -in "$structural_brain" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -out "$intermediate_dir/structural_to_MNI.nii.gz" -omat "$intermediate_dir/structural_to_MNI.mat" -dof 12
    else
        echo "Error: Invalid direction_flag. Must be 0 (T1 to reg) or 1 (reg to T1)."
        exit 1
    fi
else
    if [ "$direction_flag" -eq 0 ]; then
        echo "Registering structural to registration image..."
        flirt -in "$structural_image" -ref "$reg_image" -out "$intermediate_dir/structural_to_reg.nii.gz" -omat "$intermediate_dir/structural_to_reg.mat"
        
        if [ -n "$mask_file" ]; then
            echo "Applying mask to structural_to_reg image..."
            fslmaths "$intermediate_dir/structural_to_reg.nii.gz" -mas "$mask_file" "$intermediate_dir/structural_to_reg_brain.nii.gz"
        else   
            bet2 "$intermediate_dir/structural_to_reg" "$intermediate_dir/structural_to_reg_brain" -f 0.2 -g 0 -m
        fi

        echo "Registering structural (aligned with registration image) to MNI space..."
        flirt -in "$intermediate_dir/structural_to_reg_brain.nii.gz" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -out "$intermediate_dir/structural_to_MNI.nii.gz" -omat "$intermediate_dir/structural_to_MNI.mat" -dof 12
    elif [ "$direction_flag" -eq 1 ]; then
        if [ -n "$mask_file" ]; then
            echo "Mask file not used!"
        fi

        echo "Registering registration image to structural..."
        flirt -in "$reg_image" -ref "$structural_image" -out "$intermediate_dir/reg_to_structural.nii.gz" -omat "$intermediate_dir/reg_to_structural.mat"
        flirt -in "$target_image" -ref "$structural_image" -applyxfm -init "$intermediate_dir/reg_to_structural.mat" -out "$intermediate_dir/target_to_structural.nii.gz"

        echo "Registering structural to MNI space..."
        flirt -in "$structural_brain" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -out "$intermediate_dir/structural_to_MNI.nii.gz" -omat "$intermediate_dir/structural_to_MNI.mat" -dof 12
    else
        echo "Error: Invalid direction_flag. Must be 0 (T1 to reg) or 1 (reg to T1)."
        exit 1
    fi
fi

# Step 3: 将配准矩阵应用到目标图像
final_output="$target_dir/${target_basename}_MNI.nii.gz"
echo "Applying transformation to target image and saving as $final_output..."
if [ "$direction_flag" -eq 0 ]; then
    flirt -in "$target_image" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -applyxfm -init "$intermediate_dir/structural_to_MNI.mat" -out "$final_output" -dof 12
elif [ "$direction_flag" -eq 1 ]; then
    flirt -in "$intermediate_dir/target_to_structural.nii.gz" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -applyxfm -init "$intermediate_dir/structural_to_MNI.mat" -out "$final_output" -dof 12
fi

echo "Registration completed successfully! Final output: $final_output"
