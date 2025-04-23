#!/bin/bash

# 记录脚本开始时间
start_time=$(date +%s)

# 输入目录
input_directory="/mnt/e/Neuroimage/QSM_test/HC45/freesurfer"

# 输出目录
output_directory="/mnt/e/Neuroimage/QSM_test/HC45/freesurfer"

# 设置 FreeSurfer 的 SUBJECTS_DIR 环境变量
export SUBJECTS_DIR=$output_directory

# 确保输出目录存在
mkdir -p $SUBJECTS_DIR

# 使用 GNU Parallel 并行运行 recon-all
find $input_directory -name "sub-*_T1w.nii" | parallel --progress --eta -j 10 \
    recon-all -s {/.}_output -i {} -all

# 计算并打印总共花费的时间
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "所有 FreeSurfer recon-all 任务已完成。"
echo "总共花费时间：$elapsed 秒。"
