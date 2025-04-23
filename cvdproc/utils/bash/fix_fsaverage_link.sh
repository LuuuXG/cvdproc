#!/bin/bash

# 目标链接的正确路径
correct_target="/usr/local/freesurfer/7-dev/subjects/fsaverage"

# 遍历所有 sub-* 目录
for subdir in /mnt/f/BIDS/SVD_BIDS/derivatives/freesurfer/sub-*; do
    link_path="$subdir/fsaverage"

    # 如果链接存在，删除它
    if [ -L "$link_path" ]; then
        echo "Removing old symlink: $link_path -> $(readlink "$link_path")"
        rm "$link_path"
    fi

    # 重新创建正确的软链接
    ln -s "$correct_target" "$link_path"
    echo "Created new symlink: $link_path -> $correct_target"
done
