#!/bin/bash

# wrapper for DRBUDDI with retry and CPU fallback

# 保存原始参数
args=("$@")

# 最大尝试次数
max_tries=5
success=0

# CUDA版本尝试
for ((i=1; i<=max_tries; i++)); do
    echo ">>> Attempt $i running DRBUDDI_cuda..."
    DRBUDDI_cuda "${args[@]}"
    status=$?
    if [ $status -eq 0 ]; then
        echo ">>> DRBUDDI_cuda succeeded on attempt $i"
        success=1
        break
    else
        echo ">>> DRBUDDI_cuda failed (exit code $status), retrying..."
    fi
done

# 如果CUDA版失败，尝试CPU版
if [ $success -eq 0 ]; then
    echo ">>> DRBUDDI_cuda failed after $max_tries attempts, falling back to CPU version..."

    # 将参数数组转换为字符串
    arg_str="${args[@]}"

    # 如果有--ncores参数，替换为--ncores 8，否则直接加上
    if [[ "$arg_str" =~ --ncores[[:space:]]+[0-9]+ ]]; then
        echo ">>> Replacing existing --ncores with --ncores 8"
        arg_str=$(echo "$arg_str" | sed -E 's/--ncores[[:space:]]+[0-9]+/--ncores 8/')
    else
        echo ">>> Adding --ncores 8"
        arg_str="$arg_str --ncores 8"
    fi

    # 运行CPU版
    DRBUDDI $arg_str
    status=$?

    if [ $status -eq 0 ]; then
        echo ">>> DRBUDDI (CPU) succeeded"
    else
        echo ">>> DRBUDDI (CPU) failed (exit code $status)"
        exit $status
    fi
fi
