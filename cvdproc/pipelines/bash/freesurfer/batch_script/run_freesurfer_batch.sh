#!/bin/bash
set -e

bids_dir="/mnt/f/BIDS/WCH_SVD_3T_BIDS"
jobs_file="/mnt/f/BIDS/WCH_SVD_3T_BIDS/code/freesurfer/jobs.csv"

# 清理 BOM 和 CRLF
sed -i '1s/^\xEF\xBB\xBF//' "$jobs_file"
sed -i 's/\r$//' "$jobs_file"

run_one () {
  local bids_dir=$1
  local subject=$2
  local session=$3
  /mnt/f/BIDS/WCH_SVD_3T_BIDS/code/freesurfer/freesurfer_reconall_single.sh "$bids_dir" "$subject" "$session"
}
export -f run_one

# 直接用 parallel 读 CSV
parallel -C, -j 4 run_one "$bids_dir" {1} {2} :::: "$jobs_file"
