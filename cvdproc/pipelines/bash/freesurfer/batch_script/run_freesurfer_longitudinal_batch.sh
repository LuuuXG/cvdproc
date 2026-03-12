#!/bin/bash
set -euo pipefail

bids_dir="/mnt/f/BIDS/WCH_SVD_3T_BIDS"
jobs_file="/mnt/f/BIDS/WCH_SVD_3T_BIDS/code/freesurfer/jobs_longitudinal.csv"

# The per-subject longitudinal script (base creation).
# It must accept: <bids_dir> <subject_id_without_sub_prefix>
long_script="/mnt/f/BIDS/WCH_SVD_3T_BIDS/code/freesurfer/freesurfer_reconall_longitudinal_single.sh"

# -------------------------
# Basic checks
# -------------------------
if [[ ! -d "$bids_dir" ]]; then
  echo "ERROR: bids_dir not found: $bids_dir"
  exit 2
fi

if [[ ! -f "$jobs_file" ]]; then
  echo "ERROR: jobs_file not found: $jobs_file"
  exit 2
fi

if [[ ! -x "$long_script" ]]; then
  echo "ERROR: longitudinal script not executable or not found: $long_script"
  echo "HINT: chmod +x $long_script"
  exit 2
fi

# -------------------------
# Clean BOM and CRLF
# -------------------------
sed -i '1s/^\xEF\xBB\xBF//' "$jobs_file"
sed -i 's/\r$//' "$jobs_file"

# -------------------------
# Filter subjects: remove blank lines and comments
# -------------------------
tmp_subjects="$(mktemp)"
grep -vE '^\s*($|#)' "$jobs_file" | sed -E 's/^\s+//; s/\s+$//' > "$tmp_subjects"

if [[ ! -s "$tmp_subjects" ]]; then
  echo "ERROR: No valid subject IDs found in $jobs_file"
  rm -f "$tmp_subjects"
  exit 3
fi

# -------------------------
# Worker
# -------------------------
run_one () {
  local bids_dir="$1"
  local subject_id="$2"
  local long_script="$3"

  # Optional: normalize "sub-XXX" -> "XXX"
  subject_id="${subject_id#sub-}"

  bash "$long_script" "$bids_dir" "$subject_id"
}
export -f run_one

# -------------------------
# Run in parallel
# -------------------------
parallel -j 4 run_one "$bids_dir" {} "$long_script" :::: "$tmp_subjects"

rm -f "$tmp_subjects"
