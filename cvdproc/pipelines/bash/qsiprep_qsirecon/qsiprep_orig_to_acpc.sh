#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Usage
# -------------------------
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <subject_id> <session_id> <fa_file> <preprocess_t1w> <orig_t1w> <output_dir>"
  exit 1
fi

subject_id="$1"
session_id="$2"
fa_file="$3"
preprocess_t1w="$4"   # ACPC T1w (1mm) aligned with DWI ACPC (only resolution differs)
orig_t1w="$5"         # original T1w
output_dir="$6"

# -------------------------
# Basic checks
# -------------------------
for f in "$fa_file" "$preprocess_t1w" "$orig_t1w"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: File not found: $f"
    exit 2
  fi
done

mkdir -p "$output_dir"

# -------------------------
# Derive output FA name
# -------------------------
fa_base="$(basename "$fa_file")"

if [[ "$fa_base" == *"space-ACPC"* ]]; then
  output_fa_file="${output_dir}/${fa_base/space-ACPC/space-T1w}"
else
  if [[ "$fa_base" == *.nii.gz ]]; then
    output_fa_file="${output_dir}/${fa_base%.nii.gz}_space-T1w.nii.gz"
  elif [[ "$fa_base" == *.nii ]]; then
    output_fa_file="${output_dir}/${fa_base%.nii}_space-T1w.nii"
  else
    output_fa_file="${output_dir}/${fa_base}_space-T1w.nii.gz"
  fi
fi

# -------------------------
# Output mats
# -------------------------
# 1) T1_ACPC -> T1_orig (estimated with optimization)
t1acpc2orig_tmp1="${output_dir}/tmp_t1acpc2orig_1.mat"
t1acpc2orig_tmp2="${output_dir}/tmp_t1acpc2orig_2.mat"
t1acpc2orig_mat="${output_dir}/sub-${subject_id}_ses-${session_id}_from-ACPCT1w_to-T1w_xfm.mat"

# 2) FA_ACPC -> T1_ACPC (header-only, no optimization)
fa2t1acpc_mat="${output_dir}/sub-${subject_id}_ses-${session_id}_from-ACPCFA_to-ACPCT1w_xfm.mat"

# 3) Final: FA_ACPC -> T1_orig (concatenated)
fa2orig_mat="${output_dir}/sub-${subject_id}_ses-${session_id}_from-ACPC_to-T1w_xfm.mat"

# -------------------------
# FLIRT options
# -------------------------
cost="normmi"
usesqform_flag="-usesqform"
search_flags="-searchrx -60 60 -searchry -60 60 -searchrz -60 60"

# -------------------------
# Step 1: Estimate T1_ACPC -> T1_orig (this is the main registration)
# -------------------------
flirt \
  -in "$preprocess_t1w" \
  -ref "$orig_t1w" \
  -omat "$t1acpc2orig_tmp1" \
  -dof 6 \
  -cost "$cost" \
  $usesqform_flag \
  $search_flags

flirt \
  -in "$preprocess_t1w" \
  -ref "$orig_t1w" \
  -init "$t1acpc2orig_tmp1" \
  -omat "$t1acpc2orig_tmp2" \
  -dof 6 \
  -cost "$cost" \
  $usesqform_flag \
  -nosearch

cp "$t1acpc2orig_tmp2" "$t1acpc2orig_mat"

# -------------------------
# Step 2: Compute FA_ACPC -> T1_ACPC using header only (no optimization)
# -------------------------
# This assumes FA_ACPC and T1_ACPC are already aligned in physical space and differ only by sampling.
flirt \
  -in "$fa_file" \
  -ref "$preprocess_t1w" \
  -dof 6 \
  -cost "$cost" \
  $usesqform_flag \
  -nosearch \
  -omat "$fa2t1acpc_mat" 
# -------------------------
# Step 3: Concatenate to get FA_ACPC -> T1_orig
#   (T1_ACPC -> T1_orig) o (FA_ACPC -> T1_ACPC)
# -------------------------
convert_xfm \
  -omat "$fa2orig_mat" \
  -concat "$t1acpc2orig_mat" "$fa2t1acpc_mat"

# -------------------------
# Step 4: Apply final transform to FA
# -------------------------
flirt \
  -in "$fa_file" \
  -ref "$orig_t1w" \
  -applyxfm -init "$fa2orig_mat" \
  -out "$output_fa_file"

# -------------------------
# Cleanup
# -------------------------
rm -f "$t1acpc2orig_tmp1" "$t1acpc2orig_tmp2" "$t1acpc2orig_mat" "$fa2t1acpc_mat"

echo "Done."
# echo "Outputs:"
# echo "  T1_ACPC->T1_orig: $t1acpc2orig_mat"
# echo "  FA_ACPC->T1_ACPC (header-only): $fa2t1acpc_mat"
# echo "  FA_ACPC->T1_orig (final): $fa2orig_mat"
# echo "  FA resampled: $output_fa_file"
