#!/usr/bin/env bash

set -e

fs_subjects_dir=$1
fs_subject_id=$2
output_gm_mask=$3

# merge lh.ribbon.mgz and rh.ribbon.mgz to create a gm mask, and delete temporary files
output_dir=$(dirname "${output_gm_mask}")
mkdir -p "${output_dir}"

# make a temp dir in output dir
temp_dir=$(mktemp -d -p "${output_dir}")
mri_convert "${fs_subjects_dir}/${fs_subject_id}/mri/lh.ribbon.mgz" "${temp_dir}/lh.ribbon.nii.gz"
mri_convert "${fs_subjects_dir}/${fs_subject_id}/mri/rh.ribbon.mgz" "${temp_dir}/rh.ribbon.nii.gz"

fslmaths "${temp_dir}/lh.ribbon.nii.gz" \
    -add "${temp_dir}/rh.ribbon.nii.gz" \
    -bin "${output_gm_mask}"

# remove temp dir
rm -rf "${temp_dir}"

echo "Cortical GM mask created at: ${output_gm_mask}"