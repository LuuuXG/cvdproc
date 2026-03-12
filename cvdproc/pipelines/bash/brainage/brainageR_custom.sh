#!/usr/bin/env bash
set -euo pipefail

t1w_image=$1
output_dir=$2
output_csv_filename=$3

mkdir -p "$output_dir"

# Uncompress the T1w image (.nii.gz -> .nii)

# Filename: Fetch from t1w_image
t1w_image_filename=$(basename "$t1w_image" .nii.gz)

gunzip -c "$t1w_image" > "${output_dir}/${t1w_image_filename}.nii"

# if not 1*1*1, reslice to 1mm iso
if ! [ "$(fslinfo "${output_dir}/${t1w_image_filename}.nii" | grep 'pixdim1' | awk '{print $2}')" == "1.000000" ] || ! [ "$(fslinfo "${output_dir}/${t1w_image_filename}.nii" | grep 'pixdim2' | awk '{print $2}')" == "1.000000" ] || ! [ "$(fslinfo "${output_dir}/${t1w_image_filename}.nii" | grep 'pixdim3' | awk '{print $2}')" == "1.000000" ]; then
  # Reslice to 1mm isotropic
  flirt -in "${output_dir}/${t1w_image_filename}.nii" -ref "${output_dir}/${t1w_image_filename}.nii" -out "${output_dir}/${t1w_image_filename}_1mm.nii.gz" -applyisoxfm 1
  gunzip -c "${output_dir}/${t1w_image_filename}_1mm.nii.gz" > "${output_dir}/${t1w_image_filename}.nii"

  rm -f "${output_dir}/${t1w_image_filename}_1mm.nii.gz"
fi

# brainageR
brainageR -f "${output_dir}/${t1w_image_filename}.nii" -o "${output_dir}/${output_csv_filename}"

# Post-processing

# clear nii file
rm -f "${output_dir}/${t1w_image_filename}.nii"

# rename qc dir (-> QC)
qc_dir="${output_dir}/slicesdir_${t1w_image_filename}.nii"
mv "$qc_dir" "${output_dir}/QC"
