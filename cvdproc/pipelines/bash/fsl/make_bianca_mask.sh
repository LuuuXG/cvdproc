#!/usr/bin/env bash

set -e

fsl_anat_dir=$1
bianca_mask_out=$2

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <fsl_anat_dir> <bianca_mask_out>"
  exit 1
fi

if [[ ! -d "$fsl_anat_dir" ]]; then
  echo "Error: fsl_anat_dir '$fsl_anat_dir' does not exist or is not a directory."
  exit 1
fi

# output dir: dirname of bianca_mask_out
output_dir=$(dirname "$bianca_mask_out")
mkdir -p "$output_dir"

make_bianca_mask "${fsl_anat_dir}/T1_biascorr" "${fsl_anat_dir}/T1_fast_pve_0" "${fsl_anat_dir}/MNI_to_T1_nonlin_field.nii.gz"

bianca_mask="${fsl_anat_dir}/T1_biascorr_bianca_mask.nii.gz"

# transform to original T1 space
t1_orig="${fsl_anat_dir}/T1_orig.nii.gz"
t1_roi="${fsl_anat_dir}/T1_biascorr.nii.gz"
roi2orig_mat="${fsl_anat_dir}/T1_roi2orig.mat"

flirt -in "$bianca_mask" \
      -ref "$t1_orig" \
      -out "$bianca_mask_out" \
      -applyxfm \
      -init "$roi2orig_mat" \
      -interp nearestneighbour

echo "BIANCA mask saved to: $bianca_mask_out"