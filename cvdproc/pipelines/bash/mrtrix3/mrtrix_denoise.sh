#!/usr/bin/env bash
set -e

# === Argument parsing ===
DWI_NIFTI=$1
DWI_BVEC=$2
DWI_BVAL=$3
OUTPUT_DIR=$4
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <DWI.nii.gz> <DWI.bvec> <DWI.bval> <output_dir>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

mrconvert "$DWI_NIFTI" "$OUTPUT_DIR/dwi4denoise.mif" -fslgrad "$DWI_BVEC" "$DWI_BVAL" -force

dwidenoise "$OUTPUT_DIR/dwi4denoise.mif" "$OUTPUT_DIR/dwi_denoise.mif" -force -nthreads 8

#mrdegibbs "$OUTPUT_DIR/dwi_denoise.mif" "$OUTPUT_DIR/dwi_degibbs.mif" -nthreads 4 -force

# convert to NIfTI format
mrconvert "$OUTPUT_DIR/dwi_denoise.mif" "$OUTPUT_DIR/dwi_denoise.nii.gz" -export_grad_fsl "$OUTPUT_DIR/dwi_denoise.bvec" "$OUTPUT_DIR/dwi_denoise.bval" -force
# clean up intermediate files
rm -f "$OUTPUT_DIR/dwi4denoise.mif" "$OUTPUT_DIR/dwi_denoise.mif"
