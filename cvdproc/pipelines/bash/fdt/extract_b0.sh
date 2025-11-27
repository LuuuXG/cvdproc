#!/bin/bash

set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <input_dwi> <input_bval> <output_dir> <output_b0_filename> <output_b0_mask_filename>"
  exit 1
fi

input_dwi=$1
input_bval=$2
output_dir=$3
output_b0_filename=$4
output_b0_mask_filename=$5

# Create output directory if it does not exist
mkdir -p "${output_dir}"

output_b0="${output_dir}/${output_b0_filename}"
output_b0_mask="${output_dir}/${output_b0_mask_filename}"

# Temporary files
tmp_b0_4d="${output_dir}/tmp_b0_4d.nii.gz"

echo "Input DWI: ${input_dwi}"
echo "Input bval: ${input_bval}"
echo "Output directory: ${output_dir}"
echo "Output b0 image: ${output_b0}"
echo "Output b0 mask: ${output_b0_mask}"

###############################################
# 1. Extract all b0 volumes (b < 100) and average
###############################################

# Get indices (0-based) of b0 volumes (b < 100)
b0_indices=$(awk '{
  for (i = 1; i <= NF; i++) {
    if ($i < 100) {
      # 0-based index for NIfTI volumes
      printf("%d,", i-1);
    }
  }
}' "${input_bval}" | sed 's/,$//')

if [ -z "${b0_indices}" ]; then
  echo "Error: no b0 volumes found (no b < 100 in ${input_bval})."
  exit 1
fi

echo "b0 volume indices (0-based): ${b0_indices}"

# Extract b0 volumes into a 4D file
fslselectvols -i "${input_dwi}" -o "${tmp_b0_4d}" --vols="${b0_indices}"

# Average across time (b0 volumes) to get a single 3D b0 image
fslmaths "${tmp_b0_4d}" -Tmean "${output_b0}"

echo "Mean b0 saved to: ${output_b0}"

###############################################
# 2. Run mri_synthstrip to create a brain mask
###############################################

# # We also create a brain-extracted b0 image (not passed as an argument, internal use)
# b0_brain="${output_dir}/$(basename "${output_b0_filename%.*}")_brain.nii.gz"

mri_synthstrip \
  -i "${output_b0}" \
  -m "${output_b0_mask}"

echo "Brain mask saved to: ${output_b0_mask}"

# Optional: remove temporary files
rm -f "${tmp_b0_4d}"

echo "Done."
