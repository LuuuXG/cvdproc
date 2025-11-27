#!/usr/bin/env bash

set -e

fs_subjects_dir=$1
fs_subject_id=$2
output_matrix=$3
output_inverse_matrix=$4

# tkregister2 --mov "${FS_OUTPUT}/mri/orig.mgz" \
#             --targ "${FS_OUTPUT}/mri/rawavg.mgz" \
#             --regheader \
#             --reg junk \
#             --fslregout "${OUTPUT_DIR}/freesurfer2struct.mat" \
#             --noedit

output_dir=$(dirname $output_matrix)
mkdir -p $output_dir

tkregister2 --mov "${fs_subjects_dir}/${fs_subject_id}/mri/orig.mgz" \
            --targ "${fs_subjects_dir}/${fs_subject_id}/mri/rawavg.mgz" \
            --regheader \
            --reg junk \
            --fslregout "$output_matrix" \
            --noedit

convert_xfm -omat "$output_inverse_matrix" \
            -inverse "$output_matrix"

echo "Transformation matrices saved to:"
echo "  $output_matrix"
echo "  $output_inverse_matrix"