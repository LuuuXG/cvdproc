#!/bin/bash

# Order fdt eddy output files to another directory

process_subject() {
    eddy_output_dir=$1
    eddy_output_filename=$2
    new_output_dir=$3
    new_output_filename=$4
    bval=$5
    output_resolution=$6

    mkdir -p "${new_output_dir}"

    # find eddy corrected dwi and bvec files
    eddy_dwi="${eddy_output_dir}/${eddy_output_filename}.nii.gz"
    eddy_bvec="${eddy_output_dir}/${eddy_output_filename}.eddy_rotated_bvecs"
    eddy_bval=$bval
    
    # target new output file
    new_dwi="${new_output_dir}/${new_output_filename}.nii.gz"
    new_bvec="${new_output_dir}/${new_output_filename}.bvec"
    new_bval="${new_output_dir}/${new_output_filename}.bval"

    # Copy eddy corrected files to new output directory (if new files do not exist)
    if [[ ! -f "${new_dwi}" ]]; then
        #cp "${eddy_dwi}" "${new_dwi}"
        dx=$(fslval "${eddy_dwi}" pixdim1)
        dy=$(fslval "${eddy_dwi}" pixdim2)
        dz=$(fslval "${eddy_dwi}" pixdim3)
        dt=$(fslval "${eddy_dwi}" pixdim4)
        echo "Original spacing: ${dx} ${dy} ${dz} ${dt}"

        spacing_str="${output_resolution}x${output_resolution}x${output_resolution}x${dt}"
        echo "Resampling to spacing: ${spacing_str} (BSpline)"

        ResampleImage 4 "${eddy_dwi}" "${new_dwi}" "${spacing_str}" 0 4
    else
        echo "File ${new_dwi} already exists, skipping resampling."
    fi
    if [[ ! -f "${new_bvec}" ]]; then
        cp "${eddy_bvec}" "${new_bvec}"
    else
        echo "File ${new_bvec} already exists, skipping copy."
    fi
    if [[ ! -f "${new_bval}" ]]; then
        cp "${eddy_bval}" "${new_bval}"
    else
        echo "File ${new_bval} already exists, skipping copy."
    fi
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4 $5 $6