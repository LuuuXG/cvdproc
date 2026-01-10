#!/bin/bash

set -e

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
        echo "Resampling DWI to isotropic ${output_resolution} mm..."

        # Report original spacing
        dx=$(fslval "${eddy_dwi}" pixdim1)
        dy=$(fslval "${eddy_dwi}" pixdim2)
        dz=$(fslval "${eddy_dwi}" pixdim3)
        dt=$(fslval "${eddy_dwi}" pixdim4)
        echo "Original spacing: ${dx} ${dy} ${dz} ${dt}"

        spacing_str="${output_resolution}x${output_resolution}x${output_resolution}x${dt}"
        echo "Resampling to spacing: ${spacing_str} (BSpline)"

        ResampleImage 4 "${eddy_dwi}" "${new_dwi}" "${spacing_str}" 0 4

        # # Temporary directories
        # dwi_split_dir="${new_output_dir}/dwi_split_temp"
        # dwi_resampled_dir="${new_output_dir}/dwi_resampled_temp"
        # mkdir -p "${dwi_split_dir}" "${dwi_resampled_dir}"

        # # 1) Split 4D DWI into 3D volumes
        # echo "Splitting 4D DWI into 3D volumes..."
        # fslsplit "${eddy_dwi}" "${dwi_split_dir}/dwi_vol_" -t

        # # 2) Create a 2mm reference image from the first volume
        # first_vol=$(ls "${dwi_split_dir}"/dwi_vol_*.nii.gz | sort | head -n 1)
        # if [[ -z "${first_vol}" ]]; then
        #     echo "Error: no split volumes found in ${dwi_split_dir}"
        #     exit 1
        # fi

        # ref_image="${new_output_dir}/ref_${output_resolution}mm.nii.gz"
        # echo "Creating reference image ${ref_image} with ${output_resolution} mm isotropic voxels..."
        # ResampleImage 3 \
        #     "${first_vol}" \
        #     "${ref_image}" \
        #     "${output_resolution}x${output_resolution}x${output_resolution}" \
        #     0 \
        #     1

        # # 3) Resample each 3D volume to the reference grid using BSpline interpolation
        # echo "Resampling each 3D volume with antsApplyTransforms (BSpline)..."
        # for vol in "${dwi_split_dir}"/dwi_vol_*.nii.gz; do
        #     fname=$(basename "${vol}")
        #     antsApplyTransforms \
        #         -d 3 \
        #         -i "${vol}" \
        #         -r "${ref_image}" \
        #         -n BSpline[3] \
        #         -o "${dwi_resampled_dir}/${fname}" \
        #         --float \
        #         --default-value 0
        # done

        # # 4) Merge resampled 3D volumes back into 4D DWI
        # echo "Merging resampled volumes into 4D DWI: ${new_dwi}"
        # fslmerge -t "${new_dwi}" "${dwi_resampled_dir}"/dwi_vol_*.nii.gz

        # # Optional: clean up temporary directories
        # rm -rf "${dwi_split_dir}" "${dwi_resampled_dir}"
        # rm "${ref_image}"
        # echo "Resampling completed: ${new_dwi}"
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