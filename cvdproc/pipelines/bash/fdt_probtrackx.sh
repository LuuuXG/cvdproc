#!/bin/bash

# FSL FDT probtrackX post-processing

process_subject() {
    BEDPOSTX_OUTPUT=$1
    PREPROCESS_PATH=$2
    FS_OUTPUT=$3
    SEED_MASK=$4 # in fs orig space
    OUTPUT_DIR=$5 # output dir of probtrackx, with 10000 --nsamples

    SUBJECTS_DIR_temp=$(dirname "${FS_OUTPUT}")
    SUBJECT_ID=$(basename "${FS_OUTPUT}")
    export SUBJECTS_DIR="${SUBJECTS_DIR_temp}"

    # project fdt_paths to subject surface
    mri_vol2surf --mov "${OUTPUT_DIR}/fdt_paths.nii.gz" \
     --o "${OUTPUT_DIR}/fdt_paths.mgh" \
     --regheader $SUBJECT_ID --hemi lh --projfrac 0.5

    mri_vol2surf --mov "${OUTPUT_DIR}/fdt_paths.nii.gz" \
     --o "${OUTPUT_DIR}/fdt_paths.mgh" \
     --regheader $SUBJECT_ID --hemi rh --projfrac 0.5
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4 $5