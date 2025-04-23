#!/bin/bash

# preprocess for probtrackX

process_subject() {
    FS_OUTPUT=$1
    T1w_PATH=$2
    FA_PATH=$3
    SEED_PATH=$4 # in FA/DTI space
    OUTPUT_DIR=$5

    # Get the transformation matrix between T1w, freesurfer and diffusion
    tkregister2 --mov "${FS_OUTPUT}/mri/orig.mgz" \
            --targ "${FS_OUTPUT}/mri/rawavg.mgz" \
            --regheader \
            --reg junk \
            --fslregout "${OUTPUT_DIR}/freesurfer2struct.mat" \
            --noedit
    
    mri_convert "${FS_OUTPUT}/mri/orig.mgz" \
            "${OUTPUT_DIR}/orig.nii.gz"
    
    convert_xfm -omat "${OUTPUT_DIR}/struct2freesurfer.mat" \
            -inverse "${OUTPUT_DIR}/freesurfer2struct.mat"
    
    mri_synthstrip -i "${T1w_PATH}" \
            -o "${OUTPUT_DIR}/T1w_brain.nii.gz" \
            -m "${OUTPUT_DIR}/T1w_brain_mask.nii.gz"
    
    flirt -in "${FA_PATH}" \
    -ref "${OUTPUT_DIR}/T1w_brain.nii.gz" \
    -omat "${OUTPUT_DIR}/fa2struct.mat"

    convert_xfm -omat "${OUTPUT_DIR}/struct2fa.mat" \
            -inverse "${OUTPUT_DIR}/fa2struct.mat"

    convert_xfm -omat "${OUTPUT_DIR}/fa2freesurfer.mat" \
                -concat "${OUTPUT_DIR}/struct2freesurfer.mat" "${OUTPUT_DIR}/fa2struct.mat"

    convert_xfm -omat "${OUTPUT_DIR}/freesurfer2fa.mat" \
                -inverse "${OUTPUT_DIR}/fa2freesurfer.mat"
    
    # transform the seed mask to freesurfer space
    flirt -in "${SEED_PATH}" \
    -ref "${OUTPUT_DIR}/orig.nii.gz" \
    -applyxfm -init "${OUTPUT_DIR}/fa2freesurfer.mat" \
    -out "${OUTPUT_DIR}/seed_mask_in_fs.nii.gz" \
    -interp nearestneighbour

    # transform the seed mask to t1w
    flirt -in "${SEED_PATH}" \
        -ref "${OUTPUT_DIR}/T1w_brain.nii.gz" \
        -applyxfm -init "${OUTPUT_DIR}/fa2struct.mat" \
        -out "${OUTPUT_DIR}/seed_mask_in_t1w.nii.gz" \
        -interp nearestneighbour

    # get the surface mask from freesurfer output
    mri_convert "${FS_OUTPUT}/mri/lh.ribbon.mgz" "${OUTPUT_DIR}/lh.ribbon.nii.gz"
    mri_convert "${FS_OUTPUT}/mri/rh.ribbon.mgz" "${OUTPUT_DIR}/rh.ribbon.nii.gz"

    fslmaths "${OUTPUT_DIR}/lh.ribbon.nii.gz" \
        -add "${OUTPUT_DIR}/rh.ribbon.nii.gz" \
        "${OUTPUT_DIR}/cortical_GM.nii.gz"

    # binarize the cortical GM mask
    fslmaths "${OUTPUT_DIR}/cortical_GM.nii.gz" \
        -bin "${OUTPUT_DIR}/cortical_GM.nii.gz"

    rm "${OUTPUT_DIR}/lh.ribbon.nii.gz" "${OUTPUT_DIR}/rh.ribbon.nii.gz"

    flirt -in "${OUTPUT_DIR}/cortical_GM.nii.gz" \
        -ref "${FA_PATH}" \
        -applyxfm -init "${OUTPUT_DIR}/freesurfer2fa.mat" \
        -out "${OUTPUT_DIR}/cortical_GM_in_dif.nii.gz" \
        -interp nearestneighbour
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4 $5