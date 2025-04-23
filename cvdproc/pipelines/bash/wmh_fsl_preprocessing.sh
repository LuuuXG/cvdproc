#!/bin/bash

# This script performs the following operations in parallel for each subject directory:
# 1. Register the T1 image to the FLAIR image
# 2. Generate the anatomical-related images and files using "fsl_anat"
# 3. Create a binary mask (using "make_bianca_mask") which is used to exclude areas of the brain that WMH are unlikely to be found
# 4. Generate the dist_to_vent_periventricular mask and transform it to the original space
# 5. Register the T1, FLAIR, and ventricle mask images to the MNI template

# This function processes a single subject
process_subject() {
    FLAIR_PATH=$1
    T1w_PATH=$2
    OUTPUT_DIR=$3

    echo "Processing subject with FLAIR: $FLAIR_PATH, T1w: $T1w_PATH, output will be saved to: $OUTPUT_DIR"

    # Set MNI template path
    MNI_PATH="$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz"  # MNI template path

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Register the T1 image to the FLAIR image
    flirt -ref "$FLAIR_PATH" -in "$T1w_PATH" -out "$OUTPUT_DIR/T1_2_FLAIR" -omat "$OUTPUT_DIR/T1_2_FLAIR.mat"

    rt1AnatDir="$OUTPUT_DIR/T1_2_FLAIR.anat"
    if [[ ! -d $rt1AnatDir ]]; then
        echo "T1_2_FLAIR.anat directory not found, performing the operation"
        # fsl_anat: generate anatomical-related images and files
        fsl_anat -i "$OUTPUT_DIR/T1_2_FLAIR" --clobber
    else
        echo "T1_2_FLAIR.anat directory already exists, skipping anatomical processing"
    fi

    cd $rt1AnatDir
    # make_bianca_mask: create a binary mask
    #mri_synthstrip -i T1_biascorr.nii.gz -o T1_biascorr_brain.nii.gz
    make_bianca_mask T1_biascorr T1_fast_pve_0 MNI_to_T1_nonlin_field.nii.gz
            
    # generate the dist_to_vent_periventricular mask and transform it to the original space
    distancemap -i T1_biascorr_ventmask.nii.gz -o dist_to_vent
    fslmaths dist_to_vent -uthr 10 -bin dist_to_vent_periventricular_10mm
    fslmaths dist_to_vent -uthr 5 -bin dist_to_vent_periventricular_5mm
    fslmaths dist_to_vent -uthr 3 -bin dist_to_vent_periventricular_3mm
    fslmaths dist_to_vent -uthr 1 -bin dist_to_vent_periventricular_1mm

    flirt -in T1_biascorr_brain_mask -ref T1_orig -out "$OUTPUT_DIR/T1_2_FLAIR_brain_mask" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in T1_biascorr_bianca_mask -ref T1_orig -out "$OUTPUT_DIR/T1_biascorr_bianca_mask_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in T1_biascorr_ventmask -ref T1_orig -out "$OUTPUT_DIR/T1_biascorr_ventmask_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_10mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_10mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_5mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_5mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_3mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_3mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_1mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_1mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour

    # Register the FLAIR, and ventricle mask images to the MNI template
    fslmaths "$FLAIR_PATH" -mas "$OUTPUT_DIR/T1_2_FLAIR_brain_mask" "$OUTPUT_DIR/FLAIR_brain"

    flirt -in "$OUTPUT_DIR/FLAIR_brain" -out "$OUTPUT_DIR/FLAIR_brain_roi" -ref "$rt1AnatDir/T1_biascorr.nii.gz"  -applyxfm -init "$rt1AnatDir/T1_orig2roi.mat"
    applywarp --ref=$MNI_PATH --in="$OUTPUT_DIR/FLAIR_brain_roi.nii.gz" --out="$OUTPUT_DIR/FLAIR_2_MNI_brain" --warp="$rt1AnatDir/T1_to_MNI_nonlin_field.nii.gz"
    rm "$OUTPUT_DIR/FLAIR_brain_roi.nii.gz"

    applywarp --ref=$MNI_PATH --in="$rt1AnatDir/T1_biascorr_ventmask.nii.gz" --out="$OUTPUT_DIR/T1_biascorr_ventmask_2_MNI" --warp="$rt1AnatDir/T1_to_MNI_nonlin_field.nii.gz" --interp="nn"

    echo "Processing for subject completed!"
}

# Call the function with the provided paths
process_subject $1 $2 $3
