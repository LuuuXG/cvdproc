#!/bin/bash

# This function processes a single subject
process_subject() {
    STRIPPED_FLAIR_IN_T1W_PATH=$1
    WMHMASK_IN_T1W_PATH=$2
    WMHPROBMAP_IN_T1W_PATH=$3
    FSLANAT_DIR=$4
    OUTPUT_DIR=$5

    # Set MNI template path
    MNI_PATH="$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz"  # MNI template path

    cd $FSLANAT_DIR
    # make_bianca_mask: create a binary mask
    make_bianca_mask T1_biascorr T1_fast_pve_0 MNI_to_T1_nonlin_field.nii.gz
            
    # generate the dist_to_vent_periventricular mask and transform it to the original space
    distancemap -i T1_biascorr_ventmask.nii.gz -o dist_to_vent
    fslmaths dist_to_vent -uthr 10 -bin dist_to_vent_periventricular_10mm
    fslmaths dist_to_vent -uthr 5 -bin dist_to_vent_periventricular_5mm
    fslmaths dist_to_vent -uthr 3 -bin dist_to_vent_periventricular_3mm
    fslmaths dist_to_vent -uthr 1 -bin dist_to_vent_periventricular_1mm

    flirt -in T1_biascorr_bianca_mask -ref T1_orig -out "$OUTPUT_DIR/T1_biascorr_bianca_mask_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in T1_biascorr_ventmask -ref T1_orig -out "$OUTPUT_DIR/T1_biascorr_ventmask_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_10mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_10mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_5mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_5mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_3mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_3mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour
    flirt -in dist_to_vent_periventricular_1mm -ref T1_orig -out "$OUTPUT_DIR/dist_to_vent_periventricular_1mm_orig" -applyxfm -init T1_roi2orig.mat -interp nearestneighbour

    flirt -in "$STRIPPED_FLAIR_IN_T1W_PATH" -out "$OUTPUT_DIR/FLAIR_roi" -ref "$FSLANAT_DIR/T1_biascorr.nii.gz" -applyxfm -init "$FSLANAT_DIR/T1_orig2roi.mat"
    applywarp --ref=$MNI_PATH --in="$OUTPUT_DIR/FLAIR_roi.nii.gz" --out="$OUTPUT_DIR/FLAIR_MNI" --warp="$FSLANAT_DIR/T1_to_MNI_nonlin_field.nii.gz"
    rm "$OUTPUT_DIR/FLAIR_roi.nii.gz"

    flirt -in "$WMHMASK_IN_T1W_PATH" -out "$OUTPUT_DIR/WMHmask_roi" -ref "$FSLANAT_DIR/T1_biascorr.nii.gz" -applyxfm -init "$FSLANAT_DIR/T1_orig2roi.mat"
    applywarp --ref=$MNI_PATH --in="$OUTPUT_DIR/WMHmask_roi.nii.gz" --out="$OUTPUT_DIR/WMHmask_in_MNI" --warp="$FSLANAT_DIR/T1_to_MNI_nonlin_field.nii.gz" --interp="nn"
    rm "$OUTPUT_DIR/WMHmask_roi.nii.gz"

    flirt -in "$WMHPROBMAP_IN_T1W_PATH" -out "$OUTPUT_DIR/WMHprobmap_roi" -ref "$FSLANAT_DIR/T1_biascorr.nii.gz" -applyxfm -init "$FSLANAT_DIR/T1_orig2roi.mat"
    applywarp --ref=$MNI_PATH --in="$OUTPUT_DIR/WMHprobmap_roi.nii.gz" --out="$OUTPUT_DIR/WMHprobmap_in_MNI" --warp="$FSLANAT_DIR/T1_to_MNI_nonlin_field.nii.gz"
    rm "$OUTPUT_DIR/WMHprobmap_roi.nii.gz"

    applywarp --ref=$MNI_PATH --in="$FSLANAT_DIR/T1_biascorr_ventmask.nii.gz" --out="$OUTPUT_DIR/T1_biascorr_ventmask_2_MNI" --warp="$FSLANAT_DIR/T1_to_MNI_nonlin_field.nii.gz" --interp="nn"

    echo "Processing for subject completed!"
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4 $5
