#!/bin/bash

# This function processes a single subject
process_subject() {
    TWMH_file=$1
    PWMH_file=$2
    DWMH_file=$3
    VENTMASK_IN_MNI=$4
    FSLANAT_DIR=$5
    OUTPUT_DIR=$6

    # Set MNI template path
    MNI_PATH="$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz"  # MNI template path

    # register the WMH mask image to the MNI template
    flirt -in "$TWMH_file" -out "$OUTPUT_DIR/TWMH_roi" -ref "$FSLANAT_DIR/T1_biascorr.nii.gz"  -applyxfm -init "$FSLANAT_DIR/T1_orig2roi.mat"
    applywarp --ref=$MNI_PATH --in="$OUTPUT_DIR/TWMH_roi.nii.gz" --out="$OUTPUT_DIR/TWMH_MNI.nii.gz" --warp="$FSLANAT_DIR/T1_to_MNI_nonlin_field.nii.gz" --interp="nn"
    rm "$OUTPUT_DIR/TWMH_roi.nii.gz"

    flirt -in "$PWMH_file" -out "$OUTPUT_DIR/PWMH_roi" -ref "$FSLANAT_DIR/T1_biascorr.nii.gz"  -applyxfm -init "$FSLANAT_DIR/T1_orig2roi.mat"
    applywarp --ref=$MNI_PATH --in="$OUTPUT_DIR/PWMH_roi.nii.gz" --out="$OUTPUT_DIR/PWMH_MNI.nii.gz" --warp="$FSLANAT_DIR/T1_to_MNI_nonlin_field.nii.gz" --interp="nn"
    rm "$OUTPUT_DIR/PWMH_roi.nii.gz"

    flirt -in "$DWMH_file" -out "$OUTPUT_DIR/DWMH_roi" -ref "$FSLANAT_DIR/T1_biascorr.nii.gz"  -applyxfm -init "$FSLANAT_DIR/T1_orig2roi.mat"
    applywarp --ref=$MNI_PATH --in="$OUTPUT_DIR/DWMH_roi.nii.gz" --out="$OUTPUT_DIR/DWMH_MNI.nii.gz" --warp="$FSLANAT_DIR/T1_to_MNI_nonlin_field.nii.gz" --interp="nn"
    rm "$OUTPUT_DIR/DWMH_roi.nii.gz"

    # 根据PWMH优化侧脑室mask，避免PWMH与侧脑室mask重叠
    fslmaths "$OUTPUT_DIR/PWMH_MNI" -mul "$VENTMASK_IN_MNI" "$OUTPUT_DIR/overlap"
    fslmaths "$VENTMASK_IN_MNI" -sub "$OUTPUT_DIR/overlap" "$VENTMASK_IN_MNI"
    rm "$OUTPUT_DIR/overlap.nii.gz"

    echo "WMH transformation to MNI space completed!"
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4 $5 $6
