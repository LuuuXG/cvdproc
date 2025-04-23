#!/bin/bash

# This function processes a single subject
process_subject() {
    BULLSEYES_INPUT_DIR=$1
    FREESURFER_DIR=$2
    FREESURFER_SUBJECT=$3
    T12FLAIR_MAT=$4
    FLAIR_PATH=$5
    OUTPUT_DIR=$6

    FULL_FREESURFER_DIR="$FREESURFER_DIR/$FREESURFER_SUBJECT"

    tkregister2 --mov "$FULL_FREESURFER_DIR/mri/orig.mgz" --targ "$FULL_FREESURFER_DIR/mri/rawavg.mgz" --regheader --reg junk --fslregout "$OUTPUT_DIR/freesurfer2struct.mat" --noedit
    convert_xfm -omat "$OUTPUT_DIR/struct2freesurfer.mat" -inverse "$OUTPUT_DIR/freesurfer2struct.mat"
    
    convert_xfm -omat "$OUTPUT_DIR/FLAIR_2_T1.mat" -inverse "$T12FLAIR_MAT"

    convert_xfm -omat "$OUTPUT_DIR/flair2freesurfer.mat" -concat "$OUTPUT_DIR/struct2freesurfer.mat" "$OUTPUT_DIR/FLAIR_2_T1.mat"
    convert_xfm -omat "$OUTPUT_DIR/freesurfer2flair.mat" -inverse "$OUTPUT_DIR/flair2freesurfer.mat"
    
    flirt -in "$BULLSEYES_INPUT_DIR/bullseye_wmparc.nii.gz" -ref "$FLAIR_PATH" -applyxfm -init "$OUTPUT_DIR/freesurfer2flair.mat" -out "$OUTPUT_DIR/bullseye_wmparc_2_flair.nii.gz" -interp nearestneighbour

    echo "Bullseyes preprocessing done for subject $FREESURFER_SUBJECT"
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4 $5 $6
