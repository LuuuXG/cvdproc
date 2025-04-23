#!/bin/bash

# normalize multiple QSM images (native space) to MNI space using command-line arguments
# Required arguments:
# --t1w <T1w image>
# --mag <Magnitude image>
# --output <Output directory>
# --anat <FSL_ANAT directory>
# --input <One or more QSM images>

usage() {
    echo "Usage: $0 --t1w <T1w image> --mag <Magnitude image> --output <Output directory> --anat <FSL_ANAT directory> --input <QSM images...>"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --t1w)
            T1W_IMAGE="$2"; shift 2;;
        --mag)
            MAG_IMAGE="$2"; shift 2;;
        --output)
            OUTPUT_DIR="$2"; shift 2;;
        --anat)
            FSL_ANAT_DIR="$2"; shift 2;;
        --input)
            shift
            QSM_IMAGES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                QSM_IMAGES+=("$1")
                shift
            done;;
        *)
            echo "Unknown argument: $1"
            usage;;
    esac
done

# Validate required arguments
if [[ -z "$T1W_IMAGE" || -z "$MAG_IMAGE" || -z "$OUTPUT_DIR" || -z "$FSL_ANAT_DIR" || ${#QSM_IMAGES[@]} -eq 0 ]]; then
    usage
fi

# Check if the MNI template exists
if [ ! -f "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" ]; then
    echo "MNI template not found, exiting..."
    exit 1
fi

non_lin_wrap="$FSL_ANAT_DIR/T1_to_MNI_nonlin_field.nii.gz"
t1_orig2roi="$FSL_ANAT_DIR/T1_orig2roi.mat"
t1_roi="$FSL_ANAT_DIR/T1_biascorr.nii.gz"

# Get the mag2t1w(orig) transformation matrix
mri_synthstrip -i $T1W_IMAGE -o $OUTPUT_DIR/T1w_stripped.nii.gz
mri_synthstrip -i $MAG_IMAGE -o $OUTPUT_DIR/mag_stripped.nii.gz
flirt -in $OUTPUT_DIR/mag_stripped.nii.gz -ref $OUTPUT_DIR/T1w_stripped.nii.gz -omat $OUTPUT_DIR/mag2t1w.mat

for QSM_IMAGE in "${QSM_IMAGES[@]}"; do
    # Get the basename of QSM_IMAGE (remove .nii or .nii.gz)
    qsm_basename=$(basename $QSM_IMAGE)
    qsm_basename_noext=$(echo "$qsm_basename" | sed -E 's/\.nii(\.gz)?$//')

    # qsm -> t1w(orig) -> t1w(roi) -> MNI
    flirt -in $QSM_IMAGE -ref $T1W_IMAGE -out $OUTPUT_DIR/${qsm_basename_noext}_t1w -applyxfm -init $OUTPUT_DIR/mag2t1w.mat
    flirt -in $OUTPUT_DIR/${qsm_basename_noext}_t1w -ref $t1_roi -out $OUTPUT_DIR/${qsm_basename_noext}_t1w_roi -applyxfm -init $t1_orig2roi
    applywarp --ref=$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz --in=$OUTPUT_DIR/${qsm_basename_noext}_t1w_roi.nii.gz --out=$OUTPUT_DIR/${qsm_basename_noext}_MNI --warp=$non_lin_wrap

    # Alternative: ANTs
    # flirt -in "$OUTPUT_DIR/T1w_stripped" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -out "$OUTPUT_DIR/T1w2MNI_lin.nii.gz" -omat "$OUTPUT_DIR/T1w2MNI_lin.mat" -dof 12
    # antsRegistration -d 3 -m MI["$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz", "$OUTPUT_DIR/T1w2MNI_lin.nii.gz", 1, 32, Regular, 0.25] -c [1000x500x250x0,1e-7,5] -t SyN[0.25] -f 8x4x2x1 -s 4x2x1x0 -u 1 -o "$OUTPUT_DIR/T1affine"

    # flirt -in "$OUTPUT_DIR/${qsm_basename_noext}_t1w" -ref "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -applyxfm -init "$OUTPUT_DIR/T1w2MNI_lin.mat" -out "$OUTPUT_DIR/${qsm_basename_noext}_MNI_lin.nii.gz"
    # antsApplyTransforms -d 3 -i "$OUTPUT_DIR/${qsm_basename_noext}_MNI_lin.nii.gz" -o "${qsm_basename_noext}_MNI_ants.nii.gz" -r "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" -t "$OUTPUT_DIR/T1affine0Warp.nii.gz" -n Linear

    # Rename output file
    mv $OUTPUT_DIR/${qsm_basename_noext}_MNI.nii.gz $OUTPUT_DIR/${qsm_basename_noext}_MNI.nii.gz

    # Delete intermediate files
    #rm $OUTPUT_DIR/${qsm_basename_noext}_t1w.nii.gz
    rm $OUTPUT_DIR/${qsm_basename_noext}_t1w_roi.nii.gz
    echo "$QSM_IMAGE has been normalized to MNI space!"

done

# Cleanup common intermediate files
rm $OUTPUT_DIR/T1w_stripped.nii.gz
rm $OUTPUT_DIR/mag_stripped.nii.gz
