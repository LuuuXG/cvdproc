#!/bin/bash

# Register one modality to another with optional skull stripping

process_subject() {
    IMAGE_TARGET=$1
    IMAGE_TARGET_STRIP=$2
    IMAGE_SOURCE=$3
    IMAGE_SOURCE_STRIP=$4
    FLIRT_DIRECTION=$5
    OUTPUT_DIR=$6

    REGISTERED_IMAGE_NAME=$7
    SOURCE_TO_TARGET_MAT=$8
    TARGET_TO_SOURCE_MAT=$9

    DOF=${10}

    # Get base names without .nii or .nii.gz
    IMAGE_TARGET_NAME=$(basename "$IMAGE_TARGET")
    IMAGE_TARGET_NAME=${IMAGE_TARGET_NAME%.nii.gz}
    IMAGE_TARGET_NAME=${IMAGE_TARGET_NAME%.nii}

    IMAGE_SOURCE_NAME=$(basename "$IMAGE_SOURCE")
    IMAGE_SOURCE_NAME=${IMAGE_SOURCE_NAME%.nii.gz}
    IMAGE_SOURCE_NAME=${IMAGE_SOURCE_NAME%.nii}

    # Strip skull if needed
    if [ $IMAGE_TARGET_STRIP -eq 0 ]; then
        echo "Stripping the target image"
        MASK_PATH="${OUTPUT_DIR}/${IMAGE_TARGET_NAME}_mask.nii.gz"
        IMAGE_TARGET_STRIPPED="${OUTPUT_DIR}/${IMAGE_TARGET_NAME}_brain.nii.gz"
        #mri_synthstrip -i $IMAGE_TARGET -o $IMAGE_TARGET_STRIPPED -m $MASK_PATH
        mri_synthstrip -i $IMAGE_TARGET -o $IMAGE_TARGET_STRIPPED
    else
        echo "Target image already stripped"
        IMAGE_TARGET_STRIPPED=$IMAGE_TARGET
    fi

    if [ $IMAGE_SOURCE_STRIP -eq 0 ]; then
        echo "Stripping the source image"
        MASK_PATH="${OUTPUT_DIR}/${IMAGE_SOURCE_NAME}_mask.nii.gz"
        IMAGE_SOURCE_STRIPPED="${OUTPUT_DIR}/${IMAGE_SOURCE_NAME}_brain.nii.gz"
        #mri_synthstrip -i $IMAGE_SOURCE -o $IMAGE_SOURCE_STRIPPED -m $MASK_PATH
        mri_synthstrip -i $IMAGE_SOURCE -o $IMAGE_SOURCE_STRIPPED
    else
        echo "Source image already stripped"
        IMAGE_SOURCE_STRIPPED=$IMAGE_SOURCE
    fi

    # Registration
    echo "Registering..."
    if [ $FLIRT_DIRECTION -eq 0 ]; then
        echo "flirt with -ref using the source image"
        flirt -in $IMAGE_TARGET_STRIPPED -ref $IMAGE_SOURCE_STRIPPED -omat "$OUTPUT_DIR/${TARGET_TO_SOURCE_MAT}" -dof $DOF

        convert_xfm -omat "$OUTPUT_DIR/${SOURCE_TO_TARGET_MAT}" -inverse "$OUTPUT_DIR/${TARGET_TO_SOURCE_MAT}"

        flirt -in $IMAGE_SOURCE_STRIPPED -ref $IMAGE_TARGET_STRIPPED -applyxfm -init "$OUTPUT_DIR/${SOURCE_TO_TARGET_MAT}" -out "$OUTPUT_DIR/${REGISTERED_IMAGE_NAME}"
    else
        echo "flirt with -ref using the target image"
        flirt -in $IMAGE_SOURCE_STRIPPED -ref $IMAGE_TARGET_STRIPPED -omat "$OUTPUT_DIR/${SOURCE_TO_TARGET_MAT}" -out "$OUTPUT_DIR/${REGISTERED_IMAGE_NAME}" -dof $DOF

        convert_xfm -omat "$OUTPUT_DIR/${TARGET_TO_SOURCE_MAT}" -inverse "$OUTPUT_DIR/${SOURCE_TO_TARGET_MAT}"
    fi

    if [ $IMAGE_TARGET_STRIP -eq 0 ]; then
        echo "Removing the stripped target image"
        rm -f "$IMAGE_TARGET_STRIPPED"
    fi
    if [ $IMAGE_SOURCE_STRIP -eq 0 ]; then
        echo "Removing the stripped source image"
        rm -f "$IMAGE_SOURCE_STRIPPED"
    fi
}

process_subject $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}