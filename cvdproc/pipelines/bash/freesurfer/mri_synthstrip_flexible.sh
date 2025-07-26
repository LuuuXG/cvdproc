#!/usr/bin/env bash

[ ! -e "$FREESURFER_HOME" ] && echo "error: freesurfer has not been properly sourced" && exit 1

USE_4D=0
ARGS=()
INPUT_FILE=""
OUTPUT_FILE=""
MASK_FILE=""
TMP_3D_INPUT=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -4d)
            USE_4D=1
            ;;
        -i)
            INPUT_FILE="$2"
            shift
            ;;
        -o)
            OUTPUT_FILE="$2"
            shift
            ;;
        -m)
            MASK_FILE="$2"
            shift
            ;;
        *)
            ARGS+=("$1")
            ;;
    esac
    shift
done

if [[ $USE_4D -eq 1 ]]; then
    if [[ -z "$INPUT_FILE" || -z "$MASK_FILE" ]]; then
        echo "error: -4d mode requires both -i <input_4d.nii.gz> and -m <mask_output.nii.gz>" >&2
        exit 1
    fi

    # Create temporary 3D image for the first volume
    TMP_3D_INPUT=$(mktemp /tmp/synthstrip_input_XXXXXX.nii.gz)
    echo "Extracting first volume from 4D image: $INPUT_FILE"
    fslroi "$INPUT_FILE" "$TMP_3D_INPUT" 0 1

    echo "Running mri_synthstrip on extracted 3D image..."
    "$FREESURFER_HOME/bin/fspython" "$FREESURFER_HOME/python/scripts/mri_synthstrip" \
        -i "$TMP_3D_INPUT" -m "$MASK_FILE" "${ARGS[@]}"

    if [[ -n "$OUTPUT_FILE" ]]; then
        echo "Applying brain mask to 4D input and saving to $OUTPUT_FILE"
        fslmaths "$INPUT_FILE" -mas "$MASK_FILE" "$OUTPUT_FILE"
    fi

    rm -f "$TMP_3D_INPUT"

else
    # Non -4d mode: pass through all arguments as-is
    exec "$FREESURFER_HOME/bin/fspython" "$FREESURFER_HOME/python/scripts/mri_synthstrip" \
        -i "$INPUT_FILE" ${OUTPUT_FILE:+-o "$OUTPUT_FILE"} ${MASK_FILE:+-m "$MASK_FILE"} "${ARGS[@]}"
fi
