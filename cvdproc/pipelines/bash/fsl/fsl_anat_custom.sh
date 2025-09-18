#!/usr/bin/env bash

# ========= fsl_anat_custom.sh =========

# Exit if FSL is not properly sourced
[ -z "$FSLDIR" ] && echo "error: FSL not sourced. Run 'source $FSLDIR/etc/fslconf/fsl.sh'" && exit 1

# Parse args
OUTPUT_PREFIX=""
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--o)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Check for required -o argument
if [[ -z "$OUTPUT_PREFIX" ]]; then
    echo "error: Missing required -o argument (output prefix)"
    exit 1
fi

ANAT_DIR="${OUTPUT_PREFIX}.anat"
COMPLETION_MARKER="${ANAT_DIR}/T1_to_MNI_nonlin.nii.gz"

# Check if output already exists
if [[ -f "$COMPLETION_MARKER" ]]; then
    echo "Detected existing output at $COMPLETION_MARKER. Skipping fsl_anat."
    exit 0
fi

# Construct command
CMD=(fsl_anat "${ARGS[@]}" -o "$OUTPUT_PREFIX")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
status=$?

if [[ $status -eq 0 ]]; then
    echo "fsl_anat completed successfully."
else
    echo "fsl_anat failed with code $status."
fi

exit $status
