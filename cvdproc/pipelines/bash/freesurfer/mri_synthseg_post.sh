#!/usr/bin/env bash
set -euo pipefail

# Post-processing for FreeSurfer's mri_synthseg
# Create a white-matter mask from specified labels

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <synthseg_input_img> <wm_output_name>"
  exit 1
fi

SYNTHSEG_INPUT_IMG=$1
WM_OUTPUT_NAME=$2

# Basic checks
if [[ ! -f "$SYNTHSEG_INPUT_IMG" ]]; then
  echo "Error: input file not found: $SYNTHSEG_INPUT_IMG" >&2
  exit 1
fi

if ! command -v mri_binarize >/dev/null 2>&1; then
  echo "Error: mri_binarize not found. Make sure FreeSurfer is sourced." >&2
  exit 1
fi

echo "Creating WM mask from labels: 2 41 7 46 77 85 251 252 253 254 255"
mri_binarize \
  --i "$SYNTHSEG_INPUT_IMG" \
  --match 2 41 7 46 77 85 251 252 253 254 255 \
  --o "$WM_OUTPUT_NAME"

echo "WM mask saved to: $WM_OUTPUT_NAME"
