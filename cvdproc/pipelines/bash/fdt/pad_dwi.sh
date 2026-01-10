#!/bin/bash

# Pad DWI to ensure have odd number in Z dimension

process_subject() {
    DWI_PATH=$1
    DWI_BVEC=$2
    DWI_BVAL=$3
    DWI_JSON=$4
    OUTPUT_DWI=$5

    Nz=$(fslval "$DWI_PATH" dim3)

    if [ $((Nz % 2)) -eq 1 ]; then
        echo "Z dimension ($Nz) is odd. Removing the bottom slice..."
        OUTPUT_DIR=$(dirname "$OUTPUT_DWI")
        mkdir -p "$OUTPUT_DIR"

        fslroi "$DWI_PATH" "$OUTPUT_DWI" 0 -1 0 -1 1 $((Nz-1)) 0 -1

        # replace '.nii.gz' with '.bvec' and '.bval'
        OUTPUT_DWI_BVAL="${OUTPUT_DWI%.nii.gz}.bval"
        OUTPUT_DWI_BVEC="${OUTPUT_DWI%.nii.gz}.bvec"
        OUTPUT_DWI_JSON="${OUTPUT_DWI%.nii.gz}.json"
        cp "$DWI_BVAL" "$OUTPUT_DWI_BVAL"
        cp "$DWI_BVEC" "$OUTPUT_DWI_BVEC"
        cp "$DWI_JSON" "$OUTPUT_DWI_JSON" # Copy the JSON file as well. Note: this may lead to inaccuracies in the JSON file if it contains slice information.
    else
        echo "Z dimension ($Nz) is already even. Do nothing."
    fi
}

# Call the function with the provided paths
process_subject "$1" "$2" "$3" "$4" "$5"