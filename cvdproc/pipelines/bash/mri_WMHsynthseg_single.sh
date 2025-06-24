#!/usr/bin/env bash

[ ! -e "$FREESURFER_HOME" ] && echo "error: freesurfer has not been properly sourced" && exit 1

# check that the model file has been downloaded and installed
# if not, show instructions for downloading and installing
if [[ ! -f $FREESURFER_HOME/"models/WMH-SynthSeg_v10_231110.pth" ]]; then
    echo " "
    echo "   Atlas files not found. Please download atlas from: "
    echo "      https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/Histo_Atlas_Iglesias_2023/atlas.zip "
    echo "   and uncompress it into:  "
    echo "      $FREESURFER_HOME/models/ "
    echo "   You only need to do this once. You can use the following commands (may require root access): "
    echo "      1: cd $FREESURFER_HOME/models/"
    echo "      2a (in Linux): wget https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth "
    echo "      2b (in MAC): curl -o WMH-SynthSeg_v10_231110.pth https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth "
    echo " "
    echo "   After correct download and instillation, the directory: "
    echo "      $FREESURFER_HOME/models/ "
    echo "   should now contain an additional file: WMH-SynthSeg_v10_231110.pth"
    echo " "
    exit 1
fi

# Initialize variables
INPUT_IMAGE=""
OUTPUT_PATH=""
PROB_FILEPATH=""
WMH_FILEPATH=""
SAVE_PROBS=false
ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --i)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        --o)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --prob_filepath)
            PROB_FILEPATH="$2"
            shift 2
            ;;
        --wmh_filepath)
            WMH_FILEPATH="$2"
            shift 2
            ;;
        --save_lesion_probabilities)
            SAVE_PROBS=true
            ARGS+=("$1")
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required inputs
if [[ -z $INPUT_IMAGE || -z $OUTPUT_PATH ]]; then
    echo "Missing required --i or --o"
    exit 1
fi

if [[ -d "$INPUT_IMAGE" || -d "$OUTPUT_PATH" ]]; then
    echo "Error: --i and --o must be file paths, not directories"
    exit 1
fi

# If output already exists, skip inference
if [[ -f "$OUTPUT_PATH" ]]; then
    echo "Output already exists at $OUTPUT_PATH, skipping inference.py"
else
    # Run inference
    INFER_ARGS=(--i "$INPUT_IMAGE" --o "$OUTPUT_PATH" "${ARGS[@]}")
    echo "Running inference.py..."
    $FREESURFER_HOME/bin/fspython "$FREESURFER_HOME/python/packages/WMHSynthSeg/inference.py" "${INFER_ARGS[@]}"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "inference.py failed with code $status"
        exit $status
    fi
fi

# Move lesion probability map if applicable
if [[ "$SAVE_PROBS" == true && -n "$PROB_FILEPATH" ]]; then
    input_dir=$(dirname "$INPUT_IMAGE")
    input_file=$(basename "$INPUT_IMAGE")
    input_stem="${input_file%.*}"
    [[ "$input_file" == *.nii.gz ]] && input_stem="${input_file%.nii.gz}" || input_stem="${input_file%.*}"

    prob_source="${input_dir}/${input_stem}_seg.lesion_probs.nii.gz"
    
    if [[ -f "$prob_source" ]]; then
        mv "$prob_source" "$PROB_FILEPATH"
        echo "Moved lesion probability map to: $PROB_FILEPATH"
    else
        echo "Warning: Lesion probability map not found at expected path: $prob_source"
    fi
fi

# Extract WMH mask where label == 77
if [[ -n "$WMH_FILEPATH" ]]; then
    echo "Extracting WMH mask (label == 77) from $OUTPUT_PATH..."
    python3 - <<EOF
import nibabel as nib
import numpy as np
import sys

try:
    seg_img = nib.load("$OUTPUT_PATH")
    seg_data = seg_img.get_fdata()
    wmh_mask = (seg_data == 77).astype(np.uint8)
    wmh_img = nib.Nifti1Image(wmh_mask, seg_img.affine, seg_img.header)
    nib.save(wmh_img, "$WMH_FILEPATH")
    print("Saved WMH mask to $WMH_FILEPATH")
except Exception as e:
    print("Failed to extract WMH:", str(e))
    sys.exit(1)
EOF
fi

echo "All done for WMHsynthseg and post-processing."