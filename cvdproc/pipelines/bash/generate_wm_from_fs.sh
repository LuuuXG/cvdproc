#!/bin/bash

# Use recon-all output to generate WM mask in DWI space
# generate_wm_from_fs.sh --fs_output <FREESURFER_OUTPUT_DIR> --fs_to_dwi <FS_TO_DWI_TRANSFORM> --output_dir <OUTPUT_DIR> --dwi <DWI_IMAGE> --exclude [MASK1 MASK2 ...]

usage() {
    echo "Usage: $0 --fs_output <FREESURFER_OUTPUT_DIR> --fs_to_dwi <FS_TO_DWI_TRANSFORM> --output_dir <OUTPUT_DIR> --dwi <DWI_IMAGE> --exclude [MASK1 MASK2 ...]"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fs_output)
            FS_OUTPUT="$2"; shift 2;;
        --fs_to_dwi)
            FS_TO_DWI="$2"; shift 2;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2;;
        --dwi)
            DWI_IMAGE="$2"; shift 2;;
        --exclude)
            EXCLUDE_MASKS=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXCLUDE_MASKS+=("$1")
                shift
            done;;
        *)
            echo "Unknown argument: $1"
            usage;;
    esac
done

# Validate required arguments
if [[ -z "$FS_OUTPUT" || -z "$FS_TO_DWI" || -z "$OUTPUT_DIR" || -z "$DWI_IMAGE" ]]; then
    usage
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Step 1: Convert aseg to NIfTI
mri_convert $FS_OUTPUT/mri/aseg.mgz $OUTPUT_DIR/aseg.nii.gz
mri_convert $FS_OUTPUT/mri/orig.mgz $OUTPUT_DIR/orig.nii.gz

flirt -in $OUTPUT_DIR/aseg.nii.gz \
    -ref $OUTPUT_DIR/orig.nii.gz \
    -out $OUTPUT_DIR/aseg.nii.gz \
    -applyxfm -usesqform -interp nearestneighbour

rm -f $OUTPUT_DIR/orig.nii.gz

# Step 2: Initialize empty mask
fslmaths $OUTPUT_DIR/aseg.nii.gz -mul 0 $OUTPUT_DIR/temp_wm.nii.gz

# L R
# 2 41: Cerebral-White-Matter
# 7 46: Cerebellum-White-Matter
# 77: WM-hypointensities
# 85: Optic-Chiasm
# 251： CC_Posterior
# 252： CC_Mid_Posterior
# 253： CC_Central
# 254： CC_Mid_Anterior
# 255： CC_Anterior
# 1004 2004: corpuscallosum (actually not in aseg.mgz or aparc+aseg.mgz)

# Step 3: Loop through label IDs
#for i in 2 41 7 46 77 85 251 252 253 254 255 1004 2004; do

mri_binarize \
  --i "$OUTPUT_DIR/aseg.nii.gz" \
  --match 2 41 7 46 77 85 251 252 253 254 255 \
  --o "$OUTPUT_DIR/temp_wm.nii.gz"

# Step 4: Optional binarization (in case of overlaps)
fslmaths $OUTPUT_DIR/temp_wm.nii.gz -bin $OUTPUT_DIR/WM_in_fs.nii.gz

# Step 5: Transform to DWI space
flirt -in $OUTPUT_DIR/WM_in_fs.nii.gz -ref $DWI_IMAGE -applyxfm -init $FS_TO_DWI -out $OUTPUT_DIR/WM_in_dwi.nii.gz -interp nearestneighbour

# Step 6: Apply exclude masks (remove value=1 in each mask)
cp "$OUTPUT_DIR/WM_in_dwi.nii.gz" "$OUTPUT_DIR/WM_final.nii.gz"

for excl in "${EXCLUDE_MASKS[@]}"; do
    if [[ -f "$excl" ]]; then
        echo "Excluding mask: $excl"
        fslmaths "$excl" -bin -sub 1 -mul -1 "$OUTPUT_DIR/tmp_inv_excl.nii.gz"
        fslmaths "$OUTPUT_DIR/WM_final.nii.gz" -mas "$OUTPUT_DIR/tmp_inv_excl.nii.gz" "$OUTPUT_DIR/WM_final.nii.gz"
    else
        echo "Mask not found, skipping: $excl"
    fi
done

# Cleanup
rm -f "$OUTPUT_DIR/tmp_bin.nii.gz" "$OUTPUT_DIR/tmp_inv_excl.nii.gz" "$OUTPUT_DIR/temp_wm.nii.gz"