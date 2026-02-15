#!/usr/bin/env bash
set -euo pipefail

# Absolute path to this bash script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_EXTEND="${SCRIPT_DIR}/../../../utils/python/extend_mask_x.py"
if [[ ! -f "$PY_EXTEND" ]]; then
  echo "[ERROR] extend_mask_x.py not found: $PY_EXTEND"
  exit 1
fi

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <fs_output_dir> <subject_id> <session_id> <fs_to_dwi_xfm> <output_dir> <dwi_ref> <space_entity>"
  exit 2
fi

fs_output_dir="$1"  # have /mri, /surf...
subject_id="$2"
session_id="$3"
fs_to_dwi_xfm="$4"
output_dir="$5"
dwi_ref="$6"
space_entity="$7"   # e.g., ACPC

mkdir -p "$output_dir"

fs_subject_id="$(basename "$fs_output_dir")"
fs_subjects_dir="$(dirname "$fs_output_dir")"
export SUBJECTS_DIR="$fs_subjects_dir"

#############################
# Skip if final outputs exist
#############################
key_outputs=(
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-V1exvivo_mask.nii.gz"
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-V1exvivo_mask.nii.gz"
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_desc-dilate3x_mask.nii.gz"
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_desc-dilate3x_mask.nii.gz"
  "$output_dir/sub-${subject_id}_ses-${session_id}_space-${space_entity}_label-opticchiasm_mask.nii.gz"
)

all_exist=true
for f in "${key_outputs[@]}"; do
  if [[ ! -s "$f" ]]; then
    all_exist=false
    break
  fi
done

if [[ "$all_exist" == true ]]; then
  echo "[INFO] All key ROI outputs already exist. Skip ROI preparation."
  exit 0
fi

###################
# LEFT HEMISPHERE #
###################

# 1. V1 cortex
mri_label2vol \
  --subject "$fs_subject_id" \
  --hemi lh \
  --label "$fs_output_dir/label/lh.V1_exvivo.label" \
  --temp "$fs_output_dir/mri/orig.mgz" \
  --regheader "$fs_output_dir/mri/orig.mgz" \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_mask.nii.gz" \
  --proj frac 0 1 0.1

# only keep voxels in lh.ribbon.mgz
mri_convert "$fs_output_dir/mri/lh.ribbon.mgz" "$output_dir/lh.ribbon.nii.gz"
fslmaths "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_mask.nii.gz" \
  -mas "$output_dir/lh.ribbon.nii.gz" \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_mask.nii.gz"

mri_label2vol \
  --subject "$fs_subject_id" \
  --hemi lh \
  --label "$fs_output_dir/label/lh.V1_exvivo.label" \
  --temp "$fs_output_dir/mri/orig.mgz" \
  --regheader "$fs_output_dir/mri/orig.mgz" \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  --proj abs -2 0 0.1

# add V1 cortex
fslmaths "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -add "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_mask.nii.gz" \
  -bin "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz"

# transform to DWI space
flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-V1exvivo_mask.nii.gz" \
  -interp nearestneighbour

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -interp nearestneighbour

# 2. LGN (8109 for left)
mri_binarize \
  --i "$fs_output_dir/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz" \
  --match 8109 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-LGN_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-LGN_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_mask.nii.gz" \
  -interp nearestneighbour

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_desc-dilate1_mask.nii.gz"

# dilate 3 voxel in x axis
python "$PY_EXTEND" \
  --in_mask "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_mask.nii.gz" \
  --out_mask "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_desc-dilate3x_mask.nii.gz" \
  --extend_part "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_desc-extendpart_mask.nii.gz" \
  --n_vox 3 \
  --direction plus

# whole thalamus for possible use (10 for left)
mri_binarize \
  --i "$fs_output_dir/mri/aparc+aseg.mgz" \
  --match 10 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-thalamus_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-thalamus_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-thalamus_mask.nii.gz" \
  -interp nearestneighbour

mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-thalamus_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-thalamus_desc-dilate1_mask.nii.gz"

####################
# RIGHT HEMISPHERE #
####################

# 1. V1 cortex
mri_label2vol \
  --subject "$fs_subject_id" \
  --hemi rh \
  --label "$fs_output_dir/label/rh.V1_exvivo.label" \
  --temp "$fs_output_dir/mri/orig.mgz" \
  --regheader "$fs_output_dir/mri/orig.mgz" \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_mask.nii.gz" \
  --proj frac 0 1 0.1

mri_convert "$fs_output_dir/mri/rh.ribbon.mgz" "$output_dir/rh.ribbon.nii.gz"
fslmaths "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_mask.nii.gz" \
  -mas "$output_dir/rh.ribbon.nii.gz" \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_mask.nii.gz"

mri_label2vol \
  --subject "$fs_subject_id" \
  --hemi rh \
  --label "$fs_output_dir/label/rh.V1_exvivo.label" \
  --temp "$fs_output_dir/mri/orig.mgz" \
  --regheader "$fs_output_dir/mri/orig.mgz" \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  --proj abs -2 0 0.1

fslmaths "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -add "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_mask.nii.gz" \
  -bin "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz"

# transform to DWI space
flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-V1exvivo_mask.nii.gz" \
  -interp nearestneighbour

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -interp nearestneighbour

# 2. LGN (8209 for right)
mri_binarize \
  --i "$fs_output_dir/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz" \
  --match 8209 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-LGN_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-LGN_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_mask.nii.gz" \
  -interp nearestneighbour

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_desc-dilate1_mask.nii.gz"

# dilate 3 voxel in x axis
python "$PY_EXTEND" \
  --in_mask "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_mask.nii.gz" \
  --out_mask "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_desc-dilate3x_mask.nii.gz" \
  --extend_part "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_desc-extendpart_mask.nii.gz" \
  --n_vox 3 \
  --direction minus

# whole thalamus for possible use (49 for right)
mri_binarize \
  --i "$fs_output_dir/mri/aparc+aseg.mgz" \
  --match 49 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-thalamus_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-thalamus_mask.nii.gz" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-thalamus_mask.nii.gz" \
  -interp nearestneighbour

mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-thalamus_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-thalamus_desc-dilate1_mask.nii.gz"

################
# Optic chiasm #
################

cho_in_fs="$output_dir/sub-${subject_id}_ses-${session_id}_space-fs_label-opticchiasm_mask.nii.gz"
if [[ -f "$cho_in_fs" ]]; then
  echo "[INFO] $cho_in_fs exists, skipping optic chiasm extraction from aparc+aseg"
else
  echo "[INFO] Extracting optic chiasm from aparc+aseg"
  mri_binarize \
    --i "$fs_output_dir/mri/aparc+aseg.mgz" \
    --match 85 \
    --o "$cho_in_fs"
fi

flirt -in "$cho_in_fs" \
  -ref "$dwi_ref" \
  -applyxfm -init "$fs_to_dwi_xfm" \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_space-${space_entity}_label-opticchiasm_mask.nii.gz" \
  -interp nearestneighbour

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_space-${space_entity}_label-opticchiasm_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_space-${space_entity}_label-opticchiasm_desc-dilate1_mask.nii.gz"

rm -f "$output_dir/lh.ribbon.nii.gz" "$output_dir/rh.ribbon.nii.gz"
echo "[INFO] ROI preparation (for visual pathway tractography) completed."
