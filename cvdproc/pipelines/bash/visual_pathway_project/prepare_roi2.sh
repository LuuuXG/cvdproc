#!/usr/bin/env bash

set -e

fs_output_dir=$1 # have /mri, /surf...
subject_id=$2
session_id=$3
fs_to_dwi_xfm=$4
output_dir=$5
dwi_ref=$6
space_entity=$7 # e.g., ACPC

mkdir -p $output_dir

fs_subject_id=$(basename $fs_output_dir)
fs_subjects_dir=$(dirname $fs_output_dir)
# set the SUBJECTS_DIR env variable for freesurfer commands
export SUBJECTS_DIR=$fs_subjects_dir

###################
# LEFT HEMISPHERE #
###################

# 1. V1 cortex
mri_label2vol \
  --subject $fs_subject_id \
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
  --subject $fs_subject_id \
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
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-V1exvivo_mask.nii.gz" \
  -interp nearestneighbour

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -interp nearestneighbour

# 2. LGN
# 8209 for right LGN, 8109 for left LGN
mri_binarize \
  --i "$fs_output_dir/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz" \
  --match 8109 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-LGN_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-LGN_mask.nii.gz" \
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_mask.nii.gz" \
  -interp nearestneighbour

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-${space_entity}_label-LGN_desc-dilate1_mask.nii.gz"

# new: whole thalamus for possible use
mri_binarize \
  --i "$fs_output_dir/mri/aparc+aseg.mgz" \
  --match 10 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-thalamus_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-L_space-fs_label-thalamus_mask.nii.gz" \
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
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
  --subject $fs_subject_id \
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
  --subject $fs_subject_id \
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
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-V1exvivo_mask.nii.gz" \
  -interp nearestneighbour

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-V1exvivo_desc-extend2mm_mask.nii.gz" \
  -interp nearestneighbour

# 2. LGN
# 8209 for right LGN, 8109 for left LGN
mri_binarize \
  --i "$fs_output_dir/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz" \
  --match 8209 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-LGN_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-LGN_mask.nii.gz" \
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_mask.nii.gz" \
  -interp nearestneighbour

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-${space_entity}_label-LGN_desc-dilate1_mask.nii.gz"

# new: whole thalamus for possible use
mri_binarize \
  --i "$fs_output_dir/mri/aparc+aseg.mgz" \
  --match 49 \
  --o "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-thalamus_mask.nii.gz"

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_hemi-R_space-fs_label-thalamus_mask.nii.gz" \
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
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
# sometimes the seg is poor, if the cho_in_fs already exists (we manually modified it), skip this step
if [ -f $cho_in_fs ]; then
    echo "$cho_in_fs exists, skipping optic chiasm extraction from aparc+aseg"
else
    echo "Extracting optic chiasm from aparc+aseg"
    # 85 for optic chiasm
    mri_binarize \
        --i "$fs_output_dir/mri/aparc+aseg.mgz" \
        --match 85 \
        --o "$output_dir/sub-${subject_id}_ses-${session_id}_space-fs_label-opticchiasm_mask.nii.gz"
fi

flirt -in "$output_dir/sub-${subject_id}_ses-${session_id}_space-fs_label-opticchiasm_mask.nii.gz" \
  -ref $dwi_ref \
  -applyxfm -init $fs_to_dwi_xfm \
  -out "$output_dir/sub-${subject_id}_ses-${session_id}_space-${space_entity}_label-opticchiasm_mask.nii.gz" \
  -interp nearestneighbour

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/sub-${subject_id}_ses-${session_id}_space-${space_entity}_label-opticchiasm_mask.nii.gz" \
  dilate 1 \
  "$output_dir/sub-${subject_id}_ses-${session_id}_space-${space_entity}_label-opticchiasm_desc-dilate1_mask.nii.gz"

rm "$output_dir/lh.ribbon.nii.gz"
rm "$output_dir/rh.ribbon.nii.gz"
echo "ROI preparation (for visual pathway tractography) completed."