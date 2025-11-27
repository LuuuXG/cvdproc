#!/usr/bin/env bash

set -e

fs_output_dir=$1 # have /mri, /surf...
fs_to_orig_mat=$2
t1w_ref=$3
dwi_ref=$4
output_dir=$5
dwi_final_ref=$6

mkdir -p $output_dir

fs_subject_id=$(basename $fs_output_dir)
fs_subjects_dir=$(dirname $fs_output_dir)
# set the SUBJECTS_DIR env variable for freesurfer commands
export SUBJECTS_DIR=$fs_subjects_dir

output_check_file="$output_dir/optic_chiasm_in_DWI_dil1.nii.gz"
if [ -f $output_check_file ]; then
    echo "$output_check_file exists, skipping ROI preparation"
    exit 0
fi

# get orig to dwi warp

warp_check_file="$output_dir/orig_to_dwi_warp.nii.gz"
if [ -f $warp_check_file ]; then
    echo "$warp_check_file exists, skipping orig to dwi warp calculation"
else
    echo "Calculating orig to dwi warp"
    mri_synthmorph \
      -o "$output_dir/orig_to_dwi_t1w.nii.gz" \
      -t "$output_dir/orig_to_dwi_warp.nii.gz" \
      $t1w_ref \
      $dwi_ref \
      -g
fi

orig_to_dwi_warp="$output_dir/orig_to_dwi_warp.nii.gz"

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
  --o "$output_dir/lh_V1_exvivo_in_fs.nii.gz" \
  --proj abs -2 1 0.1 # extend 2mm into white matter

# transform to DWI space (fs -> orig -> dwi), and resample to desired resolution
# fs to orig
flirt -in "$output_dir/lh_V1_exvivo_in_fs.nii.gz" \
  -ref $t1w_ref \
  -applyxfm -init $fs_to_orig_mat \
  -out "$output_dir/lh_V1_exvivo_in_T1w.nii.gz" \
  -interp nearestneighbour

# orig to dwi
mri_convert -at $orig_to_dwi_warp \
  "$output_dir/lh_V1_exvivo_in_T1w.nii.gz" \
  "$output_dir/lh_V1_exvivo_in_DWI.nii.gz" \
  -rt nearest

# 2. LGN

# 8209 for right LGN, 8109 for left LGN
mri_binarize \
  --i "$fs_output_dir/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz" \
  --match 8109 \
  --o "$output_dir/lh_LGN_in_fs.nii.gz"

flirt -in "$output_dir/lh_LGN_in_fs.nii.gz" \
  -ref $t1w_ref \
  -applyxfm -init $fs_to_orig_mat \
  -out "$output_dir/lh_LGN_in_T1w.nii.gz" \
  -interp nearestneighbour

mri_convert -at $orig_to_dwi_warp \
  "$output_dir/lh_LGN_in_T1w.nii.gz" \
  "$output_dir/lh_LGN_in_DWI.nii.gz" \
  -rt nearest

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/lh_LGN_in_DWI.nii.gz" \
  dilate 1 \
  "$output_dir/lh_LGN_in_DWI_dil1.nii.gz"

# new: whole thalamus for possible use
mri_binarize \
  --i "$fs_output_dir/mri/aparc+aseg.mgz" \
  --match 10 \
  --o "$output_dir/lh_thalamus_in_fs.nii.gz"

flirt -in "$output_dir/lh_thalamus_in_fs.nii.gz" \
  -ref $t1w_ref \
  -applyxfm -init $fs_to_orig_mat \
  -out "$output_dir/lh_thalamus_in_T1w.nii.gz" \
  -interp nearestneighbour

mri_convert -at $orig_to_dwi_warp \
  "$output_dir/lh_thalamus_in_T1w.nii.gz" \
  "$output_dir/lh_thalamus_in_DWI.nii.gz" \
  -rt nearest

mri_morphology \
  "$output_dir/lh_thalamus_in_DWI.nii.gz" \
  dilate 1 \
  "$output_dir/lh_thalamus_in_DWI_dil1.nii.gz"

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
  --o "$output_dir/rh_V1_exvivo_in_fs.nii.gz" \
  --proj abs -2 1 0.1 # extend 2mm into white matter

# transform to DWI space (fs -> orig -> dwi)
# fs to orig
flirt -in "$output_dir/rh_V1_exvivo_in_fs.nii.gz" \
  -ref $t1w_ref \
  -applyxfm -init $fs_to_orig_mat \
  -out "$output_dir/rh_V1_exvivo_in_T1w.nii.gz" \
  -interp nearestneighbour

# orig to dwi
mri_convert -at $orig_to_dwi_warp \
  "$output_dir/rh_V1_exvivo_in_T1w.nii.gz" \
  "$output_dir/rh_V1_exvivo_in_DWI.nii.gz" \
  -rt nearest

# 2. LGN
# 8209 for right LGN, 8109 for left LGN
mri_binarize \
  --i "$fs_output_dir/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz" \
  --match 8209 \
  --o "$output_dir/rh_LGN_in_fs.nii.gz"

flirt -in "$output_dir/rh_LGN_in_fs.nii.gz" \
  -ref $t1w_ref \
  -applyxfm -init $fs_to_orig_mat \
  -out "$output_dir/rh_LGN_in_T1w.nii.gz" \
  -interp nearestneighbour

mri_convert -at $orig_to_dwi_warp \
  "$output_dir/rh_LGN_in_T1w.nii.gz" \
  "$output_dir/rh_LGN_in_DWI.nii.gz" \
  -rt nearest

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/rh_LGN_in_DWI.nii.gz" \
  dilate 1 \
  "$output_dir/rh_LGN_in_DWI_dil1.nii.gz"

# new: whole thalamus for possible use
mri_binarize \
  --i "$fs_output_dir/mri/aparc+aseg.mgz" \
  --match 49 \
  --o "$output_dir/rh_thalamus_in_fs.nii.gz"

flirt -in "$output_dir/rh_thalamus_in_fs.nii.gz" \
  -ref $t1w_ref \
  -applyxfm -init $fs_to_orig_mat \
  -out "$output_dir/rh_thalamus_in_T1w.nii.gz" \
  -interp nearestneighbour 

mri_convert -at $orig_to_dwi_warp \
  "$output_dir/rh_thalamus_in_T1w.nii.gz" \
  "$output_dir/rh_thalamus_in_DWI.nii.gz" \
  -rt nearest

mri_morphology \
  "$output_dir/rh_thalamus_in_DWI.nii.gz" \
  dilate 1 \
  "$output_dir/rh_thalamus_in_DWI_dil1.nii.gz"

################
# Optic chiasm #
################

cho_in_fs="$output_dir/optic_chiasm_in_fs.nii.gz"
# sometimes the seg is poor, if the cho_in_fs already exists (we manually modified it), skip this step
if [ -f $cho_in_fs ]; then
    echo "$cho_in_fs exists, skipping optic chiasm extraction from aparc+aseg"
else
    echo "Extracting optic chiasm from aparc+aseg"
    # 85 for optic chiasm
  mri_binarize \
    --i "$fs_output_dir/mri/aparc+aseg.mgz" \
    --match 85 \
    --o "$output_dir/optic_chiasm_in_fs.nii.gz"
fi

flirt -in "$output_dir/optic_chiasm_in_fs.nii.gz" \
  -ref $t1w_ref \
  -applyxfm -init $fs_to_orig_mat \
  -out "$output_dir/optic_chiasm_in_T1w.nii.gz" \
  -interp nearestneighbour

mri_convert -at $orig_to_dwi_warp \
  "$output_dir/optic_chiasm_in_T1w.nii.gz" \
  "$output_dir/optic_chiasm_in_DWI.nii.gz" \
  -rt nearest

# also dilate 1 voxel for possible use
mri_morphology \
  "$output_dir/optic_chiasm_in_DWI.nii.gz" \
  dilate 1 \
  "$output_dir/optic_chiasm_in_DWI_dil1.nii.gz"

##################
# Final resample #
##################
# if set dwi_final_ref, resample to that resolution
if [ ! -z "$dwi_final_ref" ]; then
    for roi in lh_LGN_in_DWI.nii.gz lh_LGN_in_DWI_dil1.nii.gz lh_V1_exvivo_in_DWI.nii.gz optic_chiasm_in_DWI.nii.gz optic_chiasm_in_DWI_dil1.nii.gz rh_LGN_in_DWI.nii.gz rh_LGN_in_DWI_dil1.nii.gz rh_V1_exvivo_in_DWI.nii.gz lh_thalamus_in_DWI.nii.gz lh_thalamus_in_DWI_dil1.nii.gz rh_thalamus_in_DWI.nii.gz rh_thalamus_in_DWI_dil1.nii.gz; do
        antsApplyTransforms -d 3 \
          -i "$output_dir/${roi}" \
          -r $dwi_final_ref \
          -o "$output_dir/${roi}" \
          -n NearestNeighbor
    done
    fi