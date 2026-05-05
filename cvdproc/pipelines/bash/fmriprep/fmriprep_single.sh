#!/usr/bin/env bash

set -e

bids_dir=$1
subject_id=$2
session_id=$3

# Search for already processed fmriprep output
qsiprep_dir=$bids_dir/derivatives/fmriprep/sub-${subject_id}/ses-${session_id}/func
# Find whether have a *space-ACPC_desc-preproc_dwi.nii.gz file
preproc_bold_in_t1w_file=$(find $qsiprep_dir -type f -name "*space-T1w_desc-preproc_bold.nii.gz" | head -n 1)
if [ -z "$preproc_bold_in_t1w_file" ]; then
  echo "No preprocessed BOLD in T1w file found for subject ${subject_id}, session ${session_id} in fmriprep output."

  # so we need to run fmriprep
  docker run -ti --rm \
            --gpus all \
            -v $bids_dir:/data/input \
            -v $bids_dir/derivatives/qsiprep:/data/output \
            -v $bids_dir/derivatives/workflows/sub-${subject_id}/ses-${session_id}:/work \
            -v $bids_dir/derivatives/freesurfer/sub-${subject_id}/ses-${session_id}:/precomputed_freesurfer/sub-${subject_id}_ses-${session_id} \
            -v $bids_dir/code/license.txt:/opt/freesurfer/license.txt \
            -v $bids_dir/code/fmriprep_filter.json:/opt/fmriprep_filter.json \
            fmriprep:25.2.5 \
            /data/input /data/output participant \
            --skip-bids-validation \
            --participant-label $subject_id \
            --session-label $session_id \
            --nprocs 12 \
            --subject-anatomical-reference sessionwise \
            --skip-anat-based-spatial-normalization \
            --fs-license-file /opt/freesurfer/license.txt \
            --work-dir /work \
            --bids-filter-file /opt/fmriprep_filter.json \
            --fs-subjects-dir /precomputed_freesurfer \
            --force syn-sdc \
            --bold2anat-init t1w \
            --output-spaces T1w
else
  echo "Preprocessed BOLD in T1w file already exists for subject ${subject_id}, session ${session_id} in fmriprep output. Skipping fmriprep."
fi

# Post-process
fmriprep_anat_dir=$bids_dir/derivatives/fmriprep/sub-${subject_id}/ses-${session_id}/anat
# Search for preproc T1w (*space-ACPC_desc-preproc_T1w.nii.gz)
preproc_t1w_file=$(find $fmriprep_anat_dir -type f -name "*desc-preproc_T1w.nii.gz" | head -n 1)
if [ -z "$preproc_t1w_file" ]; then
  echo "No preprocessed T1w file found for subject ${subject_id}, session ${session_id} in fmriprep output. Exiting."
  exit 1
fi

# check if warp file already exists
if [ -f "$fmriprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5" ] && [ -f "$fmriprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5" ]; then
  echo "Warp files already exist for subject ${subject_id}, session ${session_id}. Skipping warp generation."
  exit 0
fi

mri_synthmorph -t $fmriprep_anat_dir/T1w_to_MNI.nii.gz -T $fmriprep_anat_dir/MNI_to_T1w.nii.gz $preproc_t1w_file /mnt/e/Codes/cvdproc/cvdproc/data/standard/MNI152/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz -g
python3 /mnt/e/Codes/cvdproc/cvdproc/utils/python/nifti_warp_to_h5.py \
    --input "$fmriprep_anat_dir/MNI_to_T1w.nii.gz" \
    --output "$fmriprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"

python3 /mnt/e/Codes/cvdproc/cvdproc/utils/python/nifti_warp_to_h5.py \
    --input "$fmriprep_anat_dir/T1w_to_MNI.nii.gz" \
    --output "$fmriprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5"

# delete the intermediate files (nifti warps)
rm "$fmriprep_anat_dir/T1w_to_MNI.nii.gz" "$fmriprep_anat_dir/MNI_to_T1w.nii.gz"