#!/usr/bin/env bash

set -e

bids_dir=$1
subject_id=$2
session_id=$3

# Search for already processed qsiprep output
qsiprep_dir=$bids_dir/derivatives/qsiprep/sub-${subject_id}/ses-${session_id}/dwi
# Find whether have a *space-ACPC_desc-preproc_dwi.nii.gz file
preproc_dwi_file=$(find $qsiprep_dir -type f -name "*space-ACPC_desc-preproc_dwi.nii.gz" | head -n 1)
if [ -z "$preproc_dwi_file" ]; then
  echo "No preprocessed DWI file found for subject ${subject_id}, session ${session_id} in qsiprep output."
  
  # so we need to run qsiprep
  docker run -ti --rm \
            --gpus all \
            -v $bids_dir:/data/input \
            -v $bids_dir/derivatives/qsiprep:/data/output \
            -v $bids_dir/derivatives/workflows/sub-${subject_id}/ses-${session_id}:/work \
            -v $bids_dir/code/license.txt:/opt/freesurfer/license.txt \
            -v $bids_dir/code/qsiprep_filter.json:/opt/qsiprep_filter.json \
            -v $bids_dir/code/DRBUDDI_cuda.sh:/opt/DRBUDDI_cuda.sh \
            qsiprep:1.0.1-custom \
            /data/input /data/output participant \
            --skip-bids-validation \
            --participant-label $subject_id \
            --session-id $session_id \
            --nprocs 6 \
            --omp-nthreads 1 \
            --anat-modality T1w \
            --ignore t2w flair \
            --subject-anatomical-reference sessionwise \
            --skip-anat-based-spatial-normalization \
            --output-resolution 2 \
            --hmc-model 3dSHORE \
            --fs-license-file /opt/freesurfer/license.txt \
            --shoreline-iters 1 \
            --pepolar-method DRBUDDI \
            --work-dir /work \
            --bids-filter-file /opt/qsiprep_filter.json \
            --resource-monitor
else
  echo "Preprocessed DWI file already exists for subject ${subject_id}, session ${session_id} in qsiprep output. Skipping qsiprep."
fi

# Post-process
qsiprep_anat_dir=$bids_dir/derivatives/qsiprep/sub-${subject_id}/ses-${session_id}/anat
# Search for preproc T1w (*space-ACPC_desc-preproc_T1w.nii.gz)
preproc_t1w_file=$(find $qsiprep_anat_dir -type f -name "*space-ACPC_desc-preproc_T1w.nii.gz" | head -n 1)
if [ -z "$preproc_t1w_file" ]; then
  echo "No preprocessed T1w file found for subject ${subject_id}, session ${session_id} in qsiprep output. Exiting."
  exit 1
fi

# check if warp file already exists
if [ -f "$qsiprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5" ] && [ -f "$qsiprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5" ]; then
  echo "Warp files already exist for subject ${subject_id}, session ${session_id}. Skipping warp generation."
  exit 0
fi

mri_synthmorph -t $qsiprep_anat_dir/T1w_to_MNI.nii.gz -T $qsiprep_anat_dir/MNI_to_T1w.nii.gz $preproc_t1w_file /mnt/e/Codes/cvdproc/cvdproc/data/standard/MNI152/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz -g
python3 /mnt/e/Codes/cvdproc/cvdproc/utils/python/nifti_warp_to_h5.py \
    --input "$qsiprep_anat_dir/MNI_to_T1w.nii.gz" \
    --output "$qsiprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"

python3 /mnt/e/Codes/cvdproc/cvdproc/utils/python/nifti_warp_to_h5.py \
    --input "$qsiprep_anat_dir/T1w_to_MNI.nii.gz" \
    --output "$qsiprep_anat_dir/sub-${subject_id}_ses-${session_id}_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5"

# delete the intermediate files (nifti warps)
rm "$qsiprep_anat_dir/T1w_to_MNI.nii.gz" "$qsiprep_anat_dir/MNI_to_T1w.nii.gz"