#!/usr/bin/env bash

#set -e

bids_dir=$1
subject_id=$2
session_id=$3

qsirecon_dsistudio_dir=$bids_dir/derivatives/qsirecon-DSIStudio/sub-${subject_id}/ses-${session_id}/dwi
# Search for *space-ACPC_model-gqi_dwimap.fib.gz
fib_file=$(find $qsirecon_dsistudio_dir -type f -name "*space-ACPC_connectivity.mat" | head -n 1)
if [ -z "$fib_file" ]; then
  echo "No .mat file found for subject ${subject_id}, session ${session_id} in qsirecon-DSIStudio output."
  # so we need to run qsirecon

  docker run -ti --rm \
    --gpus all \
    -v $bids_dir/derivatives/qsiprep:/data/qsiprep \
    -v $bids_dir:/data/qsirecon \
    -v $bids_dir/derivatives/workflows/sub-${subject_id}/ses-${session_id}:/work \
    -v $bids_dir/code/license.txt:/opt/freesurfer/license.txt \
    pennlinc/qsirecon:1.0.0 \
    /data/qsiprep /data/qsirecon participant \
    --participant-label $subject_id \
    --session-id $session_id \
    --fs-license-file /opt/freesurfer/license.txt \
    --recon-spec dsi_studio_gqi \
    --nprocs 8 \
    --omp-nthreads 1 \
    --atlases AAL116 \
    --input-type qsiprep \
    --work-dir /work
else
  echo ".mat file already exists for subject ${subject_id}, session ${session_id} in qsirecon-DSIStudio output. Skipping qsirecon."
fi

qsiprep_dwi_dir=$bids_dir/derivatives/qsiprep/sub-${subject_id}/ses-${session_id}/dwi
# Search for preproc DWI (*space-ACPC_desc-preproc_dwi.nii.gz)
preproc_dwi_file=$(find $qsiprep_dwi_dir -type f -name "*space-ACPC_desc-preproc_dwi.nii.gz" | head -n 1)
if [ -z "$preproc_dwi_file" ]; then
  echo "No preprocessed DWI file found for subject ${subject_id}, session ${session_id} in qsiprep output. Exiting."
  exit 1
fi

fname="$(basename "$preproc_dwi_file")"
fname="${fname%.nii.gz}"

recon_foldername="${fname%_desc-preproc_dwi}_desc-preproc_recon_wf"
# replace - to _
recon_foldername=${recon_foldername//-/_}

src_dir=$bids_dir/derivatives/workflows/sub-${subject_id}/ses-${session_id}/qsirecon_1_0_wf/sub-${subject_id}_dsistudio_pipeline/$recon_foldername/dsistudio_gqi/create_src
# Search for *space-ACPC_desc-preproc_dwi.src.gz
src_file=$(find $src_dir -type f -name "*space-ACPC_desc-preproc_dwi.src.gz" | head -n 1)
if [ -z "$src_file" ]; then
  echo "No .src.gz file found for subject ${subject_id}, session ${session_id} in qsirecon output. Exiting."
  exit 1
fi
# copy to qsirecon folder
qsirecon_dsistudio_dir=$bids_dir/derivatives/qsirecon-DSIStudio/sub-${subject_id}/ses-${session_id}/dwi
cp $src_file $qsirecon_dsistudio_dir/

#F:\BIDS\WCH_AF_Project\derivatives\workflows\sub-HC0059\ses-baseline\qsirecon_1_0_wf\sub-HC0059_dsistudio_pipeline\sub_HC0059_ses_baseline_acq_DSIb4000_dir_AP_space_ACPC_desc_preproc_recon_wf\tractography\tracking
tracking_dir=$bids_dir/derivatives/workflows/sub-${subject_id}/ses-${session_id}/qsirecon_1_0_wf/sub-${subject_id}_dsistudio_pipeline/$recon_foldername/tractography/tracking
#sub-HC0059_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.src.gz.odf.gqi.1.25.fib.trk.gz
trk_file=$(find $tracking_dir -type f -name "*space-ACPC_desc-preproc_dwi.src.gz.odf.gqi.1.25.fib.trk.gz" | head -n 1)
if [ -z "$trk_file" ]; then
  echo "No .trk.gz file found for subject ${subject_id}, session ${session_id} in qsirecon output. Exiting."
  exit 1
fi
# copy .trk.gz file to qsirecon folder
cp $trk_file $qsirecon_dsistudio_dir/'sub-'${subject_id}'_ses-'${session_id}'_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_streamlines.trk.gz'