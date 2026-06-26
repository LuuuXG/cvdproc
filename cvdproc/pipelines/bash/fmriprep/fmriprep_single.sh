#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash fmriprep_single.sh <bids_dir> <subject_id> <session_id> [license_file] [filter_file]"
  exit 1
fi

bids_dir="$1"
subject_id="$2"
session_id="$3"
license_file="${4:-/home/lxg/license.txt}"
filter_file="${5:-${bids_dir}/code/fmriprep_filter.json}"

fmriprep_dir="${bids_dir}/derivatives/fmriprep"
work_dir="${bids_dir}/derivatives/workflows/sub-${subject_id}/ses-${session_id}"
freesurfer_dir="${bids_dir}/derivatives/freesurfer/sub-${subject_id}/ses-${session_id}"
fmriprep_func_dir="${fmriprep_dir}/sub-${subject_id}/ses-${session_id}/func"

template_space="MNI152NLin2009cAsym"
template_res="res-2"
template_spec="${template_space}:res-2"

mkdir -p "$fmriprep_dir" "$work_dir"

if [ ! -d "$freesurfer_dir" ]; then
  echo "FreeSurfer directory not found: $freesurfer_dir"
  exit 1
fi

if [ ! -f "$license_file" ]; then
  echo "FreeSurfer license file not found: $license_file"
  exit 1
fi

if [ ! -f "$filter_file" ]; then
  echo "BIDS filter file not found: $filter_file"
  exit 1
fi

mni_bold_file=$(find "$fmriprep_func_dir" -maxdepth 1 -type f -name "*_task-rest*_space-${template_space}_${template_res}_desc-preproc_bold.nii.gz" 2>/dev/null | sort | head -n 1 || true)

if [ -n "$mni_bold_file" ]; then
  echo "MNI preprocessed BOLD already exists: $mni_bold_file"
  echo "Skipping fMRIPrep."
  exit 0
fi

docker run -ti --rm \
  -v "${bids_dir}:/data" \
  -v "${fmriprep_dir}:/out" \
  -v "${work_dir}:/work" \
  -v "${freesurfer_dir}:/precomputed_freesurfer/sub-${subject_id}_ses-${session_id}" \
  -v "${license_file}:/opt/freesurfer/license.txt" \
  -v "${filter_file}:/opt/fmri_filter.json" \
  nipreps/fmriprep:25.2.5 \
  /data /out \
  participant \
  --participant-label "$subject_id" \
  --session-label "$session_id" \
  -w /work \
  --nprocs 18 \
  --subject-anatomical-reference sessionwise \
  --use-syn-sdc \
  --fs-subjects-dir /precomputed_freesurfer \
  --fs-license-file /opt/freesurfer/license.txt \
  --bids-filter-file /opt/fmri_filter.json \
  --output-spaces func "$template_spec" \
  --skip-bids-validation \
  --ignore t2w flair

mni_bold_file=$(find "$fmriprep_func_dir" -maxdepth 1 -type f -name "*_task-rest*_space-${template_space}_${template_res}_desc-preproc_bold.nii.gz" 2>/dev/null | sort | head -n 1 || true)

if [ -n "$mni_bold_file" ]; then
  echo "fMRIPrep completed. MNI BOLD output: $mni_bold_file"
else
  echo "fMRIPrep finished, but no MNI152NLin2009cAsym res-2 preprocessed BOLD file was found."
  exit 1
fi