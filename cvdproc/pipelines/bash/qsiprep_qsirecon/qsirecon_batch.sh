#!/usr/bin/env bash

set -u

if [ $# -lt 1 ]; then
  echo "Usage: bash qsirecon_batch.sh <bids_dir>"
  exit 1
fi

bids_dir="$1"

single_script="$bids_dir/code/qsirecon_single.sh"
qsiprep_root="$bids_dir/derivatives/qsiprep"
qsirecon_root="$bids_dir/derivatives/qsirecon-DSIStudio"

if [ ! -d "$bids_dir" ]; then
  echo "ERROR: bids_dir does not exist: $bids_dir"
  exit 1
fi

if [ ! -d "$qsiprep_root" ]; then
  echo "ERROR: qsiprep directory does not exist: $qsiprep_root"
  exit 1
fi

if [ ! -f "$single_script" ]; then
  echo "ERROR: single-subject script not found: $single_script"
  exit 1
fi

echo "BIDS directory:   $bids_dir"
echo "QSIPrep root:     $qsiprep_root"
echo "QSIRecon root:    $qsirecon_root"
echo "Single script:    $single_script"
echo

n_total=0
n_success=0
n_failed=0
n_skipped=0

for sub_dir in "$qsiprep_root"/sub-*; do
  [ -d "$sub_dir" ] || continue
  sub=$(basename "$sub_dir")
  subject_id="${sub#sub-}"

  for ses_dir in "$sub_dir"/ses-*; do
    [ -d "$ses_dir" ] || continue
    ses=$(basename "$ses_dir")
    session_id="${ses#ses-}"

    n_total=$((n_total + 1))

    output_file="$qsirecon_root/sub-${subject_id}/ses-${session_id}/dwi/sub-${subject_id}_ses-${session_id}_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_streamlines.trk.gz"

    echo "=================================================="
    echo "Subject: $subject_id | Session: $session_id"
    echo "Expected output:"
    echo "$output_file"
    echo "=================================================="

    if [ -f "$output_file" ]; then
      echo "SKIPPED: output already exists for $subject_id $session_id"
      n_skipped=$((n_skipped + 1))
      echo
      continue
    fi

    bash "$single_script" "$bids_dir" "$subject_id" "$session_id"
    status=$?

    if [ "$status" -eq 0 ]; then
      echo "SUCCESS: $subject_id $session_id"
      n_success=$((n_success + 1))
    else
      echo "FAILED:  $subject_id $session_id"
      n_failed=$((n_failed + 1))
    fi

    echo
  done
done

echo "================ Final Summary ================"
echo "Total subject-session pairs: $n_total"
echo "Successful:                  $n_success"
echo "Skipped:                     $n_skipped"
echo "Failed:                      $n_failed"