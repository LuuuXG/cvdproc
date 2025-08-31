#!/bin/bash
set -euo pipefail

fs_dir="/mnt/f/BIDS/demo_BIDS/derivatives/freesurfer"
fs4fmriprep_dir="/mnt/f/BIDS/demo_BIDS/derivatives/freesurfer4fmriprep"

mkdir -p "$fs4fmriprep_dir"

shopt -s nullglob
for sub_dir in "$fs_dir"/sub-*; do
  sub=$(basename "$sub_dir")
  for ses_dir in "$sub_dir"/ses-*; do
    ses=$(basename "$ses_dir")
    dst="$fs4fmriprep_dir/$ses/$sub"
    mkdir -p "$dst"

    # Remove a previous directory-level symlink if it exists
    if [ -L "$dst" ]; then rm -f "$dst"; fi

    # Replicate the tree with per-file symlinks
    cp -as "$ses_dir/." "$dst/"
    echo "Linked files from $ses_dir -> $dst/"
  done
done
