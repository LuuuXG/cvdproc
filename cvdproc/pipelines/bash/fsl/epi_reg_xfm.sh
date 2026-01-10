#!/usr/bin/env bash
set -e

# === Argument parsing ===
EPI=$1
T1=$2
T1BRAIN=$3
OUT=$4
WMSEG=$5
EPI2T1MAT=$6
T12EPIMAT=$7

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 <epi.nii.gz> <T1.nii.gz> <T1_brain.nii.gz> <output_epi_reg.nii.gz> <wmseg.nii.gz> <epi2t1.mat> <t12epi.mat>"
  exit 1
fi

# === Run epi_reg ===
epi_reg --epi="$EPI" \
        --t1="$T1" \
        --t1brain="$T1BRAIN" \
        --out="$OUT" \
        --wmseg="$WMSEG" \
        --noclean

# OLD: replace OUT .nii.gz -> _init.mat
old_epi2t1="${OUT%.nii.gz}_init.mat"
old_t12epi="${OUT%.nii.gz}.mat"

# Rename
mv "$old_epi2t1" "$EPI2T1MAT"
mv "$old_t12epi" "$T12EPIMAT"

# Delete
rm -f "${OUT%.nii.gz}_fast_wmseg.nii.gz"
rm -f "${OUT%.nii.gz}_fast_wmedge.nii.gz"