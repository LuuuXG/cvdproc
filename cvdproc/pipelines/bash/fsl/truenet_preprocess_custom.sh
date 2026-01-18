#!/usr/bin/env bash
set -euo pipefail

###################### IMPORTANT NOTE ########################
# T1w and FLAIR should be aligned and skull-stripped already #
##############################################################

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <FLAIR_IMG> <T1w_IMG> <BRAIN_MASK> <SynthSeg_IMG> <OUTPUT_DIR> <PREFIX>" >&2
  exit 1
fi

FLAIR_IMG="$1"
T1w_IMG="$2"
BRAIN_MASK="$3"
SynthSeg_IMG="$4"
OUTPUT_DIR="$5"
PREFIX="$6"

# -------------------------
# Basic checks
# -------------------------
for f in "$FLAIR_IMG" "$T1w_IMG" "$BRAIN_MASK" "$SynthSeg_IMG"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: File not found: $f" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
TMPVISDIR="$(mktemp -d -p "$OUTPUT_DIR" "${PREFIX}.tmp.XXXXXX")"
echo "Using temp dir: ${TMPVISDIR}"

cleanup() {
  rm -rf "$TMPVISDIR"
}
trap cleanup EXIT

# -------------------------
# Reorient to std (work in tmp)
# -------------------------
fslreorient2std "$FLAIR_IMG" "${TMPVISDIR}/FLAIR.nii.gz"
fslreorient2std "$T1w_IMG"   "${TMPVISDIR}/T1.nii.gz"
fslreorient2std "$BRAIN_MASK" "${TMPVISDIR}/brainmask.nii.gz"
fslreorient2std "$SynthSeg_IMG" "${TMPVISDIR}/SynthSeg.nii.gz"

# -------------------------
# Bias correction on FLAIR (FAST)
# -------------------------
#fast -B --nopve "${TMPVISDIR}/FLAIR.nii.gz"

#imcp "${TMPVISDIR}/FLAIR_restore.nii.gz" "${OUTPUT_DIR}/${PREFIX}_FLAIR.nii.gz"
imcp "${TMPVISDIR}/FLAIR.nii.gz" "${OUTPUT_DIR}/${PREFIX}_FLAIR.nii.gz"
imcp "${TMPVISDIR}/T1.nii.gz"            "${OUTPUT_DIR}/${PREFIX}_T1.nii.gz"
imcp "${TMPVISDIR}/brainmask.nii.gz"     "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz"

# -------------------------
# WM mask from SynthSeg labels (use mri_binarize)
# Labels: 2 41 7 46 85 11 50
# -------------------------
if command -v mri_binarize >/dev/null 2>&1; then
  mri_binarize \
    --i "${TMPVISDIR}/SynthSeg.nii.gz" \
    --match 2 41 7 46 85 11 50 \
    --o "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz"
else
  echo "ERROR: mri_binarize not found. Please load FreeSurfer or implement a fallback." >&2
  exit 3
fi

# -------------------------
# Vent mask (labels 4 and 43 are lateral ventricles)
# -------------------------
# Note: keep consistent with reoriented SynthSeg
mri_binarize \
  --i "${TMPVISDIR}/SynthSeg.nii.gz" \
  --match 4 43 \
  --o "${TMPVISDIR}/ventmask.nii.gz"

# -------------------------
# Distance maps
# -------------------------
distancemap -i "${TMPVISDIR}/ventmask.nii.gz" -o "${TMPVISDIR}/ventdistmap_full.nii.gz"
fslmaths "${TMPVISDIR}/ventdistmap_full.nii.gz" -mas "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz" "${TMPVISDIR}/ventdistmap.nii.gz"

# Extend ventricles (<= 6mm), fill holes, create GMmask
fslmaths "${TMPVISDIR}/ventdistmap.nii.gz" -thr -1 -uthr 6 -bin -fillh26 "${TMPVISDIR}/extended_ventricles.nii.gz"
fslmaths "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz" -add "${TMPVISDIR}/extended_ventricles.nii.gz" -thr 0 -bin "${TMPVISDIR}/nonGMmask.nii.gz"
fslmaths "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz" -sub "${TMPVISDIR}/nonGMmask.nii.gz" -thr 0 -bin "${TMPVISDIR}/GMmask.nii.gz"

distancemap -i "${TMPVISDIR}/GMmask.nii.gz" -o "${TMPVISDIR}/GMdistmap_full.nii.gz"
fslmaths "${TMPVISDIR}/GMdistmap_full.nii.gz" -mas "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz" "${TMPVISDIR}/GMdistmap.nii.gz"

# Mask the distance maps to WM
fslmaths "${TMPVISDIR}/ventdistmap.nii.gz" -mas "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz" "${OUTPUT_DIR}/${PREFIX}_ventdistmap.nii.gz"
fslmaths "${TMPVISDIR}/GMdistmap.nii.gz"   -mas "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz" "${OUTPUT_DIR}/${PREFIX}_GMdistmap.nii.gz"


# #!/bin/bash

# set -e
# ###################### IMPORTANT NOTE ########################
# # T1w and FLAIR should be aligned and skull-stripped already #
# ##############################################################

# FLAIR_IMG=$1
# T1w_IMG=$2 
# BRAIN_MASK=$3
# SynthSeg_IMG=$4
# OUTPUT_DIR=$5
# PREFIX=$6

# # make a temp directory in output dir
# mkdir -p ${OUTPUT_DIR}
# TMPVISDIR=$(mktemp -d -p "${OUTPUT_DIR}")
# echo "Using temp dir: ${TMPVISDIR}"

# fslreorient2std "${FLAIR_IMG}" "${TMPVISDIR}/FLAIR.nii.gz"
# fslreorient2std "${T1w_IMG}" "${TMPVISDIR}/T1.nii.gz"

# fast -B --nopve ${TMPVISDIR}/FLAIR.nii.gz
# imcp ${TMPVISDIR}/FLAIR_restore.nii.gz ${OUTPUT_DIR}/${PREFIX}_FLAIR.nii.gz
# imcp ${TMPVISDIR}/T1.nii.gz ${OUTPUT_DIR}/${PREFIX}_T1.nii.gz
# imcp $BRAIN_MASK ${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz

# # get WM mask
# fslmaths $FLAIR_IMG -mul 0 $TMPVISDIR/temp_wm.nii.gz

# # include 11 50 (caudate)
# for i in 2 41 7 46 85 11 50; do
#     # Extract single label mask
#     fslmaths $SynthSeg_IMG -thr $i -uthr $i -bin $TMPVISDIR/tmp_bin.nii.gz
#     # Add to cumulative mask
#     fslmaths $TMPVISDIR/temp_wm.nii.gz -add $TMPVISDIR/tmp_bin.nii.gz $TMPVISDIR/temp_wm.nii.gz
# done
# imcp ${TMPVISDIR}/temp_wm.nii.gz ${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz

# # get distance map
# # first we need a ventmask. label 4 and 43 are lateral ventricles
# fslmaths $SynthSeg_IMG -thr 4 -uthr 4 -bin $TMPVISDIR/tmp_bin.nii.gz
# fslmaths $SynthSeg_IMG -thr 43 -uthr 43 -bin -add $TMPVISDIR/tmp_bin.nii.gz $TMPVISDIR/ventmask.nii.gz

# distancemap -i $TMPVISDIR/ventmask.nii.gz -o ${TMPVISDIR}/ventdistmap_full.nii.gz
# fslmaths ${TMPVISDIR}/ventdistmap_full.nii.gz -mas ${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz ${TMPVISDIR}/ventdistmap.nii.gz
# fslmaths ${TMPVISDIR}/ventdistmap.nii.gz -thr -1 -uthr 6 -bin -fillh26 ${TMPVISDIR}/extended_ventricles.nii.gz
# fslmaths ${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz -add ${TMPVISDIR}/extended_ventricles.nii.gz -thr 0 -bin ${TMPVISDIR}/nonGMmask.nii.gz
# fslmaths ${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz -sub ${TMPVISDIR}/nonGMmask.nii.gz -thr 0 -bin ${TMPVISDIR}/GMmask.nii.gz

# distancemap -i ${TMPVISDIR}/GMmask.nii.gz -o ${TMPVISDIR}/GMdistmap_full.nii.gz
# fslmaths ${TMPVISDIR}/GMdistmap_full.nii.gz -mas ${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz ${TMPVISDIR}/GMdistmap.nii.gz
# fslmaths ${TMPVISDIR}/ventdistmap.nii.gz -mas ${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz ${OUTPUT_DIR}/${PREFIX}_ventdistmap.nii.gz
# fslmaths ${TMPVISDIR}/GMdistmap.nii.gz -mas ${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz ${OUTPUT_DIR}/${PREFIX}_GMdistmap.nii.gz

# # delete temp dir
# rm -rf ${TMPVISDIR}