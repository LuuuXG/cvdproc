#!/usr/bin/env bash
set -euo pipefail

###################### IMPORTANT NOTE ########################
# T1w and FLAIR should be aligned and skull-stripped already #
##############################################################

usage() {
  echo "Usage (with FLAIR): $0 <FLAIR_IMG> <T1w_IMG> <BRAIN_MASK> <SynthSeg_IMG> <OUTPUT_DIR> <PREFIX> [KEEP_T1W]" >&2
  echo "Usage (T1w only):   $0 NONE <T1w_IMG> <BRAIN_MASK> <SynthSeg_IMG> <OUTPUT_DIR> <PREFIX> [KEEP_T1W]" >&2
  echo "KEEP_T1W: 1 keep T1w output, 0 remove T1w output after preprocessing. Default: 1" >&2
  exit 1
}

if [[ $# -lt 6 ]]; then
  usage
fi

FLAIR_IMG="$1"
T1w_IMG="$2"
BRAIN_MASK="$3"
SynthSeg_IMG="$4"
OUTPUT_DIR="$5"
PREFIX="$6"
KEEP_T1W="${7:-1}"

if [[ "$KEEP_T1W" != "0" && "$KEEP_T1W" != "1" ]]; then
  echo "ERROR: KEEP_T1W must be 0 or 1." >&2
  exit 1
fi

HAS_FLAIR=1
if [[ "$FLAIR_IMG" == "NONE" || "$FLAIR_IMG" == "none" || "$FLAIR_IMG" == "null" || "$FLAIR_IMG" == "-" ]]; then
  HAS_FLAIR=0
fi

if [[ $HAS_FLAIR -eq 1 ]]; then
  if [[ ! -f "$FLAIR_IMG" ]]; then
    echo "ERROR: File not found: $FLAIR_IMG" >&2
    exit 2
  fi
fi

for f in "$T1w_IMG" "$BRAIN_MASK" "$SynthSeg_IMG"; do
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

if [[ $HAS_FLAIR -eq 1 ]]; then
  fslreorient2std "$FLAIR_IMG" "${TMPVISDIR}/FLAIR.nii.gz"
fi

fslreorient2std "$T1w_IMG"      "${TMPVISDIR}/T1.nii.gz"
fslreorient2std "$BRAIN_MASK"   "${TMPVISDIR}/brainmask.nii.gz"
fslreorient2std "$SynthSeg_IMG" "${TMPVISDIR}/SynthSeg.nii.gz"

if [[ $HAS_FLAIR -eq 1 ]]; then
  imcp "${TMPVISDIR}/FLAIR.nii.gz" "${OUTPUT_DIR}/${PREFIX}_FLAIR.nii.gz"
fi

imcp "${TMPVISDIR}/T1.nii.gz"        "${OUTPUT_DIR}/${PREFIX}_T1.nii.gz"
imcp "${TMPVISDIR}/brainmask.nii.gz" "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz"

if command -v mri_binarize >/dev/null 2>&1; then
  mri_binarize \
    --i "${TMPVISDIR}/SynthSeg.nii.gz" \
    --match 2 41 7 46 85 11 50 \
    --o "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz"
else
  echo "ERROR: mri_binarize not found. Please load FreeSurfer or implement a fallback." >&2
  exit 3
fi

mri_binarize \
  --i "${TMPVISDIR}/SynthSeg.nii.gz" \
  --match 4 43 \
  --o "${TMPVISDIR}/ventmask.nii.gz"

distancemap -i "${TMPVISDIR}/ventmask.nii.gz" -o "${TMPVISDIR}/ventdistmap_full.nii.gz"
fslmaths "${TMPVISDIR}/ventdistmap_full.nii.gz" -mas "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz" "${TMPVISDIR}/ventdistmap.nii.gz"

fslmaths "${TMPVISDIR}/ventdistmap.nii.gz" -thr -1 -uthr 6 -bin -fillh26 "${TMPVISDIR}/extended_ventricles.nii.gz"
fslmaths "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz" -add "${TMPVISDIR}/extended_ventricles.nii.gz" -thr 0 -bin "${TMPVISDIR}/nonGMmask.nii.gz"
fslmaths "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz" -sub "${TMPVISDIR}/nonGMmask.nii.gz" -thr 0 -bin "${TMPVISDIR}/GMmask.nii.gz"

distancemap -i "${TMPVISDIR}/GMmask.nii.gz" -o "${TMPVISDIR}/GMdistmap_full.nii.gz"
fslmaths "${TMPVISDIR}/GMdistmap_full.nii.gz" -mas "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz" "${TMPVISDIR}/GMdistmap.nii.gz"

fslmaths "${TMPVISDIR}/ventdistmap.nii.gz" -mas "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz" "${OUTPUT_DIR}/${PREFIX}_ventdistmap.nii.gz"
fslmaths "${TMPVISDIR}/GMdistmap.nii.gz"   -mas "${OUTPUT_DIR}/${PREFIX}_WMmask.nii.gz" "${OUTPUT_DIR}/${PREFIX}_GMdistmap.nii.gz"

if [[ $HAS_FLAIR -eq 0 ]]; then
  fslmaths "${OUTPUT_DIR}/${PREFIX}_T1.nii.gz" \
    -mas "${OUTPUT_DIR}/${PREFIX}_brainmask.nii.gz" \
    "${OUTPUT_DIR}/${PREFIX}_T1.nii.gz"
fi

if [[ $HAS_FLAIR -eq 1 && "$KEEP_T1W" == "0" ]]; then
  rm -f "${OUTPUT_DIR}/${PREFIX}_T1.nii.gz"
fi

echo "Done."
if [[ $HAS_FLAIR -eq 1 ]]; then
  if [[ "$KEEP_T1W" == "0" ]]; then
    echo "Prepared files: FLAIR + masks. T1w output was removed."
  else
    echo "Prepared files: FLAIR + T1 + masks."
  fi
else
  echo "Prepared files: T1 only + masks."
fi