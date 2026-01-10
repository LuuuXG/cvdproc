#!/usr/bin/env bash
set -e

# ============================================
# Native-space lesion filling using SynthMorph
#
# Final output:
#   A single full-head lesion-filled T1 image at the path
#   specified by -o (no separate brain-only output).
#
# Requirements:
#   - FSL (fslreorient2std, fslswapdim, fslmaths)
#   - FreeSurfer 7.3+ (mri_synthstrip, mri_synthmorph)
#
# Usage:
#   lesion_fill_synthmorph_native.sh \
#       -t T1.nii.gz \
#       -m lesion_mask.nii.gz \
#       -o filled_T1_out.nii.gz \
#       [-j filled_T1_out.json] \
#       [-s smooth_sigma] \
#       [-d dilate_iters]
#
# ============================================

# -------------- Parse arguments --------------

T1_IN=""
LESION_IN=""
OUTFILE=""
JSON_OUT=""
SMOOTH_SIGMA=1      # default for small RSSI lesions
DILATE_ITERS=1      # default: 1 voxel shell

print_usage() {
  echo "Usage:"
  echo "  $(basename "$0") -t T1.nii.gz -m lesion_mask.nii.gz -o filled_T1_out.nii.gz [-j filled_T1_out.json] [-s smooth_sigma] [-d dilate_iters]"
  echo
  echo "Required:"
  echo "  -t   T1 image (NIfTI)"
  echo "  -m   lesion mask (same space as T1)"
  echo "  -o   output full-head lesion-filled T1 image path"
  echo
  echo "Optional:"
  echo "  -j   output JSON sidecar path for the filled T1 (if not set, no JSON is generated)"
  echo "  -s   smoothing sigma for lesion (default: 1)"
  echo "  -d   number of morphological dilations (default: 1)"
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -t)
      T1_IN="$2"
      shift 2
      ;;
    -m)
      LESION_IN="$2"
      shift 2
      ;;
    -o)
      OUTFILE="$2"
      shift 2
      ;;
    -j)
      JSON_OUT="$2"
      shift 2
      ;;
    -s)
      SMOOTH_SIGMA="$2"
      shift 2
      ;;
    -d)
      DILATE_ITERS="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# -------------- Check inputs --------------

if [[ -z "$T1_IN" || -z "$LESION_IN" || -z "$OUTFILE" ]]; then
  echo "Error: -t, -m, and -o are required."
  echo
  print_usage
  exit 1
fi

if [[ ! -f "$T1_IN" ]]; then
  echo "Error: T1 image not found: $T1_IN"
  exit 1
fi

if [[ ! -f "$LESION_IN" ]]; then
  echo "Error: lesion mask not found: $LESION_IN"
  exit 1
fi

OUTDIR="$(dirname "$OUTFILE")"
mkdir -p "$OUTDIR"

WORK="${OUTDIR}/lesion_fill_synthmorph_work"
mkdir -p "$WORK"

# -------------- Check required commands --------------

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH."
    exit 1
  fi
}

need_cmd fslreorient2std
need_cmd fslswapdim
need_cmd fslmaths
need_cmd mri_synthmorph
need_cmd mri_synthstrip

echo "============================================"
echo " Native-space lesion filling with SynthMorph"
echo " T1:          $T1_IN"
echo " Lesion mask: $LESION_IN"
echo " Output file: $OUTFILE"
if [[ -n "$JSON_OUT" ]]; then
  echo " JSON output: $JSON_OUT"
else
  echo " JSON output: (none, will not be created)"
fi
echo " Smooth sigma: $SMOOTH_SIGMA"
echo " Dilate iters: $DILATE_ITERS"
echo " Work dir:     $WORK"
echo "============================================"
echo

# -------------- Step 1: Reorient --------------

echo "=== Step 1: Reorient T1 and lesion to standard orientation ==="
T1_STD="${WORK}/T1_std.nii.gz"
LESION_STD="${WORK}/lesion_std.nii.gz"

fslreorient2std "$T1_IN"     "$T1_STD"
fslreorient2std "$LESION_IN" "$LESION_STD"

# -------------- Step 2: Brain extraction --------------

echo "=== Step 2: Brain extraction with mri_synthstrip ==="
T1_BRAIN="${WORK}/T1_brain.nii.gz"
T1_BRAIN_MASK="${WORK}/T1_brain_mask.nii.gz"

mri_synthstrip \
  -i "$T1_STD" \
  -o "$T1_BRAIN" \
  -m "$T1_BRAIN_MASK"

# -------------- Step 3: Lesion mask smoothing and dilation --------------

echo "=== Step 3: Smooth and dilate lesion mask (RSSI-tuned) ==="
LESION_BIN="${WORK}/lesion_bin.nii.gz"
LESION_DIL="${WORK}/lesion_dil.nii.gz"
LESION_INV_DIL="${WORK}/lesion_inv_dil.nii.gz"

# Smooth and threshold lesion
fslmaths "$LESION_STD" -s "$SMOOTH_SIGMA" -thr 0.2 -bin "$LESION_BIN"

# Dilate a small neighborhood around lesion
cp "$LESION_BIN" "$LESION_DIL"
if [[ "$DILATE_ITERS" -gt 0 ]]; then
  TMP="${WORK}/lesion_tmp.nii.gz"
  for ((i=1; i<=DILATE_ITERS; i++)); do
    fslmaths "$LESION_DIL" -dilM "$TMP"
    mv "$TMP" "$LESION_DIL"
  done
fi

# Inverse of dilated lesion within brain mask
fslmaths "$T1_BRAIN_MASK" -sub "$LESION_DIL" -thr 0 -bin "$LESION_INV_DIL"

# -------------- Step 4: Left-right flip T1 in native space --------------

echo "=== Step 4: Left-right flip T1 in native space ==="
T1_FLIP="${WORK}/T1_flip.nii.gz"
fslswapdim "$T1_STD" -x y z "$T1_FLIP"

# -------------- Step 5: Nonlinear registration (SynthMorph) --------------

echo "=== Step 5: Nonlinear registration (SynthMorph: flipped -> original) ==="
T1_FLIP2ORIG="${WORK}/T1_flip2orig.nii.gz"

mri_synthmorph \
  -o "$T1_FLIP2ORIG" \
  -g \
  "$T1_FLIP" \
  "$T1_STD"

# -------------- Step 6: Build donor patch --------------

echo "=== Step 6: Build donor patch from contralateral tissue ==="
DONOR_PATCH="${WORK}/donor_patch.nii.gz"

fslmaths "$T1_FLIP2ORIG" \
  -mas "$LESION_DIL" \
  "$DONOR_PATCH"

# -------------- Step 7: Preserve original T1 outside lesion neighborhood --------------

echo "=== Step 7: Preserve original T1 in non-lesion regions ==="
T1_HEALTHY="${WORK}/T1_healthy.nii.gz"

fslmaths "$T1_STD" \
  -mas "$LESION_INV_DIL" \
  "$T1_HEALTHY"

# -------------- Step 8: Combine and restore non-brain tissue --------------

echo "=== Step 8: Combine filled brain with original non-brain tissue ==="

# 8.1 Brain-only filled T1 (kept only in work dir)
FILLED_T1_BRAIN="${WORK}/T1_filled_brain.nii.gz"
fslmaths "$T1_HEALTHY" \
  -add "$DONOR_PATCH" \
  "$FILLED_T1_BRAIN"

# 8.2 Invert brain mask to get non-brain regions
BRAIN_MASK_INV="${WORK}/T1_brain_mask_inv.nii.gz"
fslmaths "$T1_BRAIN_MASK" -binv "$BRAIN_MASK_INV"

# 8.3 Extract non-brain tissue from original T1 (reoriented)
NONBRAIN="${WORK}/T1_nonbrain.nii.gz"
fslmaths "$T1_STD" -mas "$BRAIN_MASK_INV" "$NONBRAIN"

# 8.4 Keep only brain region from filled brain image
FILLED_T1_BRAIN_MASKED="${WORK}/T1_filled_brain_masked.nii.gz"
fslmaths "$FILLED_T1_BRAIN" -mas "$T1_BRAIN_MASK" "$FILLED_T1_BRAIN_MASKED"

# 8.5 Final full-head filled T1 = filled brain + original non-brain
fslmaths "$FILLED_T1_BRAIN_MASKED" \
  -add "$NONBRAIN" \
  "$OUTFILE"

# -------------- Optional: generate JSON sidecar --------------

# Helper to build a BIDS-like relative path starting at "sub-"
make_bids_like_path() {
  local p="$1"
  local rest="${p#*sub-}"
  if [[ "$rest" != "$p" ]]; then
    echo "sub-${rest}"
  else
    basename "$p"
  fi
}

if [[ -n "$JSON_OUT" ]]; then
  echo "=== Generating JSON sidecar ==="
  JSON_DIR="$(dirname "$JSON_OUT")"
  mkdir -p "$JSON_DIR"

  # Convert both file paths to BIDS-style relative paths
  T1_SRC_REL="$(make_bids_like_path "$T1_IN")"
  LESION_SRC_REL="$(make_bids_like_path "$LESION_IN")"

  cat > "$JSON_OUT" <<EOF
{
  "Description": "Lesion-filled T1w image generated using a native-space SynthMorph-based pipeline.",
  "Sources": [
    "bids:raw:${T1_SRC_REL}"
  ],
  "SpatialReference": "orig"
}
EOF
fi

# remove work dir
rm -rf "$WORK"

echo
echo "=== DONE ==="
echo "Final full-head lesion-filled T1 saved as:"
echo "  $OUTFILE"
if [[ -n "$JSON_OUT" ]]; then
  echo "JSON sidecar saved as:"
  echo "  $JSON_OUT"
fi
