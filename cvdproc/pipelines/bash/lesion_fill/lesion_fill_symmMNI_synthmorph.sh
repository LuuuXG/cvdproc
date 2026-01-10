#!/usr/bin/env bash
set -e

# ======================================================
# Symmetric-MNI-based contralateral mask generation (SynthMorph)
#
# Modes:
#   Default mode:
#     - Generates filled T1 in native space and JSON sidecar (as original)
#   Contra-only mode (--contra-only):
#     - Generates only contralateral lesion mask in native T1 space
#     - Skips writing filled T1 and JSON
#
# Required flags (both modes):
#   --t1 PATH
#   --lesion-mask PATH
#   --mni-template PATH
#   --warp-fwd PATH
#   --warp-inv PATH
#   --contra-mask PATH
#
# Additional required flags (default mode only):
#   --t1-mni PATH
#   --filled-t1 PATH
#   --bids-dir PATH
# ======================================================

print_usage() {
  echo "Usage:"
  echo "  $(basename "$0") \\"
  echo "    --t1 T1w.nii.gz \\"
  echo "    --lesion-mask lesion_T1w.nii.gz \\"
  echo "    --mni-template MNIsym_T1w.nii.gz \\"
  echo "    --warp-fwd warp_fwd.nii.gz \\"
  echo "    --warp-inv warp_inv.nii.gz \\"
  echo "    --contra-mask contra_mask_T1w.nii.gz \\"
  echo "    [--contra-only] \\"
  echo "    [--t1-mni T1_in_MNI.nii.gz] \\"
  echo "    [--filled-t1 filled_T1w.nii.gz] \\"
  echo "    [--bids-dir /path/to/BIDS_root]"
  echo
}

# ------------------ Parse flags ------------------------

T1_IN=""
MASK_IN=""
MNI_TEMPLATE=""
WARP_FWD=""
WARP_INV=""
T1_MNI_OUT=""
FILLED_T1NII=""
CONTRA_MASK_T1=""
BIDS_DIR=""
CONTRA_ONLY=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --t1)
      T1_IN="$2"; shift 2;;
    --lesion-mask)
      MASK_IN="$2"; shift 2;;
    --mni-template)
      MNI_TEMPLATE="$2"; shift 2;;
    --warp-fwd)
      WARP_FWD="$2"; shift 2;;
    --warp-inv)
      WARP_INV="$2"; shift 2;;
    --t1-mni)
      T1_MNI_OUT="$2"; shift 2;;
    --filled-t1)
      FILLED_T1NII="$2"; shift 2;;
    --contra-mask)
      CONTRA_MASK_T1="$2"; shift 2;;
    --bids-dir)
      BIDS_DIR="$2"; shift 2;;
    --contra-only)
      CONTRA_ONLY=true; shift 1;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1;;
  esac
done

# ------------------ Basic checks -----------------------

if [[ -z "$T1_IN" || -z "$MASK_IN" || -z "$MNI_TEMPLATE" || -z "$WARP_FWD" || -z "$WARP_INV" || -z "$CONTRA_MASK_T1" ]]; then
  echo "Error: missing required arguments."
  print_usage
  exit 1
fi

if [[ "$CONTRA_ONLY" == false ]]; then
  if [[ -z "$T1_MNI_OUT" || -z "$FILLED_T1NII" || -z "$BIDS_DIR" ]]; then
    echo "Error: default mode requires --t1-mni, --filled-t1, and --bids-dir."
    print_usage
    exit 1
  fi
fi

if [[ ! -f "$T1_IN" ]]; then
  echo "Error: T1 image not found: $T1_IN"
  exit 1
fi

if [[ ! -f "$MASK_IN" ]]; then
  echo "Error: lesion mask not found: $MASK_IN"
  exit 1
fi

if [[ ! -f "$MNI_TEMPLATE" ]]; then
  echo "Error: MNI template not found: $MNI_TEMPLATE"
  exit 1
fi

if [[ "$CONTRA_ONLY" == false ]]; then
  if [[ ! -d "$BIDS_DIR" ]]; then
    echo "Error: BIDS directory not found: $BIDS_DIR"
    exit 1
  fi
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: required command '$1' not found."; exit 1; }
}

need_cmd fslval
need_cmd fslmaths
need_cmd fslswapdim
need_cmd mri_synthmorph
need_cmd mri_convert
need_cmd readlink

# ------------------ Dimension check --------------------

echo "=== Checking T1 and lesion mask dimensions ==="

d1_t1=$(fslval "$T1_IN" dim1)
d2_t1=$(fslval "$T1_IN" dim2)
d3_t1=$(fslval "$T1_IN" dim3)
d4_t1=$(fslval "$T1_IN" dim4)

d1_ms=$(fslval "$MASK_IN" dim1)
d2_ms=$(fslval "$MASK_IN" dim2)
d3_ms=$(fslval "$MASK_IN" dim3)
d4_ms=$(fslval "$MASK_IN" dim4)

echo "  T1 dims:   ${d1_t1} x ${d2_t1} x ${d3_t1} x ${d4_t1}"
echo "  Mask dims: ${d1_ms} x ${d2_ms} x ${d3_ms} x ${d4_ms}"

if [[ "$d1_t1" -ne "$d1_ms" || "$d2_t1" -ne "$d2_ms" || "$d3_t1" -ne "$d3_ms" ]]; then
  echo "Error: T1 and lesion mask have different spatial dimensions (dim1-3)."
  exit 1
fi

if [[ "$d4_t1" -le 1 && "$d4_ms" -le 1 ]]; then
  echo "  Note: treating dim4 <= 1 as scalar volumes (OK)."
else
  if [[ "$d4_t1" -ne "$d4_ms" ]]; then
    echo "Error: T1 and lesion mask have different number of volumes (dim4)."
    exit 1
  fi
fi

# ------------------ Setup paths ------------------------

OUTDIR_WARP_FWD=$(dirname "$WARP_FWD")
OUTDIR_WARP_INV=$(dirname "$WARP_INV")
OUTDIR_CONTRA=$(dirname "$CONTRA_MASK_T1")

mkdir -p "$OUTDIR_WARP_FWD" "$OUTDIR_WARP_INV" "$OUTDIR_CONTRA"

# Workdir anchored to contra output directory (so contra-only mode needs no filled-t1 path)
WORK="${OUTDIR_CONTRA}/symmMNI_contra_work"
mkdir -p "$WORK"

# If contra-only and t1-mni is not provided, write it to workdir as an intermediate
if [[ "$CONTRA_ONLY" == true && -z "$T1_MNI_OUT" ]]; then
  T1_MNI_OUT="$WORK/T1_in_MNI_intermediate.nii.gz"
fi

# Default-mode JSON path
FILLED_JSON=""
if [[ "$CONTRA_ONLY" == false ]]; then
  FILLED_JSON="${FILLED_T1NII%.nii.gz}.json"
  OUTDIR_FILLED=$(dirname "$FILLED_T1NII")
  OUTDIR_T1_MNI=$(dirname "$T1_MNI_OUT")
  mkdir -p "$OUTDIR_FILLED" "$OUTDIR_T1_MNI"
fi

echo "==============================================="
echo " Symmetric-MNI Contra Mask (SynthMorph)"
echo " Mode:             $([[ "$CONTRA_ONLY" == true ]] && echo "contra-only" || echo "default")"
echo " T1:               $T1_IN"
echo " Lesion mask:      $MASK_IN"
echo " MNI template:     $MNI_TEMPLATE"
echo " Warp fwd:         $WARP_FWD"
echo " Warp inv:         $WARP_INV"
echo " T1 in MNI:        $T1_MNI_OUT"
echo " Contra mask T1:   $CONTRA_MASK_T1"
if [[ "$CONTRA_ONLY" == false ]]; then
  echo " Filled T1:        $FILLED_T1NII"
  echo " JSON sidecar:     $FILLED_JSON"
  echo " BIDS root:        $BIDS_DIR"
fi
echo " Workdir:          $WORK"
echo "==============================================="

# ------------------ Step 1. T1 -> MNI (SynthMorph) ----

echo "=== Step 1: SynthMorph registration T1 (native) -> MNI (symmetric) ==="

mri_synthmorph \
  -o "$T1_MNI_OUT" \
  -t "$WARP_FWD" \
  -T "$WARP_INV" \
  "$T1_IN" \
  "$MNI_TEMPLATE" -g

# ------------------ Step 2. Mask -> MNI ----------------

echo "=== Step 2: Warp lesion mask to MNI space ==="

MASK_MNI="$WORK/mask_MNI.nii.gz"
mri_convert -at "$WARP_FWD" "$MASK_IN" "$MASK_MNI" -rt nearest
fslmaths "$MASK_MNI" -thr 0.5 -bin "$MASK_MNI"

# ------------------ Step 3. Flip mask in MNI -----------

echo "=== Step 3: Left-right flip lesion mask in MNI space ==="

MASK_MNI_CONTRA="$WORK/mask_MNI_contra.nii.gz"
fslswapdim "$MASK_MNI" -x y z "$MASK_MNI_CONTRA"
fslmaths "$MASK_MNI_CONTRA" -thr 0.5 -bin "$MASK_MNI_CONTRA"

# ------------------ Step 4. Warp contra mask back -------

echo "=== Step 4: Warp contralateral lesion mask to native T1 space ==="

MASK_CONTRA_T1_TMP="$WORK/mask_T1_contra_tmp.nii.gz"
MASK_CONTRA_T1_SMOOTH="$WORK/mask_T1_contra_smooth.nii.gz"

mri_convert -at "$WARP_INV" "$MASK_MNI_CONTRA" "$MASK_CONTRA_T1_TMP" -rt nearest

# Light smoothing + re-threshold for a cleaner ROI boundary
fslmaths "$MASK_CONTRA_T1_TMP" -s 0.5 -thr 0.35 -bin "$MASK_CONTRA_T1_SMOOTH"

cp "$MASK_CONTRA_T1_SMOOTH" "$CONTRA_MASK_T1"

# ------------------ Default-mode filling + JSON ---------
if [[ "$CONTRA_ONLY" == false ]]; then

  echo "=== Step 5: Build filled T1 (MNI) and warp back to native (default mode) ==="

  MASK_MNI_DIL="$WORK/mask_MNI_dil.nii.gz"
  fslmaths "$MASK_MNI" -dilM "$MASK_MNI_DIL"

  T1_MNI_FLIP="$WORK/T1_MNI_flip.nii.gz"
  fslswapdim "$T1_MNI_OUT" -x y z "$T1_MNI_FLIP"

  T1_MNI_NONLES="$WORK/T1_MNI_nonlesion.nii.gz"
  DONOR_MNI="$WORK/donor_MNI.nii.gz"
  T1_FILLED_MNI="$WORK/T1_filled_MNI.nii.gz"

  MASK_MNI_DIL_INV="$WORK/mask_MNI_dil_inv.nii.gz"
  fslmaths "$MASK_MNI_DIL" -binv "$MASK_MNI_DIL_INV"

  fslmaths "$T1_MNI_OUT" -mas "$MASK_MNI_DIL_INV" "$T1_MNI_NONLES"
  fslmaths "$T1_MNI_FLIP" -mas "$MASK_MNI_DIL" "$DONOR_MNI"
  fslmaths "$T1_MNI_NONLES" -add "$DONOR_MNI" "$T1_FILLED_MNI"

  T1_FILLED_WARPED="$WORK/T1_filled_warped_to_T1.nii.gz"
  mri_convert -at "$WARP_INV" "$T1_FILLED_MNI" "$T1_FILLED_WARPED"

  echo "=== Step 6: Compose final filled T1 in native T1 space (default mode) ==="

  MASK_T1_CORE="$WORK/mask_T1_core.nii.gz"
  MASK_T1_DIL="$WORK/mask_T1_dil.nii.gz"
  MASK_T1_DIL_INV="$WORK/mask_T1_dil_inv.nii.gz"

  fslmaths "$MASK_IN" -thr 0.5 -bin "$MASK_T1_CORE"
  fslmaths "$MASK_T1_CORE" -dilM "$MASK_T1_DIL"
  fslmaths "$MASK_T1_DIL" -binv "$MASK_T1_DIL_INV"

  T1_NONLES_NATIVE="$WORK/T1_native_nonlesion.nii.gz"
  DONOR_NATIVE="$WORK/donor_native_in_lesion.nii.gz"
  T1_FILLED_NATIVE="$WORK/T1_filled_native.nii.gz"

  fslmaths "$T1_IN" -mas "$MASK_T1_DIL_INV" "$T1_NONLES_NATIVE"
  fslmaths "$T1_FILLED_WARPED" -mas "$MASK_T1_DIL" "$DONOR_NATIVE"
  fslmaths "$T1_NONLES_NATIVE" -add "$DONOR_NATIVE" "$T1_FILLED_NATIVE"

  cp "$T1_FILLED_NATIVE" "$FILLED_T1NII"

  echo "=== Step 7: Generate JSON sidecar with BIDS URIs in Sources (default mode) ==="

  BIDS_ROOT_ABS=$(readlink -f "$BIDS_DIR")
  T1_ABS=$(readlink -f "$T1_IN")
  MASK_ABS=$(readlink -f "$MASK_IN")

  REL_T1=""
  REL_MASK=""

  if [[ "$T1_ABS" == "$BIDS_ROOT_ABS"* ]]; then
    REL_T1="${T1_ABS#${BIDS_ROOT_ABS}/}"
  fi

  if [[ "$MASK_ABS" == "$BIDS_ROOT_ABS"* ]]; then
    REL_MASK="${MASK_ABS#${BIDS_ROOT_ABS}/}"
  fi

  URI_T1=""
  URI_MASK=""

  if [[ -n "$REL_T1" ]]; then
    URI_T1="bids::${REL_T1}"
  fi

  if [[ -n "$REL_MASK" ]]; then
    URI_MASK="bids::${REL_MASK}"
  fi

  {
    echo "{"
    echo "  \"Modality\": \"MR\""
    if [[ -n "$URI_T1" || -n "$URI_MASK" ]]; then
      echo "  ,\"Sources\": ["
      first=true
      if [[ -n "$URI_T1" ]]; then
        echo "    \"${URI_T1}\""
        first=false
      fi
      if [[ -n "$URI_MASK" ]]; then
        if $first; then
          echo "    \"${URI_MASK}\""
        else
          echo "    ,\"${URI_MASK}\""
        fi
      fi
      echo "  ]"
    fi
    echo "}"
  } > "$FILLED_JSON"

fi

# Optional: clean up work dir
rm -rf "$WORK"

echo
echo "=== DONE ==="
echo "Contra lesion mask:   $CONTRA_MASK_T1"
if [[ "$CONTRA_ONLY" == false ]]; then
  echo "Filled T1 image:      $FILLED_T1NII"
  echo "JSON sidecar:         $FILLED_JSON"
fi
echo "Warp fwd:             $WARP_FWD"
echo "Warp inv:             $WARP_INV"
echo
