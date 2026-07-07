#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 \
    --t1w <T1w image> \
    --t1w_to_mni_warp <T1w->MNI warp .nii.gz/.mgz> \
    --qsm_to_t1w_affine <QSM->T1w affine .mat> \
    --output_dir <Output directory> \
    --input   <in1.nii.gz [in2.nii.gz ...]> \
    --output1 <out1_T1w.nii.gz [out2_T1w.nii.gz ...]> \
    --output2 <out1_MNI.nii.gz [out2_MNI.nii.gz ...]>
EOF
  exit 1
}

T1W=""
WARP_T1W2MNI=""
AFF_QSM2T1W=""
OUTDIR=""
INPUTS=()
OUTS_T1W=()
OUTS_MNI=()

if [[ $# -eq 0 ]]; then
  usage
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --t1w)
      [[ $# -ge 2 ]] || usage
      T1W="$2"
      shift 2
      ;;
    --t1w_to_mni_warp)
      [[ $# -ge 2 ]] || usage
      WARP_T1W2MNI="$2"
      shift 2
      ;;
    --qsm_to_t1w_affine)
      [[ $# -ge 2 ]] || usage
      AFF_QSM2T1W="$2"
      shift 2
      ;;
    --output_dir)
      [[ $# -ge 2 ]] || usage
      OUTDIR="$2"
      shift 2
      ;;
    --input)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        INPUTS+=("$1")
        shift
      done
      ;;
    --output1)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        OUTS_T1W+=("$1")
        shift
      done
      ;;
    --output2)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        OUTS_MNI+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

[[ -n "$T1W" && -n "$WARP_T1W2MNI" && -n "$AFF_QSM2T1W" && -n "$OUTDIR" ]] || usage

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "Error: --input is empty"
  exit 2
fi

if [[ ${#OUTS_T1W[@]} -eq 0 ]]; then
  echo "Error: --output1 is empty"
  exit 2
fi

if [[ ${#OUTS_MNI[@]} -eq 0 ]]; then
  echo "Error: --output2 is empty"
  exit 2
fi

if [[ ${#INPUTS[@]} -ne ${#OUTS_T1W[@]} || ${#INPUTS[@]} -ne ${#OUTS_MNI[@]} ]]; then
  echo "Error: --input / --output1 / --output2 must have the same number of items"
  echo "       input: ${#INPUTS[@]}, out1: ${#OUTS_T1W[@]}, out2: ${#OUTS_MNI[@]}"
  exit 2
fi

[[ -f "$T1W" ]] || { echo "T1w not found: $T1W"; exit 4; }
[[ -f "$WARP_T1W2MNI" ]] || { echo "T1w->MNI warp not found: $WARP_T1W2MNI"; exit 4; }
[[ -f "$AFF_QSM2T1W" ]] || { echo "QSM->T1w affine not found: $AFF_QSM2T1W"; exit 4; }

mkdir -p "$OUTDIR"

echo "T1W:      $T1W"
echo "WARP:     $WARP_T1W2MNI"
echo "AFFINE:   $AFF_QSM2T1W"
echo "OUTDIR:   $OUTDIR"
echo "INPUTS:   ${INPUTS[*]}"
echo "OUTPUT1s: ${OUTS_T1W[*]}"
echo "OUTPUT2s: ${OUTS_MNI[*]}"

for i in "${!INPUTS[@]}"; do
  in_img="${INPUTS[$i]}"
  out_t1w="${OUTDIR}/${OUTS_T1W[$i]}"
  out_mni="${OUTDIR}/${OUTS_MNI[$i]}"
  t1w_recomputed=0

  [[ -f "$in_img" ]] || { echo "Input not found: $in_img"; exit 5; }
  mkdir -p "$(dirname "$out_t1w")" "$(dirname "$out_mni")"

  if [[ -s "$out_t1w" && -s "$out_mni" ]]; then
    echo "[$((i+1))/${#INPUTS[@]}] Both outputs exist and are non-empty. Skipping: $in_img"
    echo "  T1w: $out_t1w"
    echo "  MNI: $out_mni"
    continue
  fi

  if [[ ! -s "$out_t1w" ]]; then
    echo "[$((i+1))/${#INPUTS[@]}] FLIRT applyxfm: $in_img -> $out_t1w"
    flirt -in "$in_img" \
          -ref "$T1W" \
          -applyxfm -init "$AFF_QSM2T1W" \
          -out "$out_t1w"
    t1w_recomputed=1
  else
    echo "[$((i+1))/${#INPUTS[@]}] T1w output exists and is non-empty. Skipping FLIRT: $out_t1w"
  fi

  if [[ ! -s "$out_mni" || "$t1w_recomputed" -eq 1 ]]; then
    if [[ "$t1w_recomputed" -eq 1 && -s "$out_mni" ]]; then
      echo "[$((i+1))/${#INPUTS[@]}] T1w was recomputed. Recomputing MNI output: $out_mni"
    else
      echo "[$((i+1))/${#INPUTS[@]}] mri_convert -at: $out_t1w -> $out_mni"
    fi

    [[ -s "$out_t1w" ]] || { echo "Required T1w output not found or empty: $out_t1w"; exit 6; }
    mri_convert -at "$WARP_T1W2MNI" "$out_t1w" "$out_mni"
  else
    echo "[$((i+1))/${#INPUTS[@]}] MNI output exists and is non-empty. Skipping mri_convert: $out_mni"
  fi
done

echo "All transformations completed."