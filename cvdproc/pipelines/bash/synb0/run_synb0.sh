#!/usr/bin/env bash
set -e

# === Argument parsing ===
T1W_IMG=$1
DWI_IMG=$2
OUTPUT_DIR=$3
DWI_JSON=$4
FMAP_DIR=$5

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <T1w.nii.gz> <DWI.nii.gz> <synb0_output_dir> <dwi.json> <fmap_output_dir>"
  exit 1
fi

# === Path setup ===
INPUTS="${OUTPUT_DIR}/INPUTS"
OUTPUTS="${OUTPUT_DIR}/OUTPUTS"
B0_ALL="${OUTPUT_DIR}/b0_all.nii.gz"

FS_LICENSE_PATH=${FS_LICENSE:-""}
if [[ -z "$FS_LICENSE_PATH" ]]; then
  echo "Error: FS_LICENSE environment variable FS_LICENSE is not set."
  exit 1
fi

mkdir -p "$INPUTS" "$OUTPUTS" "$FMAP_DIR"

# === Extract parameters from the DWI JSON ===
# PE_DIR=$(jq -r '.PhaseEncodingDirection' "$DWI_JSON")
# TOTAL_READOUT_TIME=$(jq -r '.TotalReadoutTime' "$DWI_JSON")
PE_DIR=$(grep -oP '"PhaseEncodingDirection"\s*:\s*"\K[^"]+' "$DWI_JSON")
TOTAL_READOUT_TIME=$(grep -oP '"TotalReadoutTime"\s*:\s*\K[0-9eE\.+-]+' "$DWI_JSON")

# === Map PhaseEncodingDirection to a vector ===
declare -A PE_MAP=( ["i"]="1 0 0" ["i-"]="-1 0 0" ["j"]="0 1 0" ["j-"]="0 -1 0" ["k"]="0 0 1" ["k-"]="0 0 -1" )
PE_VECTOR="${PE_MAP[$PE_DIR]}"
if [[ -z "$PE_VECTOR" ]]; then
  echo "Unsupported PhaseEncodingDirection: $PE_DIR"
  exit 1
fi

# === Convert PhaseEncodingDirection to a direction label ===
# === Use the opposite direction ===
reverse_pe_dir() {
  case "$1" in
    i) echo "i-" ;;
    i-) echo "i" ;;
    j) echo "j-" ;;
    j-) echo "j" ;;
    k) echo "k-" ;;
    k-) echo "k" ;;
    *) echo "Unsupported PhaseEncodingDirection: $1" >&2; exit 1 ;;
  esac
}
#FMAP_PE_DIR=$(reverse_pe_dir "$PE_DIR")
FMAP_PE_DIR=$PE_DIR

declare -A DIR_LABEL_MAP=( ["i"]="LR" ["i-"]="RL" ["j"]="PA" ["j-"]="AP" ["k"]="IS" ["k-"]="SI" )
DIR_LABEL="${DIR_LABEL_MAP[$FMAP_PE_DIR]}"
if [[ -z "$DIR_LABEL" ]]; then
  echo "Unknown dir label for PhaseEncodingDirection: $FMAP_PE_DIR"
  exit 1
fi

# === Check whether b0_all already exists ===
if [[ -f "$B0_ALL" ]]; then
  echo "b0_all.nii.gz already exists. Skipping synb0-disco."
else
  echo "Running mri_synthstrip..."
  mri_synthstrip -i "$T1W_IMG" -o "${INPUTS}/T1.nii.gz"

  flirt -in "${INPUTS}/T1.nii.gz" -ref "${INPUTS}/T1.nii.gz" -applyisoxfm 1 \
      -interp trilinear -out "${INPUTS}/T1.nii.gz"

  echo "Extracting b0 from DWI..."
  fslroi "$DWI_IMG" "${INPUTS}/b0.nii.gz" 0 1

  echo "Creating acqparam.txt..."
  cat > "${INPUTS}/acqparam.txt" <<EOF
$PE_VECTOR $TOTAL_READOUT_TIME
$PE_VECTOR 0
EOF

  DOCKER_IMAGE="leonyichencai/synb0-disco:v3.1"
  # if ! docker inspect "$DOCKER_IMAGE" > /dev/null 2>&1; then
  #   docker pull "$DOCKER_IMAGE"
  # fi

  echo "Running synb0-disco container..."
  docker run --rm \
    -v "${INPUTS}:/INPUTS" \
    -v "${OUTPUTS}:/OUTPUTS" \
    -v "${FS_LICENSE_PATH}:/extra/freesurfer/license.txt" \
    "$DOCKER_IMAGE" \
    --user 1000:1000 \
    --stripped \
    --notopup

  echo "Merging b0 images..."
  fslmerge -t "$B0_ALL" "${OUTPUTS}/b0_d_smooth.nii.gz" "${OUTPUTS}/b0_u.nii.gz"
fi

# === Build the IntendedFor field ===
DWI_BIDS_RELPATH=$(echo "$DWI_JSON" | sed -E 's|.*/(ses-[^/]+/dwi/[^/]+)\.json|\1.nii.gz|')
INTENDED_FOR="${DWI_BIDS_RELPATH}"

# === Extract sub- and ses- labels ===
SUBJECT=$(echo "$DWI_JSON" | grep -oP 'sub-[^/]+' | head -n 1)
SESSION=$(echo "$DWI_JSON" | grep -oP 'ses-[^/]+' | head -n 1)

FMAP_BASENAME="${SUBJECT}_${SESSION}_dir-${DIR_LABEL}_acq-synb0_epi"
FMAP_NII="${FMAP_DIR}/${FMAP_BASENAME}.nii.gz"
FMAP_JSON="${FMAP_DIR}/${FMAP_BASENAME}.json"
FMAP_BVAL="${FMAP_DIR}/${FMAP_BASENAME}.bval"
FMAP_BVECS="${FMAP_DIR}/${FMAP_BASENAME}.bvec"

echo "Copying undistorted b0 image to fmap directory..."
cp "${OUTPUTS}/b0_u.nii.gz" "$FMAP_NII"

echo "Writing fmap JSON with IntendedFor..."
cat > "$FMAP_JSON" <<EOF
{
  "PhaseEncodingDirection": "$FMAP_PE_DIR",
  "TotalReadoutTime": 0.0000001,
  "EffectiveEchoSpacing": 0.0,
  "IntendedFor": "$INTENDED_FOR"
}
EOF

echo "Creating empty bval and bvec files for fmap..."
echo "0" > "$FMAP_BVAL"
echo -e "0\n0\n0" > "$FMAP_BVECS"

echo " Synb0-DISCO complete."
echo " Fmap image: $FMAP_NII"
echo " Fmap JSON : $FMAP_JSON"
