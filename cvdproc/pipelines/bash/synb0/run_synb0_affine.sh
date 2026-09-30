#!/usr/bin/env bash
set -e

# Affine-only variant for leonyichencai/synb0-disco:v3.1.
# The container patch removes unused SyN outputs while preserving the inference pipeline.
# Use a fresh output directory when comparing this variant with the original.

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
PE_DIR=$(grep -oP '"PhaseEncodingDirection"\s*:\s*"\K[^"]+' "$DWI_JSON")
TOTAL_READOUT_TIME=$(grep -oP '"TotalReadoutTime"\s*:\s*\K[0-9eE\.+-]+' "$DWI_JSON")

if [[ -z "$PE_DIR" ]]; then
  echo "Error: PhaseEncodingDirection not found in $DWI_JSON" >&2
  exit 1
fi

if [[ -z "$TOTAL_READOUT_TIME" ]]; then
  echo "Error: TotalReadoutTime not found in $DWI_JSON" >&2
  exit 1
fi

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

# === Always create/update acqparam.txt ===
echo "Creating acqparam.txt (overwrite)..."
cat > "${INPUTS}/acqparam.txt" <<EOF
$PE_VECTOR $TOTAL_READOUT_TIME
$PE_VECTOR 0.01
EOF

# === Check whether b0_all exists ===
if [[ -f "$B0_ALL" ]]; then
  echo "b0_all.nii.gz already exists. Skipping synb0-disco."
else
  echo "Running mri_synthstrip..."
  mri_synthstrip -i "$T1W_IMG" -o "${INPUTS}/T1.nii.gz"

  flirt -in "${INPUTS}/T1.nii.gz" -ref "${INPUTS}/T1.nii.gz" -applyisoxfm 1 \
      -interp trilinear -out "${INPUTS}/T1.nii.gz"

  echo "Extracting b0 from DWI..."
  fslroi "$DWI_IMG" "${INPUTS}/b0.nii.gz" 0 1

  DOCKER_IMAGE="leonyichencai/synb0-disco:v3.1"

  echo "Running synb0-disco with rigid and affine registration only..."
  docker run --rm \
    -v "${INPUTS}:/INPUTS" \
    -v "${OUTPUTS}:/OUTPUTS" \
    -v "${FS_LICENSE_PATH}:/extra/freesurfer/license.txt" \
    --entrypoint /bin/bash \
    "$DOCKER_IMAGE" -e -c '
      # Patch only the disposable container; keep the installed image unchanged.
      prepare=/extra/prepare_input.sh
      grep -Fq "antsRegistrationSyNQuick.sh -d 3 -f" "$prepare"
      grep -q "^# Apply nonlinear transform" "$prepare"
      grep -q "^# Copy what you want" "$prepare"

      sed \
        -e "2i set -e" \
        -e "s/antsRegistrationSyNQuick.sh -d 3 -f/antsRegistrationSyNQuick.sh -d 3 -t a -f/" \
        -e "s/echo ANTS syn registration/echo ANTS rigid and affine registration/" \
        -e "/^# Apply nonlinear transform/,/^# Copy what you want/{ /^# Copy what you want/!d; }" \
        -e "/^cp .*1Warp.nii.gz /d" \
        -e "/^cp .*1InverseWarp.nii.gz /d" \
        -e "/^cp .*NONLIN_ATLAS_2_5_PATH /d" \
        "$prepare" > /tmp/prepare_input_affine.sh

      bash -n /tmp/prepare_input_affine.sh
      if grep -Eq "1Warp|1InverseWarp|NONLIN_ATLAS" /tmp/prepare_input_affine.sh; then
        echo "Error: unexpected nonlinear dependencies in the container preprocessing script." >&2
        exit 1
      fi
      cat /tmp/prepare_input_affine.sh > "$prepare"
      # Keep all five models in one process and disable training-only autograd.
      /extra/pytorch/bin/python3.6 - <<"PY"
from pathlib import Path

pipeline_path = Path("/extra/pipeline.sh")
pipeline = pipeline_path.read_text()
start = pipeline.index("# Run inference\n")
end = pipeline.index("# Take mean\n", start)
if "NUM_FOLDS=5" not in pipeline[start:end]:
    raise RuntimeError("Unexpected ensemble configuration in the container pipeline.")

Path("/tmp/inference_ensemble.py").write_text("""import glob
import sys
import time
import nibabel as nib
import torch

sys.path.insert(0, "/extra")
from inference import inference, UNet3D
import util

started = time.monotonic()
device = torch.device("cpu")
model = UNet3D(2, 1).to(device)
t1_path = "/OUTPUTS/T1_norm_lin_atlas_2_5.nii.gz"
b0_path = "/OUTPUTS/b0_d_lin_atlas_2_5.nii.gz"
template = nib.load(b0_path)
for fold in range(1, 6):
    pattern = "/extra/dual_channel_unet/num_fold_{}_total_folds_5_seed_1_num_epochs_100_lr_0.0001_betas_(0.9, 0.999)_weight_decay_1e-05_num_epoch_*.pth".format(fold)
    weights = glob.glob(pattern)
    if len(weights) != 1:
        raise RuntimeError("Expected one checkpoint for fold {}, found {}".format(fold, len(weights)))
    print("Performing inference on FOLD: {}".format(fold), flush=True)
    model.load_state_dict(torch.load(weights[0], map_location=device))
    with torch.no_grad():
        prediction = inference(t1_path, b0_path, model, device)
    output = nib.Nifti1Image(util.torch2nii(prediction.detach().cpu()), template.affine, template.header)
    nib.save(output, "/OUTPUTS/b0_u_lin_atlas_2_5_FOLD_{}.nii.gz".format(fold))
    del prediction, output
print("Five-model inference elapsed: {:.1f} seconds".format(time.monotonic() - started), flush=True)
""")
pipeline_path.write_text(pipeline[:start] + "# Run inference\npython3.6 /tmp/inference_ensemble.py\n\n" + pipeline[end:])
PY
      bash -n /extra/pipeline.sh
      exec bash -e /extra/pipeline.sh --stripped --notopup
    '

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
  "TotalReadoutTime": 0.000000001,
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
