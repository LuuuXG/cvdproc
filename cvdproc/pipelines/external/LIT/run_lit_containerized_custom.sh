#!/bin/bash
set -e

function usage() {
cat << EOF

Usage: run_lit_containerized_custom.sh \
  --input_image <input_t1w_volume> \
  --mask_image <lesion_mask_volume> \
  --output_directory <output_directory> \
  --lit_data_dir <lit_data_dir> \
  [--output_image <output_image>] \
  [OPTIONS]

This script runs LIT (Lesion Inpainting Tool) inside Docker and creates:
  (i)  an inpainted T1w image using a lesion mask
  (ii) (optional) whole brain segmentation and cortical surface reconstruction using FastSurferVINN

REQUIRED:
  -i, --input_image <input_image>
      Path to the input T1w volume
  -m, --mask_image <mask_image>
      Path to the lesion mask volume
  -o, --output_directory <output_directory>
      Path to the output directory
  --lit_data_dir <lit_data_dir>
      Directory containing LIT data (must include weights/)

OPTIONAL:
  --output_image <output_image>
      Final output image path. If set, the inpainted image
      (inpainting_volumes/inpainting_result.nii.gz) will be copied to this path.
      After copying, inpainting_images/ and inpainting_volumes/ under output_directory
      will be removed.
  --gpus <gpus>
      GPUs to use. Default: all
  --fastsurfer
      Run FastSurferVINN (requires FreeSurfer license)
  --fs_license <fs_license>
      Path to FreeSurfer license file (license.txt)
  -h, --help
      Print this message and exit
  --version
      Print the version number and exit

Examples:
  ./run_lit_containerized_custom.sh \
    -i t1w.nii.gz \
    -m lesion.nii.gz \
    -o ./output \
    --lit_data_dir /mnt/e/codes/cvdproc/cvdproc/data/lit \
    --dilate 2

  ./run_lit_containerized_custom.sh \
    -i t1w.nii.gz \
    -m lesion.nii.gz \
    -o ./output \
    --lit_data_dir /mnt/e/codes/cvdproc/cvdproc/data/lit \
    --output_image /path/to/final_inpainted.nii.gz \
    --dilate 2

EOF
}

# --------------------------------------------------
# Basic check
# --------------------------------------------------
if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

POSITIONAL_ARGS=()

# Hard-coded Docker image
DOCKER_IMAGE="deepmi/lit:0.5.0"

# Init
RUN_FASTSURFER=false
GPUS="all"
fs_license=""
LIT_DATA_DIR=""
OUTPUT_IMAGE=""

# --------------------------------------------------
# Parse arguments
# --------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    -i|--input_image)
      INPUT_IMAGE="$2"
      shift 2
      ;;
    -m|--mask_image)
      MASK_IMAGE="$2"
      shift 2
      ;;
    -o|--output_directory)
      OUT_DIR="$2"
      shift 2
      ;;
    --lit_data_dir)
      LIT_DATA_DIR="$2"
      shift 2
      ;;
    --output_image)
      OUTPUT_IMAGE="$2"
      shift 2
      ;;
    --fastsurfer)
      RUN_FASTSURFER=true
      shift
      ;;
    --fs_license)
      fs_license="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --version)
      echo "0.5.0"
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

# --------------------------------------------------
# Validate required inputs
# --------------------------------------------------
if [[ -z "${INPUT_IMAGE:-}" || -z "${MASK_IMAGE:-}" || -z "${OUT_DIR:-}" || -z "${LIT_DATA_DIR:-}" ]]; then
  echo "Error: --input_image, --mask_image, --output_directory, and --lit_data_dir are required."
  usage
  exit 1
fi

if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "Error: Input image not found: $INPUT_IMAGE"
  exit 1
fi

if [[ ! -f "$MASK_IMAGE" ]]; then
  echo "Error: Mask image not found: $MASK_IMAGE"
  exit 1
fi

if [[ ! -d "$LIT_DATA_DIR" ]]; then
  echo "Error: LIT data directory not found: $LIT_DATA_DIR"
  exit 1
fi

WEIGHTS_DIR="${LIT_DATA_DIR}/weights"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
  echo "Error: weights directory not found: $WEIGHTS_DIR"
  exit 1
fi

for w in model_coronal.pt model_axial.pt model_sagittal.pt; do
  if [[ ! -f "${WEIGHTS_DIR}/${w}" ]]; then
    echo "Error: Missing model weight: ${WEIGHTS_DIR}/${w}"
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

# Absolutize paths
INPUT_IMAGE="$(realpath "$INPUT_IMAGE")"
MASK_IMAGE="$(realpath "$MASK_IMAGE")"
OUT_DIR="$(realpath "$OUT_DIR")"
LIT_DATA_DIR="$(realpath "$LIT_DATA_DIR")"
WEIGHTS_DIR="${LIT_DATA_DIR}/weights"

if [[ -n "${OUTPUT_IMAGE:-}" ]]; then
  OUTPUT_IMAGE="$(realpath -m "$OUTPUT_IMAGE")"
fi

# Script directory (contains run_lit.sh)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/run_lit.sh" ]]; then
  echo "Error: ${SCRIPT_DIR}/run_lit.sh not found"
  exit 1
fi

# --------------------------------------------------
# FreeSurfer license handling
# --------------------------------------------------
if [[ "$RUN_FASTSURFER" = true ]]; then
  if [[ -z "${fs_license:-}" || ! -f "$fs_license" ]]; then
    echo "Error: --fastsurfer requires a valid --fs_license"
    exit 1
  fi
  POSITIONAL_ARGS+=("--fastsurfer")
else
  fs_license="/dev/null"
fi

# --------------------------------------------------
# Docker run
# --------------------------------------------------
docker run --gpus "device=${GPUS}" -it --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --rm \
  -v "${INPUT_IMAGE}:${INPUT_IMAGE}:ro" \
  -v "${MASK_IMAGE}:${MASK_IMAGE}:ro" \
  -v "${OUT_DIR}:${OUT_DIR}" \
  -v "${SCRIPT_DIR}:/inpainting:ro" \
  -v "${WEIGHTS_DIR}:/weights:ro" \
  -v "${fs_license}:/fs_license/license.txt:ro" \
  -u "$(id -u):$(id -g)" \
  "${DOCKER_IMAGE}" \
  /inpainting/run_lit.sh \
    -i "${INPUT_IMAGE}" \
    -m "${MASK_IMAGE}" \
    -o "${OUT_DIR}" \
    --weights_dir /weights \
    "${POSITIONAL_ARGS[@]}"

# --------------------------------------------------
# Optional: copy final output image and cleanup
# --------------------------------------------------
if [[ -n "${OUTPUT_IMAGE:-}" ]]; then
  SRC_IMAGE="${OUT_DIR}/inpainting_volumes/inpainting_result.nii.gz"

  if [[ ! -f "$SRC_IMAGE" ]]; then
    echo "Error: Inpainting result not found: $SRC_IMAGE"
    exit 1
  fi

  mkdir -p "$(dirname "$OUTPUT_IMAGE")"
  cp -v "$SRC_IMAGE" "$OUTPUT_IMAGE"

  rm -rf "${OUT_DIR}/inpainting_images" "${OUT_DIR}/inpainting_volumes"

  # if now OUT_DIR is empty, remove it
  if [[ -z "$(ls -A "$OUT_DIR")" ]]; then
    rmdir "$OUT_DIR"
  fi

  echo "Final output image copied to:"
  echo "  ${OUTPUT_IMAGE}"
  echo "Cleaned up:"
  echo "  ${OUT_DIR}/inpainting_images"
  echo "  ${OUT_DIR}/inpainting_volumes"
fi
