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

This script runs LIT inside Docker and creates:
  (i) an inpainted T1w image using a lesion mask

REQUIRED:
  -i, --input_image <input_image>
      Path to the input T1w volume
  -m, --mask_image, --lesion_mask <mask_image>
      Path to the lesion mask volume
  -o, --output_directory <output_directory>
      Path to the output directory
  --lit_data_dir <lit_data_dir>
      Directory containing LIT data. It must include weights/

OPTIONAL:
  --output_image <output_image>
      Final output image path. If set, the inpainted image will be copied to this path.
  --gpus <gpus>
      GPUs to use. Default: all
  --device <auto|cpu|cuda>
      Inference device inside the container. Default: auto
  --cpu
      Shorthand for --device cpu
  --tag <tag>
      Docker tag to use. Default: 0.6.0
  -h, --help
      Print this message and exit
  --version
      Print the version number and exit

Examples:
  ./run_lit_containerized_custom.sh \
    -i t1w.nii.gz \
    -m lesion.nii.gz \
    -o ./output \
    --lit_data_dir /mnt/e/Codes/cvdproc/cvdproc/data/models/lit \
    --dilate 2

EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

POSITIONAL_ARGS=()

DOCKER_REPO="deepmi/lit"
VERSION="0.6.0"
DEVICE="auto"
GPUS="all"
LIT_DATA_DIR=""
OUTPUT_IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --gpus requires a value"
        exit 1
      fi
      GPUS="$2"
      shift 2
      ;;
    --device)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --device requires a value"
        exit 1
      fi
      if [[ "$2" != "auto" && "$2" != "cpu" && "$2" != "cuda" ]]; then
        echo "Error: --device must be one of: auto, cpu, cuda"
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --cpu)
      DEVICE="cpu"
      shift
      ;;
    -i|--input_image)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --input_image requires a value"
        exit 1
      fi
      INPUT_IMAGE="$2"
      shift 2
      ;;
    -m|--mask_image|--lesion_mask)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --mask_image/--lesion_mask requires a value"
        exit 1
      fi
      MASK_IMAGE="$2"
      shift 2
      ;;
    -o|--output_directory)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --output_directory requires a value"
        exit 1
      fi
      OUT_DIR="$2"
      shift 2
      ;;
    --lit_data_dir)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --lit_data_dir requires a value"
        exit 1
      fi
      LIT_DATA_DIR="$2"
      shift 2
      ;;
    --output_image)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --output_image requires a value"
        exit 1
      fi
      OUTPUT_IMAGE="$2"
      shift 2
      ;;
    --tag)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "Error: --tag requires a value"
        exit 1
      fi
      VERSION="$2"
      shift 2
      ;;
    --version)
      echo "$VERSION"
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

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

for weight_file in model_coronal.pt model_axial.pt model_sagittal.pt; do
  if [[ ! -f "${WEIGHTS_DIR}/${weight_file}" ]]; then
    echo "Error: Missing model weight: ${WEIGHTS_DIR}/${weight_file}"
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

INPUT_IMAGE="$(realpath "$INPUT_IMAGE")"
MASK_IMAGE="$(realpath "$MASK_IMAGE")"
OUT_DIR="$(realpath "$OUT_DIR")"
LIT_DATA_DIR="$(realpath "$LIT_DATA_DIR")"
WEIGHTS_DIR="${LIT_DATA_DIR}/weights"

if [[ -n "${OUTPUT_IMAGE:-}" ]]; then
  OUTPUT_IMAGE="$(realpath -m "$OUTPUT_IMAGE")"
fi

if [[ "$DEVICE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    RESOLVED_DEVICE="cuda"
  else
    RESOLVED_DEVICE="cpu"
  fi
else
  RESOLVED_DEVICE="$DEVICE"
fi

if [[ "$VERSION" == *"/"* || "$VERSION" == *":"* ]]; then
  IMAGE="$VERSION"
else
  IMAGE="${DOCKER_REPO}:${VERSION}"
fi

DOCKER_ARGS=(run --ipc=host
  --ulimit memlock=-1 --ulimit stack=67108864 --rm
  -v "${INPUT_IMAGE}:${INPUT_IMAGE}:ro"
  -v "${MASK_IMAGE}:${MASK_IMAGE}:ro"
  -v "${OUT_DIR}:${OUT_DIR}"
  -v "${WEIGHTS_DIR}:/weights:ro"
  -u "$(id -u):$(id -g)")

if [[ "$RESOLVED_DEVICE" == "cuda" ]]; then
  DOCKER_ARGS+=(--gpus "device=${GPUS}")
fi

echo "Running LIT with Docker image: ${IMAGE}"
echo "Input image: ${INPUT_IMAGE}"
echo "Mask image: ${MASK_IMAGE}"
echo "Output directory: ${OUT_DIR}"
echo "Weights directory: ${WEIGHTS_DIR}"
echo "Device: ${RESOLVED_DEVICE}"

docker "${DOCKER_ARGS[@]}" \
  "${IMAGE}" \
  -i "${INPUT_IMAGE}" \
  -m "${MASK_IMAGE}" \
  -o "${OUT_DIR}" \
  --device "${RESOLVED_DEVICE}" \
  -c_coronal /weights/model_coronal.pt \
  -c_axial /weights/model_axial.pt \
  -c_sagittal /weights/model_sagittal.pt \
  "${POSITIONAL_ARGS[@]}"

if [[ -n "${OUTPUT_IMAGE:-}" ]]; then
  SRC_IMAGE="${OUT_DIR}/inpainting_volumes/inpainting_result.nii.gz"

  if [[ ! -f "$SRC_IMAGE" ]]; then
    echo "Error: Inpainting result not found: $SRC_IMAGE"
    echo "Please check the files generated under: $OUT_DIR"
    exit 1
  fi

  mkdir -p "$(dirname "$OUTPUT_IMAGE")"
  cp -v "$SRC_IMAGE" "$OUTPUT_IMAGE"

  rm -rf "${OUT_DIR}/inpainting_images" "${OUT_DIR}/inpainting_volumes"

  if [[ -d "$OUT_DIR" && -z "$(ls -A "$OUT_DIR")" ]]; then
    rmdir "$OUT_DIR"
  fi

  echo "Final output image copied to:"
  echo "  ${OUTPUT_IMAGE}"
fi