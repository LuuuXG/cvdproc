# #!/bin/bash

# set -e

# # Initialize default values
# RUN_FASTSURFER=false
# DILATE=0
# SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# VERSION="$(grep "^version\\s*=\\s*\"" "$(dirname "${BASH_SOURCE[0]}")/pyproject.toml")"
# VERSION="${VERSION/version = /}"
# VERSION="${VERSION//\"/}"

# function usage() {
#     echo "Usage: $0 -i <input_t1w> -m <lesion_mask> -o <output_dir> [--fastsurfer] [--fs_license <path>]"
#     echo "Required arguments:"
#     echo "  -i, --input_image     : Input T1w image"
#     echo "  -m, --lesion_mask     : Lesion mask"
#     echo "  -o, --sd              : Output directory"
#     echo "Optional arguments:"
#     echo "  --fastsurfer          : Run FastSurfer (default: false)"
#     echo "  --fs_license          : Path to FreeSurfer license"
#     echo "Other arguments:"
#     echo "  --version             : Print version number and exit"
#     echo ""
#     echo "If you use LIT for research publications, please cite:"
#     echo ""
#     echo "Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole"
#     echo "  Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Accepted for Imaging Neuroscience."
#     exit
# }

# # Check if no arguments provided
# if [ $# -eq 0 ]; then
#     usage
# fi

# # Parse arguments
# POSITIONAL_ARGS=()
# while [[ $# -gt 0 ]]; do
#   case $1 in
#     -i|--input_image|--t1)
#       INPUT_IMAGE="$(realpath "$2")"
#       shift 2
#       ;;
#     -m|--lesion_mask)
#       MASK_IMAGE="$(realpath "$2")"
#       shift 2
#       ;;
#     -o|--sd)
#       OUT_DIR="$(realpath "$2")"
#       shift 2
#       ;;
#     --dilate)
#       DILATE="$2"
#       shift 2
#       ;;
#     --fs_license)
#       fs_license="$(realpath "$2")"
#       shift 2
#       ;;
#     --fastsurfer)
#       RUN_FASTSURFER=true
#       shift
#       ;;
#     -h|--help)
#       usage
#       ;;
#     --version)
#       project_dir="$(dirname "${BASH_SOURCE[0]}")"
#       hash_file="$(dirname "${BASH_SOURCE[0]}")/git.hash"
#       if [[ -n "$(which git)" ]] && (git -C "$project_dir" rev-parse 2>/dev/null ) ; then
#         HASH="+$(git -C "$project_dir" rev-parse --short HEAD)"
#       elif [[ -e "$hash_file" ]] ; then
#         HASH="+$(cat "$hash_file")"
#       else
#         HASH=""
#       fi
#       echo "$VERSION$HASH"
#       exit
#       ;;
#     *)
#       POSITIONAL_ARGS+=("$1")
#       shift
#       ;;
#   esac
# done

# # Validate required parameters
# if [ -z "$INPUT_IMAGE" ] || [ -z "$OUT_DIR" ]; then
#     echo "Error: Input image and output directory are required"
#     usage
# fi

# # Validate input files exist
# if [ ! -f "$INPUT_IMAGE" ]; then
#   echo "Error: Input file $INPUT_IMAGE does not exist"
#   exit 1
# fi

# if [ ! -z "$MASK_IMAGE" ] && [ ! -f "$MASK_IMAGE" ]; then
#   echo "Error: Mask file $MASK_IMAGE does not exist"
#   exit 1
# fi

# # Create output directory if it doesn't exist
# mkdir -p "$OUT_DIR"

# # Set up paths
# cd "$SCRIPT_DIR" || exit 1
# CKPT_CORONAL="$PWD/weights/model_coronal.pt"
# CKPT_AXIAL="$PWD/weights/model_axial.pt"
# CKPT_SAGITTAL="$PWD/weights/model_sagittal.pt"
# INPAINTED_IMG="$OUT_DIR/inpainting_volumes/inpainting_result.nii.gz"



# # Handle FastSurfer setup
# if [ "$RUN_FASTSURFER" = true ]; then
#     # Check for FASTSURFER_HOME
#     if [ -z "$FASTSURFER_HOME" ]; then
#         echo "Error: Requested FastSurfer but FASTSURFER_HOME environment variable not set"
#         exit 1
#     fi

#     # Handle license file
#     if [ -z "$fs_license" ]; then
#         for license_path in \
#             "/fs_license/license.txt" \
#             "$FREESURFER_HOME/license.txt" \
#             "$FREESURFER_HOME/.license"; do
#             if [ -f "$license_path" ]; then
#                 fs_license="$license_path"
#                 break
#             fi
#         done
#         if [ -z "$fs_license" ]; then
#             echo "Error: FreeSurfer license file not found"
#             exit 1
#         fi
#     fi
# fi

# # Run inpainting if mask is provided
# if [ ! -z "$MASK_IMAGE" ]; then
#   if [ ! -e "$INPAINTED_IMG" ]; then
#     echo "Running inpainting..."
#     mkdir -p "$OUT_DIR/inpainting_volumes"

#     python3 lit/utils/download_checkpoints.py

#     # Check for required model files
#     for model in "$CKPT_CORONAL" "$CKPT_AXIAL" "$CKPT_SAGITTAL"; do
#         if [ ! -f "$model" ]; then
#             echo "Error: Required model file not found: $model"
#             exit 1
#         fi
#     done
    
#     python3 lit/inpaint_image.py \
#       --input_image "$INPUT_IMAGE" \
#       --mask_image "$MASK_IMAGE" \
#       --out_dir "$OUT_DIR" \
#       --checkpoint_axial "$CKPT_AXIAL" \
#       --checkpoint_sagittal "$CKPT_SAGITTAL" \
#       --checkpoint_coronal "$CKPT_CORONAL" \
#       --dilate "$DILATE"
#   else
#     echo "Inpainted image already exists: $INPAINTED_IMG"
#   fi
# fi

# # Exit if not running FastSurfer
# if [ "$RUN_FASTSURFER" = false ]; then
#   echo "Finished inpainting"
#   exit 0
# fi

# # Setup FastSurfer paths
# S_DIR=$(basename "$OUT_DIR")
# OUT_DIR=$(dirname "$OUT_DIR")

# # Validate inpainted image exists
# if [ ! -f "$INPAINTED_IMG" ]; then
#   echo "Error: Inpainted file not found: $INPAINTED_IMG"
#   exit 1
# fi

# # Run FastSurfer
# fastsurfer_command="$FASTSURFER_HOME/run_fastsurfer_segmentation.sh --sid $S_DIR --sd $OUT_DIR"
# [ ! -z "$fs_license" ] && fastsurfer_command="$fastsurfer_command --fs_license $fs_license"
# fastsurfer_command="$fastsurfer_command --t1 ${MASK_IMAGE:+$INPAINTED_IMG}"
# fastsurfer_command="$fastsurfer_command ${POSITIONAL_ARGS[*]}"

# # Set PYTHONPATH
# export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$FASTSURFER_HOME/FastSurferCNN"

# # Run FastSurfer and post-processing
# if [ ! -f "$OUT_DIR/$S_DIR/scripts/recon-surf.done" ]; then
#   eval "$fastsurfer_command"
#   cd "$SCRIPT_DIR"
# else
#   echo "FastSurfer already ran"
# fi

# # Handle post-processing
# if [ -f "$MASK_IMAGE" ]; then
#   mask_image_fastsurfer_folder="$OUT_DIR/$S_DIR/inpainting_volumes/inpainting_mask.nii.gz"
  
#   for seg_file in "aparc+aseg.mgz" "aparc.DKTatlas+aseg.deep.mgz"; do
#     if [ -f "$OUT_DIR/$S_DIR/mri/$seg_file" ]; then
#       mv "$OUT_DIR/$S_DIR/mri/$seg_file" "$OUT_DIR/$S_DIR/mri/${seg_file%.mgz}_inpainted.mgz"
#       python lit/postprocessing/lesion_to_segmentation.py \
#         -i "$OUT_DIR/$S_DIR/mri/${seg_file%.mgz}_inpainted.mgz" \
#         -m "$mask_image_fastsurfer_folder" \
#         -o "$OUT_DIR/$S_DIR/mri/$seg_file"
#     fi
#   done

#   # Set directory paths
#   mdir="$OUT_DIR/$S_DIR/mri"
#   ldir="$OUT_DIR/$S_DIR/label"
#   sdir="$OUT_DIR/$S_DIR/surf"
#   inpaintdir="$OUT_DIR/$S_DIR/inpainting_volumes"
  
#   # TODO: Might want to include other surfaces
#   # Process left and right hemispheres
#   for hemi in "lh" "rh"; do
#     python lit/postprocessing/lesion_to_surface.py \
#       --inseg "$inpaintdir/inpainting_mask.nii.gz" \
#       --insurf "$sdir/$hemi.white.preaparc" \
#       --incort "$ldir/$hemi.cortex.label" \
#       --outaparc "$ldir/$hemi.lesion.label" \
#       --surflut "lit/postprocessing/DKTatlaslookup_lesion.txt" \
#       --seglut "lit/postprocessing/hemi.DKTatlaslookup_lesion.txt" \
#       --projmm 0 \
#       --radius 0 \
#       --single_label \
#       --to_annot "$ldir/$hemi.aparc.DKTatlas.annot"
#   done
# fi

#!/bin/bash
set -e

RUN_FASTSURFER=false
DILATE=0
WEIGHTS_DIR=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

VERSION="$(grep "^version\\s*=\\s*\"" "${SCRIPT_DIR}/pyproject.toml")"
VERSION="${VERSION/version = /}"
VERSION="${VERSION//\"/}"

function usage() {
  echo "Usage: $0 -i <input_t1w> -m <lesion_mask> -o <output_dir> [--dilate <int>] [--weights_dir <path>] [--fastsurfer] [--fs_license <path>]"
  echo ""
  echo "Required arguments:"
  echo "  -i, --input_image       Input T1w image"
  echo "  -m, --lesion_mask       Lesion mask"
  echo "  -o, --sd                Output directory"
  echo ""
  echo "Optional arguments:"
  echo "  --dilate <int>          Dilate mask (default: 0)"
  echo "  --weights_dir <path>    Directory containing model_*.pt (default: <script_dir>/weights)"
  echo "  --fastsurfer            Run FastSurfer (default: false)"
  echo "  --fs_license <path>     Path to FreeSurfer license"
  echo ""
  echo "Other arguments:"
  echo "  --version               Print version number and exit"
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input_image|--t1)
      INPUT_IMAGE="$(realpath "$2")"
      shift 2
      ;;
    -m|--lesion_mask)
      MASK_IMAGE="$(realpath "$2")"
      shift 2
      ;;
    -o|--sd)
      OUT_DIR="$(realpath "$2")"
      shift 2
      ;;
    --dilate)
      DILATE="$2"
      shift 2
      ;;
    --weights_dir)
      WEIGHTS_DIR="$2"
      shift 2
      ;;
    --fs_license)
      fs_license="$(realpath "$2")"
      shift 2
      ;;
    --fastsurfer)
      RUN_FASTSURFER=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    --version)
      project_dir="$SCRIPT_DIR"
      hash_file="${SCRIPT_DIR}/git.hash"
      if [[ -n "$(which git)" ]] && (git -C "$project_dir" rev-parse 2>/dev/null); then
        HASH="+$(git -C "$project_dir" rev-parse --short HEAD)"
      elif [[ -e "$hash_file" ]]; then
        HASH="+$(cat "$hash_file")"
      else
        HASH=""
      fi
      echo "${VERSION}${HASH}"
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Validate required parameters
if [[ -z "${INPUT_IMAGE:-}" || -z "${OUT_DIR:-}" ]]; then
  echo "Error: Input image and output directory are required"
  usage
fi

if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "Error: Input file does not exist: $INPUT_IMAGE"
  exit 1
fi

if [[ -n "${MASK_IMAGE:-}" && ! -f "$MASK_IMAGE" ]]; then
  echo "Error: Mask file does not exist: $MASK_IMAGE"
  exit 1
fi

mkdir -p "$OUT_DIR"

# Resolve weights dir (default: script_dir/weights)
if [[ -z "${WEIGHTS_DIR:-}" ]]; then
  WEIGHTS_DIR="${SCRIPT_DIR}/weights"
fi

# If user passed a relative path, resolve relative to script dir
if [[ "${WEIGHTS_DIR}" != /* ]]; then
  WEIGHTS_DIR="${SCRIPT_DIR}/${WEIGHTS_DIR}"
fi

WEIGHTS_DIR="$(realpath "$WEIGHTS_DIR")"

CKPT_CORONAL="${WEIGHTS_DIR}/model_coronal.pt"
CKPT_AXIAL="${WEIGHTS_DIR}/model_axial.pt"
CKPT_SAGITTAL="${WEIGHTS_DIR}/model_sagittal.pt"

INPAINTED_IMG="${OUT_DIR}/inpainting_volumes/inpainting_result.nii.gz"

# FastSurfer checks
if [[ "$RUN_FASTSURFER" = true ]]; then
  if [[ -z "${FASTSURFER_HOME:-}" ]]; then
    echo "Error: Requested FastSurfer but FASTSURFER_HOME is not set"
    exit 1
  fi

  if [[ -z "${fs_license:-}" ]]; then
    for license_path in \
      "/fs_license/license.txt" \
      "${FREESURFER_HOME:-}/license.txt" \
      "${FREESURFER_HOME:-}/.license"; do
      if [[ -n "$license_path" && -f "$license_path" ]]; then
        fs_license="$license_path"
        break
      fi
    done
  fi

  if [[ -z "${fs_license:-}" || ! -f "$fs_license" ]]; then
    echo "Error: FreeSurfer license file not found (use --fs_license)"
    exit 1
  fi
fi

# Run inpainting if mask is provided
if [[ -n "${MASK_IMAGE:-}" ]]; then
  if [[ ! -e "$INPAINTED_IMG" ]]; then
    echo "Running inpainting..."
    mkdir -p "${OUT_DIR}/inpainting_volumes"

    # Do NOT auto-download checkpoints here; require pre-provisioned weights.
    for model in "$CKPT_CORONAL" "$CKPT_AXIAL" "$CKPT_SAGITTAL"; do
      if [[ ! -f "$model" ]]; then
        echo "Error: Required model file not found: $model"
        echo "Hint: Provide --weights_dir <dir> containing model_coronal.pt, model_axial.pt, model_sagittal.pt"
        exit 1
      fi
    done

    python3 "${SCRIPT_DIR}/lit/inpaint_image.py" \
      --input_image "$INPUT_IMAGE" \
      --mask_image "$MASK_IMAGE" \
      --out_dir "$OUT_DIR" \
      --checkpoint_axial "$CKPT_AXIAL" \
      --checkpoint_sagittal "$CKPT_SAGITTAL" \
      --checkpoint_coronal "$CKPT_CORONAL" \
      --dilate "$DILATE"
  else
    echo "Inpainted image already exists: $INPAINTED_IMG"
  fi
fi

if [[ "$RUN_FASTSURFER" = false ]]; then
  echo "Finished inpainting"
  exit 0
fi

# Setup FastSurfer paths
S_DIR="$(basename "$OUT_DIR")"
OUT_PARENT="$(dirname "$OUT_DIR")"

if [[ ! -f "$INPAINTED_IMG" ]]; then
  echo "Error: Inpainted file not found: $INPAINTED_IMG"
  exit 1
fi

fastsurfer_command="${FASTSURFER_HOME}/run_fastsurfer_segmentation.sh --sid ${S_DIR} --sd ${OUT_PARENT}"
fastsurfer_command="${fastsurfer_command} --fs_license ${fs_license}"
fastsurfer_command="${fastsurfer_command} --t1 ${INPAINTED_IMG}"
fastsurfer_command="${fastsurfer_command} ${POSITIONAL_ARGS[*]}"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${FASTSURFER_HOME}/FastSurferCNN"

if [[ ! -f "${OUT_PARENT}/${S_DIR}/scripts/recon-surf.done" ]]; then
  eval "$fastsurfer_command"
else
  echo "FastSurfer already ran"
fi

if [[ -f "$MASK_IMAGE" ]]; then
  mask_image_fastsurfer_folder="${OUT_PARENT}/${S_DIR}/inpainting_volumes/inpainting_mask.nii.gz"

  for seg_file in "aparc+aseg.mgz" "aparc.DKTatlas+aseg.deep.mgz"; do
    if [[ -f "${OUT_PARENT}/${S_DIR}/mri/${seg_file}" ]]; then
      mv "${OUT_PARENT}/${S_DIR}/mri/${seg_file}" "${OUT_PARENT}/${S_DIR}/mri/${seg_file%.mgz}_inpainted.mgz"
      python "${SCRIPT_DIR}/lit/postprocessing/lesion_to_segmentation.py" \
        -i "${OUT_PARENT}/${S_DIR}/mri/${seg_file%.mgz}_inpainted.mgz" \
        -m "$mask_image_fastsurfer_folder" \
        -o "${OUT_PARENT}/${S_DIR}/mri/${seg_file}"
    fi
  done

  mdir="${OUT_PARENT}/${S_DIR}/mri"
  ldir="${OUT_PARENT}/${S_DIR}/label"
  sdir="${OUT_PARENT}/${S_DIR}/surf"
  inpaintdir="${OUT_PARENT}/${S_DIR}/inpainting_volumes"

  for hemi in "lh" "rh"; do
    python "${SCRIPT_DIR}/lit/postprocessing/lesion_to_surface.py" \
      --inseg "${inpaintdir}/inpainting_mask.nii.gz" \
      --insurf "${sdir}/${hemi}.white.preaparc" \
      --incort "${ldir}/${hemi}.cortex.label" \
      --outaparc "${ldir}/${hemi}.lesion.label" \
      --surflut "${SCRIPT_DIR}/lit/postprocessing/DKTatlaslookup_lesion.txt" \
      --seglut "${SCRIPT_DIR}/lit/postprocessing/hemi.DKTatlaslookup_lesion.txt" \
      --projmm 0 \
      --radius 0 \
      --single_label \
      --to_annot "${ldir}/${hemi}.aparc.DKTatlas.annot"
  done
fi
