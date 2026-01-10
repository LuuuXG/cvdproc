#!/usr/bin/env bash
set -euo pipefail

# === Argument parsing ===
SUBJECTS_DIR=$1
SUBJECT_ID=$2
FSQC_OUTPUT_DIR=$3

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <subjects_dir> <subject_id> <fsqc_output_dir>"
  exit 1
fi

mkdir -p "$FSQC_OUTPUT_DIR"

# === Check fsqc ===
if ! command -v run_fsqc &> /dev/null; then
  echo "Error: run_fsqc command not found. Please ensure fsqc is installed."
  exit 1
fi

# === Run fsqc ===
run_fsqc \
  --subjects_dir "$SUBJECTS_DIR" \
  --subjects "$SUBJECT_ID" \
  --output_dir "$FSQC_OUTPUT_DIR" \
  --screenshots \
  --skullstrip \
  --fornix \
  --shape \
  --outlier \
  --no-group

# === Post-process: flatten single-subject directories ===
MODULE_DIRS=(
  brainprint
  fornix
  metrics
  outliers
  screenshots
  skullstrip
  status
)

for module in "${MODULE_DIRS[@]}"; do
  MODULE_PATH="${FSQC_OUTPUT_DIR}/${module}"
  SUBJECT_SUBDIR="${MODULE_PATH}/${SUBJECT_ID}"

  # only act if <module>/<SUBJECT_ID> exists and is a directory
  if [[ -d "$SUBJECT_SUBDIR" ]]; then
    echo "[INFO] Flattening ${module}/${SUBJECT_ID}"

    # safety check: do not overwrite existing files
    for f in "$SUBJECT_SUBDIR"/*; do
      base=$(basename "$f")
      if [[ -e "${MODULE_PATH}/${base}" ]]; then
        echo "[ERROR] Target already exists: ${MODULE_PATH}/${base}"
        echo "        Refusing to overwrite. Please inspect manually."
        exit 1
      fi
    done

    # move contents up one level
    mv "$SUBJECT_SUBDIR"/* "$MODULE_PATH/"

    # remove empty subject directory
    rmdir "$SUBJECT_SUBDIR"
  fi
done

echo "[INFO] FSQC finished and output directories flattened for subject ${SUBJECT_ID}"
