#!/usr/bin/env bash
set -euo pipefail

BIDS_DIR=""
OUT_DIR=""

usage() {
    echo "Usage: bash collect_baseline_mni_masks.sh --bids_dir /path/to/BIDS --out_dir /path/to/output"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids_dir)
            BIDS_DIR="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$BIDS_DIR" || -z "$OUT_DIR" ]]; then
    usage
fi

LESION_DIR="${BIDS_DIR}/derivatives/lesion_mask"
MASK_DIR="${OUT_DIR}/masks"
ZIP_DIR="${OUT_DIR}/zips"
LOG_FILE="${OUT_DIR}/collect_baseline_mni_masks.log"

mkdir -p "$MASK_DIR" "$ZIP_DIR"
: > "$LOG_FILE"

if [[ ! -d "$LESION_DIR" ]]; then
    echo "Error: lesion_mask directory not found: $LESION_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "BIDS_DIR: $BIDS_DIR" | tee -a "$LOG_FILE"
echo "OUT_DIR: $OUT_DIR" | tee -a "$LOG_FILE"
echo "LESION_DIR: $LESION_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

count=0
missing=0

for sub_dir in "$LESION_DIR"/sub-*; do
    [[ -d "$sub_dir" ]] || continue

    sub_id="$(basename "$sub_dir")"
    ses_dir="${sub_dir}/ses-baseline"

    if [[ ! -d "$ses_dir" ]]; then
        echo "Missing baseline folder: ${sub_id}/ses-baseline" | tee -a "$LOG_FILE"
        missing=$((missing + 1))
        continue
    fi

    shopt -s nullglob
    files=("$ses_dir"/*space-MNI152NLin6ASym_desc-RSSI_mask.nii.gz)
    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Missing mask: ${sub_id}/ses-baseline" | tee -a "$LOG_FILE"
        missing=$((missing + 1))
        continue
    fi

    if [[ ${#files[@]} -gt 1 ]]; then
        echo "Warning: multiple masks found for ${sub_id}/ses-baseline; copying all matched files" | tee -a "$LOG_FILE"
    fi

    for f in "${files[@]}"; do
        cp -f "$f" "$MASK_DIR/"
        echo "Copied: $f" | tee -a "$LOG_FILE"
        count=$((count + 1))
    done
done

echo "" | tee -a "$LOG_FILE"
echo "Copied masks: $count" | tee -a "$LOG_FILE"
echo "Missing entries: $missing" | tee -a "$LOG_FILE"

if [[ "$count" -eq 0 ]]; then
    echo "No mask files copied. Skip zip creation." | tee -a "$LOG_FILE"
    exit 0
fi

rm -f "$ZIP_DIR"/batch*.zip

mapfile -t mask_files < <(find "$MASK_DIR" -maxdepth 1 -type f -name "*space-MNI152NLin6ASym_desc-RSSI_mask.nii.gz" | sort)

batch_size=10
batch_num=1
start=0
total=${#mask_files[@]}

while [[ $start -lt $total ]]; do
    end=$((start + batch_size))
    if [[ $end -gt $total ]]; then
        end=$total
    fi

    zip_file="${ZIP_DIR}/batch${batch_num}.zip"
    tmp_list="$(mktemp)"

    for ((i=start; i<end; i++)); do
        basename "${mask_files[$i]}" >> "$tmp_list"
    done

    (
        cd "$MASK_DIR"
        zip -q "$zip_file" -@ < "$tmp_list"
    )

    rm -f "$tmp_list"
    echo "Created: $zip_file" | tee -a "$LOG_FILE"

    start=$end
    batch_num=$((batch_num + 1))
done

echo "" | tee -a "$LOG_FILE"
echo "Done." | tee -a "$LOG_FILE"