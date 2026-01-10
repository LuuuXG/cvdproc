#!/usr/bin/env bash

set -e

input_t1w=$1
output_dir=$2

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <T1w.nii.gz> <chpseg_output_dir>"
  exit 1
fi

mkdir -p "$output_dir"

input_temp_dir="${output_dir}/INPUTS"
mkdir -p "$input_temp_dir"
# make a copy of T1w in input_temp_dir -> T1w.nii.gz
cp "$input_t1w" "${input_temp_dir}/T1w.nii.gz"

docker_image="kilianhett/chp_seg:1.0.1"

echo "Running CHP Segmentation with Docker image: $docker_image"
docker run -ti --rm \
  --gpus all \
  -v "$input_temp_dir":/data/in \
  -v "$output_dir":/data/out \
  "$docker_image" \
  --sequence_type T1 \
  --name_pattern T1w.nii.gz

# delete input_temp_dir
rm -rf "$input_temp_dir"

# === post-process ===
report1_csv="${output_dir}/T1w/report.csv"
report2_csv="${output_dir}/report_chp_volumes.csv"

swap_cols() {
  local in_csv="$1"
  [[ -f "$in_csv" ]] || return 0

  local tmp_csv="${in_csv%.csv}_fixed.csv"
  python - "$in_csv" "$tmp_csv" <<'PY'
import csv, sys

in_csv, out_csv = sys.argv[1], sys.argv[2]
with open(in_csv, 'r', newline='', encoding='utf-8-sig') as f_in:
    rows = list(csv.reader(f_in))
if not rows:
    sys.exit(0)

header = rows[0]
# Normalize header fields (strip surrounding quotes/spaces)
norm = [h.strip().strip('"').strip("'") for h in header]

try:
    idx_r = norm.index("volume_right_mm3")
    idx_l = norm.index("volume_left_mm3")
except ValueError:
    # If either header is missing, just copy through unchanged
    with open(out_csv, 'w', newline='', encoding='utf-8') as f_out:
        csv.writer(f_out).writerows(rows)
    sys.exit(0)

# Write swapped file
with open(out_csv, 'w', newline='', encoding='utf-8') as f_out:
    w = csv.writer(f_out)
    w.writerow(header)
    for r in rows[1:]:
        # Pad short rows defensively
        if len(r) <= max(idx_r, idx_l):
            r = r + [''] * (max(idx_r, idx_l) + 1 - len(r))
        r[idx_r], r[idx_l] = r[idx_l], r[idx_r]
        w.writerow(r)
PY

  mv -f "$tmp_csv" "$in_csv"
  echo "Fixed Left/Right columns in: $in_csv"
}

swap_cols "$report1_csv"
swap_cols "$report2_csv"

echo "Post-processing completed. Reports saved with corrected Left/Right volumes."