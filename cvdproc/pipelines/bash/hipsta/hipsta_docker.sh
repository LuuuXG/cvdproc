#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <hippocampus_file> <hemi> <lut> <output_dir>"
    exit 1
fi

hippocampus_file="$1"
hemi="$2"
lut="$3"
output_dir="$4"

DOCKER_IMAGE="hipsta:lxgcustom"

# Check FS_LICENSE
if [ -z "${FS_LICENSE:-}" ]; then
    echo "Error: FS_LICENSE is not set."
    echo "Please export FS_LICENSE=/path/to/license.txt"
    exit 1
fi

# Resolve paths
hippocampus_file="$(realpath "$hippocampus_file")"
output_dir="$(realpath "$output_dir")"
fs_license_file="$(realpath "$FS_LICENSE")"

# Checks
if [ ! -f "$hippocampus_file" ]; then
    echo "Error: hippocampus file not found: $hippocampus_file"
    exit 1
fi

if [ ! -d "$output_dir" ]; then
    echo "Error: output directory does not exist: $output_dir"
    exit 1
fi

if [ ! -f "$fs_license_file" ]; then
    echo "Error: FS_LICENSE file not found: $fs_license_file"
    exit 1
fi

if [ "$hemi" != "lh" ] && [ "$hemi" != "rh" ]; then
    echo "Error: hemi must be 'lh' or 'rh'"
    exit 1
fi

input_dir="$(dirname "$hippocampus_file")"
input_name="$(basename "$hippocampus_file")"

echo "Running hipsta container..."

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -v "${input_dir}:/input:ro" \
    -v "${output_dir}:/output" \
    -v "${fs_license_file}:/opt/freesurfer/.license:ro" \
    "${DOCKER_IMAGE}" \
    --filename "/input/${input_name}" \
    --outputdir /output \
    --hemi "${hemi}" \
    --lut "${lut}" \
    --no-qc \
    --long-filter \
    --gauss-filter-size 2 50

echo "hipsta container finished successfully."