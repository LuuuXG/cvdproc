#!/usr/bin/env bash
set -euo pipefail

t1w_image=$1
output_dir=$2
output_csv_filename=$3

mkdir -p "$output_dir"

############################################
# Auto-detect paths
############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# brainageR_custom.sh:
# D:/Codes/cvdproc/cvdproc/pipelines/bash/brainage/brainageR_custom.sh
# project root:
# D:/Codes/cvdproc
cvdproc_dir="$(realpath "${SCRIPT_DIR}/../../../../..")"

brainageR_dir="${cvdproc_dir}/cvdproc/data/brainageR/"
spm_dir="${cvdproc_dir}/cvdproc/data/matlab_toolbox/spm12/"

if command -v matlab >/dev/null 2>&1; then
    matlab_path="$(command -v matlab)"
else
    echo "[ERROR] matlab not found in PATH"
    exit 1
fi

if [ -z "${FSLDIR:-}" ]; then
    if [ -d "/usr/local/fsl" ]; then
        FSLDIR="/usr/local/fsl"
    elif command -v fslmaths >/dev/null 2>&1; then
        FSLDIR="$(dirname "$(dirname "$(command -v fslmaths)")")"
    else
        echo "[ERROR] FSLDIR not found and fslmaths not found in PATH"
        exit 1
    fi
fi
export FSLDIR

if [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]; then
    # shellcheck disable=SC1090
    source "${FSLDIR}/etc/fslconf/fsl.sh"
fi

############################################
# Patch brainageR launcher script
############################################

brainageR_launcher="${brainageR_dir}/software/brainageR"

if [ ! -f "$brainageR_launcher" ]; then
    echo "[ERROR] brainageR launcher not found: $brainageR_launcher"
    exit 1
fi

tmp_file="$(mktemp)"

awk \
    -v new_cvdproc_dir="$cvdproc_dir" \
    -v new_brainageR_dir="$brainageR_dir" \
    -v new_spm_dir="$spm_dir" \
    -v new_matlab_path="$matlab_path" \
    -v new_FSLDIR="$FSLDIR" '
BEGIN {
    replaced_cvdproc=0
    replaced_brainageR=0
    replaced_spm=0
    replaced_matlab=0
    replaced_fsl=0
}
{
    if ($0 ~ /^cvdproc_dir=/) {
        print "cvdproc_dir=" new_cvdproc_dir
        replaced_cvdproc=1
    } else if ($0 ~ /^brainageR_dir=/) {
        print "brainageR_dir=" new_brainageR_dir
        replaced_brainageR=1
    } else if ($0 ~ /^spm_dir=/) {
        print "spm_dir=" new_spm_dir
        replaced_spm=1
    } else if ($0 ~ /^matlab_path=/) {
        print "matlab_path=" new_matlab_path
        replaced_matlab=1
    } else if ($0 ~ /^FSLDIR=/) {
        print "FSLDIR=" new_FSLDIR
        replaced_fsl=1
    } else {
        print
    }
}
END {
    if (!replaced_cvdproc)  print "cvdproc_dir=" new_cvdproc_dir
    if (!replaced_brainageR) print "brainageR_dir=" new_brainageR_dir
    if (!replaced_spm) print "spm_dir=" new_spm_dir
    if (!replaced_matlab) print "matlab_path=" new_matlab_path
    if (!replaced_fsl) print "FSLDIR=" new_FSLDIR
}
' "$brainageR_launcher" > "$tmp_file"

chmod --reference="$brainageR_launcher" "$tmp_file"
mv "$tmp_file" "$brainageR_launcher"

echo "[INFO] Patched brainageR launcher:"
echo "       $brainageR_launcher"
echo "[INFO] cvdproc_dir=$cvdproc_dir"
echo "[INFO] brainageR_dir=$brainageR_dir"
echo "[INFO] spm_dir=$spm_dir"
echo "[INFO] matlab_path=$matlab_path"
echo "[INFO] FSLDIR=$FSLDIR"

# Uncompress the T1w image (.nii.gz -> .nii)

t1w_image_filename=$(basename "$t1w_image" .nii.gz)
gunzip -c "$t1w_image" > "${output_dir}/${t1w_image_filename}.nii"

pixdim1=$(fslinfo "${output_dir}/${t1w_image_filename}.nii" | awk "/pixdim1/ {print \$2}")
pixdim2=$(fslinfo "${output_dir}/${t1w_image_filename}.nii" | awk "/pixdim2/ {print \$2}")
pixdim3=$(fslinfo "${output_dir}/${t1w_image_filename}.nii" | awk "/pixdim3/ {print \$2}")

if [[ "$pixdim1" != "1.000000" || "$pixdim2" != "1.000000" || "$pixdim3" != "1.000000" ]]; then
    flirt -in "${output_dir}/${t1w_image_filename}.nii" \
          -ref "${output_dir}/${t1w_image_filename}.nii" \
          -out "${output_dir}/${t1w_image_filename}_1mm.nii.gz" \
          -applyisoxfm 1

    gunzip -c "${output_dir}/${t1w_image_filename}_1mm.nii.gz" > "${output_dir}/${t1w_image_filename}.nii"
    rm -f "${output_dir}/${t1w_image_filename}_1mm.nii.gz"
fi

brainageR -f "${output_dir}/${t1w_image_filename}.nii" -o "${output_dir}/${output_csv_filename}"

rm -f "${output_dir}/${t1w_image_filename}.nii"

qc_dir="${output_dir}/slicesdir_${t1w_image_filename}.nii"
if [ -d "$qc_dir" ]; then
    mv "$qc_dir" "${output_dir}/QC"
fi