#!/usr/bin/env bash
set -euo pipefail

N_JOBS=12

usage() {
  echo "Usage:"
  echo "  bash run_ARTS_fast_synthmorph.sh <subject> <age> <sex> <T1w_brain.nii.gz> <FLAIR_brain.nii.gz> <FA.nii.gz> <synthseg.nii.gz> <WMH_mask.nii.gz> <output_root>"
  echo ""
  echo "Example:"
  echo "  bash run_ARTS_fast_synthmorph.sh AFib0241 75 1 T1w_brain.nii.gz FLAIR_brain.nii.gz FA.nii.gz synthseg.nii.gz WMH_mask.nii.gz /mnt/e/Neuroimage/ARTS/output_fast_synthmorph"
}

if [ "$#" -ne 9 ]; then
  usage
  exit 1
fi

SUBJECT="$1"
AGE="$2"
SEX="$3"
T1_IN="$(realpath "$4")"
FLAIR_IN="$(realpath "$5")"
FA_IN="$(realpath "$6")"
SYNTHSEG_IN="$(realpath "$7")"
WMH_IN="$(realpath "$8")"
OUT_ROOT="$(realpath "$9")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTS_DATA_DIR="$(realpath "${SCRIPT_DIR}/../../../data/arts")"

SUB_OUT="${OUT_ROOT}/${SUBJECT}"
TABLE_DIR="${SUB_OUT}/input_tables"
TABLE_FILE="${TABLE_DIR}/input_${SUBJECT}_fast_synthmorph.csv"

# Derive paths from arts_data_dir
IIT_FA="${ARTS_DATA_DIR}/IITmean_FA.nii.gz"
IIT_ATLAS_DIR="${ARTS_DATA_DIR}/IIT_atlas"
CALC_DTI_ROIS="${ARTS_DATA_DIR}/scripts/feature_extraction/calculate_DTI_ROIs"
CLASSIFIER_PY="${ARTS_DATA_DIR}/scripts/classifier/final_classifier.py"
CLASSIFIER_MODEL="${ARTS_DATA_DIR}/scripts/classifier/model_params.json"
TBSS_PREP_SCRIPT="${IIT_ATLAS_DIR}/tbss_4_prestats_iit"

for cmd in fslmaths fslstats fslinfo tbss_skeleton distancemap mri_synthmorph; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing command: ${cmd}" >&2
    exit 1
  fi
done

for f in "${IIT_FA}" "${T1_IN}" "${FLAIR_IN}" "${FA_IN}" "${SYNTHSEG_IN}" "${WMH_IN}"; do
  if [ ! -f "${f}" ]; then
    echo "Missing file: ${f}" >&2
    exit 1
  fi
done

for f in "${CALC_DTI_ROIS}" "${CLASSIFIER_PY}" "${CLASSIFIER_MODEL}" "${TBSS_PREP_SCRIPT}"; do
  if [ ! -f "${f}" ]; then
    echo "Missing file: ${f}" >&2
    exit 1
  fi
done

for f in "${IIT_ATLAS_DIR}/IITmean_FA_mask.nii.gz" "${IIT_ATLAS_DIR}/IITmean_FA_skeleton.nii.gz" "${IIT_ATLAS_DIR}/IITmean_lower_cingulum.nii.gz"; do
  if [ ! -f "${f}" ]; then
    echo "Missing IIT atlas file: ${f}" >&2
    exit 1
  fi
done

if [ ! -x "${CALC_DTI_ROIS}" ]; then
  chmod +x "${CALC_DTI_ROIS}"
fi
if [ ! -x "${TBSS_PREP_SCRIPT}" ]; then
  chmod +x "${TBSS_PREP_SCRIPT}"
fi

mkdir -p "${OUT_ROOT}"

rm -rf "${SUB_OUT}"
mkdir -p \
  "${SUB_OUT}/analysis" \
  "${SUB_OUT}/DTI" \
  "${SUB_OUT}/FLAIR" \
  "${SUB_OUT}/GMWM" \
  "${SUB_OUT}/T1" \
  "${SUB_OUT}/WMH" \
  "${SUB_OUT}/WMH_processing" \
  "${SUB_OUT}/FA_processing/tbss/stats" \
  "${SUB_OUT}/QC" \
  "${TABLE_DIR}"

export FSLOUTPUTTYPE=NIFTI_GZ

echo "${SUBJECT} ${AGE} ${SEX}" > "${SUB_OUT}/demo.txt"
echo "${SUBJECT} ${AGE} ${SEX} ${T1_IN} ${SUB_OUT}/GMWM/WM_mask.nii.gz ${SUB_OUT}/T1/coreg_to_flair.mat ${FLAIR_IN} ${WMH_IN} ${FA_IN}" > "${SUB_OUT}/biomarker_input.txt"

cat > "${TABLE_FILE}" <<EOF
subject,age,sex,T1w_brain,FLAIR_brain,FA,synthseg,WMH_mask,output,ARTS_data_dir,IITmean_FA
${SUBJECT},${AGE},${SEX},${T1_IN},${FLAIR_IN},${FA_IN},${SYNTHSEG_IN},${WMH_IN},${SUB_OUT},${ARTS_DATA_DIR},${IIT_FA}
EOF

echo "Preparing ARTS-style input folders"

fslmaths "${T1_IN}" "${SUB_OUT}/T1/T1_brain.nii.gz"
fslmaths "${FLAIR_IN}" "${SUB_OUT}/FLAIR/FLAIR_n4_reorient.nii.gz"
fslmaths "${FA_IN}" "${SUB_OUT}/DTI/FA.nii.gz"
fslmaths "${WMH_IN}" -thr 0.5 -bin "${SUB_OUT}/WMH/WMH_mask.nii.gz"

cat > "${SUB_OUT}/T1/coreg_to_flair.mat" <<EOF
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
EOF

echo "Extracting cerebral white matter mask from SynthSeg labels 2 and 41"

fslmaths "${SYNTHSEG_IN}" -thr 2 -uthr 2 -bin "${SUB_OUT}/GMWM/wm_left.nii.gz"
fslmaths "${SYNTHSEG_IN}" -thr 41 -uthr 41 -bin "${SUB_OUT}/GMWM/wm_right.nii.gz"
fslmaths "${SUB_OUT}/GMWM/wm_left.nii.gz" -add "${SUB_OUT}/GMWM/wm_right.nii.gz" -bin "${SUB_OUT}/GMWM/WM_mask.nii.gz"
rm -f "${SUB_OUT}/GMWM/wm_left.nii.gz" "${SUB_OUT}/GMWM/wm_right.nii.gz"

cp -f "${SYNTHSEG_IN}" "${SUB_OUT}/QC/synthseg.nii.gz"
cp -f "${SUB_OUT}/GMWM/WM_mask.nii.gz" "${SUB_OUT}/QC/WM_mask_from_synthseg.nii.gz"
cp -f "${SUB_OUT}/T1/coreg_to_flair.mat" "${SUB_OUT}/QC/coreg_to_flair_identity.mat"

echo "Calculating WMH feature"

fslmaths "${SUB_OUT}/GMWM/WM_mask.nii.gz" -thr 0.5 -bin "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz"
fslmaths "${SUB_OUT}/WMH/WMH_mask.nii.gz" -thr 0.5 -bin -mul "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz" "${SUB_OUT}/WMH_processing/WMH_no_cerebellum.nii.gz"

wmh_vol=$(fslstats "${SUB_OUT}/WMH_processing/WMH_no_cerebellum.nii.gz" -V | awk '{print $2}')
wm_vol=$(fslstats "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz" -V | awk '{print $2}')
wmh_ratio=$(awk -v a="${wmh_vol}" -v b="${wm_vol}" 'BEGIN{if (b > 0) print a / b; else print "nan"}')

echo "${wmh_ratio}" > "${SUB_OUT}/WMH_processing/features.txt"

cp -f "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz" "${SUB_OUT}/QC/WM_no_cerebellum.nii.gz"
cp -f "${SUB_OUT}/WMH_processing/WMH_no_cerebellum.nii.gz" "${SUB_OUT}/QC/WMH_no_cerebellum.nii.gz"

echo "WMH volume: ${wmh_vol}"
echo "WM volume: ${wm_vol}"
echo "WMH/WM feature: ${wmh_ratio}"

echo "Registering FA to IIT template with SynthMorph (CPU)"

mri_synthmorph \
  -o "${SUB_OUT}/FA_processing/tbss/stats/all_FA.nii.gz" \
  "${SUB_OUT}/DTI/FA.nii.gz" \
  "${IIT_FA}"

if [ ! -f "${SUB_OUT}/FA_processing/tbss/stats/all_FA.nii.gz" ]; then
  echo "Missing SynthMorph output: ${SUB_OUT}/FA_processing/tbss/stats/all_FA.nii.gz" >&2
  exit 1
fi

cp -f "${IIT_FA}" "${SUB_OUT}/QC/IITmean_FA.nii.gz"
cp -f "${SUB_OUT}/FA_processing/tbss/stats/all_FA.nii.gz" "${SUB_OUT}/QC/all_FA_synthmorph_to_IIT.nii.gz"

echo "Checking all_FA grid against IIT template"
echo "IITmean_FA:"
fslinfo "${IIT_FA}" | grep -E "dim[123]|pixdim[123]"
echo "all_FA:"
fslinfo "${SUB_OUT}/FA_processing/tbss/stats/all_FA.nii.gz" | grep -E "dim[123]|pixdim[123]"

echo "Running ARTS-style TBSS skeletonization and ROI extraction (native)"

TBSS_DIR="${SUB_OUT}/FA_processing/tbss"
cd "${TBSS_DIR}"

# Copy IIT atlas files into working TBSS directory
cp -f "${IIT_ATLAS_DIR}/IITmean_FA_mask.nii.gz" .
cp -f "${IIT_ATLAS_DIR}/IITmean_FA_skeleton.nii.gz" .
cp -f "${IIT_ATLAS_DIR}/IITmean_lower_cingulum.nii.gz" .
cp -f "${IIT_FA}" .

fslmaths stats/all_FA -max 0 -Tmin -bin stats/mean_FA_mask -odt char
fslmaths stats/all_FA -mas stats/mean_FA_mask stats/all_FA
fslmaths "${IIT_FA}" stats/mean_FA.nii.gz
fslmaths stats/mean_FA -bin stats/mean_FA_mask
fslmaths stats/all_FA -mas stats/mean_FA_mask stats/all_FA
fslmaths "${IIT_ATLAS_DIR}/IITmean_FA_skeleton.nii.gz" stats/mean_FA_skeleton

# Run the IIT-adapted TBSS prestats script (uses host FSL via ${FSLDIR}/bin/...)
bash "${TBSS_PREP_SCRIPT}" 0.25 > /dev/null

if [ ! -f "${TBSS_DIR}/stats/all_FA_skeletonised.nii.gz" ]; then
  echo "Missing all_FA_skeletonised.nii.gz" >&2
  exit 1
fi

# Run the DTI ROI extraction binary natively (standalone ELF, no MATLAB Runtime needed)
cd "${SUB_OUT}/FA_processing"
gunzip -f tbss/stats/all_FA_skeletonised.nii.gz
"${CALC_DTI_ROIS}" tbss/stats/all_FA_skeletonised.nii
gzip -f tbss/stats/all_FA_skeletonised.nii

if [ ! -f "${SUB_OUT}/FA_processing/features.txt" ]; then
  echo "Missing FA_processing/features.txt" >&2
  exit 1
fi

cat "${SUB_OUT}/FA_processing/features.txt"

cp -f "${SUB_OUT}/FA_processing/tbss/stats/all_FA_skeletonised.nii.gz" "${SUB_OUT}/QC/all_FA_skeletonised.nii.gz"
cp -f "${SUB_OUT}/FA_processing/features.txt" "${SUB_OUT}/QC/FA_features.txt"
cp -f "${SUB_OUT}/WMH_processing/features.txt" "${SUB_OUT}/QC/WMH_features.txt"

echo "Running ARTS classifier (native Python)"

echo "projid age sex wmh_total/wm_total ROI_artscler_1_FA ROI_artscler_2_FA ROI_artscler_3_FA ROI_artscler_4_FA" > "${SUB_OUT}/analysis/classifier_input.txt"
paste -d " " "${SUB_OUT}/demo.txt" "${SUB_OUT}/WMH_processing/features.txt" "${SUB_OUT}/FA_processing/features.txt" >> "${SUB_OUT}/analysis/classifier_input.txt"

cat "${SUB_OUT}/analysis/classifier_input.txt"

python3 "${CLASSIFIER_PY}" \
  "${SUB_OUT}/analysis/classifier_input.txt" \
  "${SUB_OUT}/analysis/score.csv" \
  "${CLASSIFIER_MODEL}"

if [ ! -f "${SUB_OUT}/analysis/score.csv" ]; then
  echo "Missing score.csv" >&2
  exit 1
fi

cp -f "${SUB_OUT}/analysis/classifier_input.txt" "${SUB_OUT}/QC/classifier_input.txt"

echo "Final ARTS score:"
cat "${SUB_OUT}/analysis/score.csv"

echo "Output folder: ${SUB_OUT}"
echo "QC folder: ${SUB_OUT}/QC"
echo "Input table: ${TABLE_FILE}"
