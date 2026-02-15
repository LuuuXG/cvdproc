#!/bin/bash
set -euo pipefail

########################################
# Usage
########################################
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bids_dir> <subject_id> <subregion_ha> <subregion_thalamus> <subregion_brainstem> <subregion_hypothalamus>"
  echo "  subregion_* should be 0 or 1"
  echo "Example:"
  echo "  $0 /mnt/f/BIDS/WCH_SVD_3T_BIDS SSI0188 1 1 1 1"
  exit 1
fi

bids_dir="$1"       # BIDS root directory, e.g. /mnt/f/BIDS/WCH_SVD_3T_BIDS
subject_id="$2"     # Subject ID without "sub-", e.g. SSI0188

do_subregion_ha="$3"
do_subregion_thalamus="$4"
do_subregion_brainstem="$5"
do_subregion_hypothalamus="$6"

for v in "$do_subregion_ha" "$do_subregion_thalamus" "$do_subregion_brainstem" "$do_subregion_hypothalamus"; do
  if [[ "$v" != "0" && "$v" != "1" ]]; then
    echo "ERROR: subregion flags must be 0 or 1, got: $v"
    exit 1
  fi
done

#######################
# Prepare directories #
#######################
subject="sub-${subject_id}"
fs_root="${bids_dir}/derivatives/freesurfer"
subject_dir="${fs_root}/${subject}"

base_aseg="${subject_dir}/${subject}/mri/aseg.mgz"

if [[ ! -d "$subject_dir" ]]; then
  echo "ERROR: Subject FreeSurfer directory not found: ${subject_dir}"
  exit 2
fi

########################################
# Step 1: enumerate ses-* timepoints   #
# Only accept directory names WITHOUT '.' (exclude *.long.* etc.)
########################################
mapfile -t timepoints < <(
  find "$subject_dir" -maxdepth 1 -type d -name "ses-*" -printf "%f\n" \
    | grep -v '\.' \
    | sort
)

if [[ "${#timepoints[@]}" -lt 1 ]]; then
  echo "ERROR: No valid ses-* directories (without '.') found under: ${subject_dir}"
  echo "INFO: Directories found (raw):"
  find "$subject_dir" -maxdepth 1 -type d -name "ses-*" -printf "  - %f\n" | sort || true
  exit 3
fi

echo "INFO: Found ${#timepoints[@]} timepoints for ${subject}:"
for tp in "${timepoints[@]}"; do
  echo "  - ${tp}"
done

########################################
# Set SUBJECTS_DIR for recon-all runs  #
########################################
export SUBJECTS_DIR="$subject_dir"
echo "INFO: SUBJECTS_DIR=${SUBJECTS_DIR}"

#############################
# Step 2: run recon-all -base
#############################
if [[ -f "$base_aseg" ]]; then
  echo "INFO: Base already exists (aseg.mgz found): ${base_aseg}"
else
  cmd=(recon-all -base "$subject")
  for tp in "${timepoints[@]}"; do
    cmd+=(-tp "$tp")
  done
  cmd+=(-all -no-isrunning)

  echo "INFO: Running base command:"
  printf '  %q' "${cmd[@]}"
  echo

  "${cmd[@]}"

  echo "DONE: recon-all -base finished for ${subject}"
fi

###########################################################
# Step 3: run recon-all -long for each ses-* timepoint
###########################################################
echo "INFO: Starting longitudinal runs (-long) for ${subject}"

for tp in "${timepoints[@]}"; do
  long_subj="${tp}.long.${subject}"
  long_aseg="${SUBJECTS_DIR}/${long_subj}/mri/aseg.mgz"

  if [[ -f "$long_aseg" ]]; then
    echo "INFO: Long already exists for ${tp} (aseg.mgz found): ${long_aseg}"
    continue
  fi

  long_cmd=(recon-all -long "$tp" "$subject" -all -qcache)

  echo "INFO: Running long command for ${tp}:"
  printf '  %q' "${long_cmd[@]}"
  echo

  "${long_cmd[@]}"

  echo "DONE: recon-all -long finished for ${tp}"
done

###########################################################
# Helper: check outputs exist for all long timepoints
###########################################################
all_long_outputs_exist() {
  local relpath="$1"
  local tp=""
  local long_subj=""
  local f=""

  for tp in "${timepoints[@]}"; do
    long_subj="${tp}.long.${subject}"
    f="${SUBJECTS_DIR}/${long_subj}/${relpath}"
    if [[ ! -f "$f" ]]; then
      return 1
    fi
  done
  return 0
}

###########################################################
# Step 4: Optional subregion segmentations (ONLY for long)
# - thalamus / brainstem / hippo-amygdala: use segment_subregions --long-base
# - hypothalamus: loop each long subject with mri_segment_hypothalamic_subunits
###########################################################
if [[ "$do_subregion_ha" == "0" && "$do_subregion_thalamus" == "0" && "$do_subregion_brainstem" == "0" && "$do_subregion_hypothalamus" == "0" ]]; then
  echo "INFO: All subregion flags are 0; skipping subregion segmentations for ${subject}"
  echo "DONE: All base/long steps completed for ${subject}"
  exit 0
fi

echo "INFO: Starting optional subregion segmentations (ONLY for long timepoints) for ${subject}"

# 4.1 HA (hippo-amygdala) via segment_subregions
# Expected key file in each long subject:
# mri/lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz
if [[ "$do_subregion_ha" == "1" ]]; then
  if all_long_outputs_exist "mri/lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz" && all_long_outputs_exist "mri/rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz"; then
    echo "INFO: subregion_ha already exists for all long timepoints; skipping segment_subregions hippo-amygdala"
  else
    echo "INFO: Running subregion_ha (hippo-amygdala) via segment_subregions --long-base ${subject}"
    segment_subregions hippo-amygdala --long-base "${subject}" --sd "${SUBJECTS_DIR}" --threads 1
    echo "DONE: subregion_ha finished (segment_subregions hippo-amygdala)"
  fi
else
  echo "INFO: subregion_ha flag=0; skipped"
fi

# 4.2 Thalamus via segment_subregions
# Expected key file in each long subject:
# mri/thalamicNuclei.v13.FSvoxelSpace.mgz
if [[ "$do_subregion_thalamus" == "1" ]]; then
  if all_long_outputs_exist "mri/thalamicNuclei.v13.FSvoxelSpace.mgz"; then
    echo "INFO: subregion_thalamus already exists for all long timepoints; skipping segment_subregions thalamus"
  else
    echo "INFO: Running subregion_thalamus via segment_subregions --long-base ${subject}"
    segment_subregions thalamus --long-base "${subject}" --sd "${SUBJECTS_DIR}" --threads 1
    echo "DONE: subregion_thalamus finished (segment_subregions thalamus)"
  fi
else
  echo "INFO: subregion_thalamus flag=0; skipped"
fi

# 4.3 Brainstem via segment_subregions
# Expected key file in each long subject:
# mri/brainstemSsLabels.v13.FSvoxelSpace.mgz
if [[ "$do_subregion_brainstem" == "1" ]]; then
  if all_long_outputs_exist "mri/brainstemSsLabels.v13.FSvoxelSpace.mgz"; then
    echo "INFO: subregion_brainstem already exists for all long timepoints; skipping segment_subregions brainstem"
  else
    echo "INFO: Running subregion_brainstem via segment_subregions --long-base ${subject}"
    segment_subregions brainstem --long-base "${subject}" --sd "${SUBJECTS_DIR}" --threads 1
    echo "DONE: subregion_brainstem finished (segment_subregions brainstem)"
  fi
else
  echo "INFO: subregion_brainstem flag=0; skipped"
fi

# 4.4 Hypothalamus via mri_segment_hypothalamic_subunits (per long subject)
# Expected key file in each long subject:
# mri/hypothalamic_subunits_seg.v1.mgz
if [[ "$do_subregion_hypothalamus" == "1" ]]; then
  if all_long_outputs_exist "mri/hypothalamic_subunits_seg.v1.mgz"; then
    echo "INFO: subregion_hypothalamus already exists for all long timepoints; skipping mri_segment_hypothalamic_subunits"
  else
    echo "INFO: Running subregion_hypothalamus via mri_segment_hypothalamic_subunits (per long timepoint)"
    for tp in "${timepoints[@]}"; do
      long_subj="${tp}.long.${subject}"
      out_seg="${SUBJECTS_DIR}/${long_subj}/mri/hypothalamic_subunits_seg.v1.mgz"
      if [[ -f "$out_seg" ]]; then
        echo "INFO: Hypothalamus already exists for ${long_subj}; skipping"
        continue
      fi

      echo "INFO: Running hypothalamus for ${long_subj}"
      mri_segment_hypothalamic_subunits --s "${long_subj}" --sd "${SUBJECTS_DIR}" --threads 1
      echo "DONE: Hypothalamus finished for ${long_subj}"
    done
  fi
else
  echo "INFO: subregion_hypothalamus flag=0; skipped"
fi

echo "DONE: All base/long + requested subregion steps completed for ${subject}"
