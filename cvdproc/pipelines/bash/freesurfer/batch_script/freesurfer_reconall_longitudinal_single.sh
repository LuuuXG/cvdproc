#!/bin/bash
set -euo pipefail

bids_dir="$1"       # BIDS root directory, e.g. /mnt/f/BIDS/WCH_SVD_3T_BIDS
subject_id="$2"     # Subject ID without "sub-", e.g. SSI0188

#######################
# Prepare directories #
#######################
subject="sub-${subject_id}"
fs_root="${bids_dir}/derivatives/freesurfer"
subject_dir="${fs_root}/${subject}"

# Base output path expected by you:
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
# Example:
# recon-all -long ses-baseline sub-SSI0008 -all -qcache
###########################################################
echo "INFO: Starting longitudinal runs (-long) for ${subject}"

for tp in "${timepoints[@]}"; do
  # FreeSurfer long subject name convention:
  # <timepoint>.long.<base>
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

echo "DONE: All base/long steps completed for ${subject}"
