#!/usr/bin/env bash
set -e

# Default values
t1=""
mni_template=""
t1_mni_out=""
brain_mask_out=""
t1_2_mni_warp=""
mni_2_t1_warp=""
t1_stripped=""
register_between_stripped=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t1)
      t1="$2"
      shift; shift
      ;;
    -mni_template)
      mni_template="$2"
      shift; shift
      ;;
    -t1_mni_out)
      t1_mni_out="$2"
      shift; shift
      ;;
    -brain_mask_out)
      brain_mask_out="$2"
      shift; shift
      ;;
    -t1_2_mni_warp)
      t1_2_mni_warp="$2"
      shift; shift
      ;;
    -mni_2_t1_warp)
      mni_2_t1_warp="$2"
      shift; shift
      ;;
    -t1_stripped)
      t1_stripped="$2"
      shift; shift
      ;;
    -register_between_stripped)
      register_between_stripped=true
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# sanity check
if [[ -z "$t1" || -z "$mni_template" || -z "$t1_mni_out" || -z "$t1_2_mni_warp" || -z "$mni_2_t1_warp" ]]; then
  echo "Usage: $0 -t1 <T1w> -mni_template <MNI> -t1_mni_out <output.nii.gz> -t1_2_mni_warp <warp.nii.gz> -mni_2_t1_warp <warp.nii.gz> [options]"
  exit 1
fi

# temp directory
temp_dir=$(dirname "$t1_mni_out")/synthmorph_temp
mkdir -p "$temp_dir"

# Determine which T1 to use
if [[ -n "$t1_stripped" ]]; then
  t1_input="$t1_stripped"
elif [[ "$register_between_stripped" == true ]]; then
  t1_input="$temp_dir/t1w_brain.nii.gz"
  if [[ -z "$brain_mask_out" ]]; then
    brain_mask_out="$temp_dir/t1w_brain_mask.nii.gz"
  fi
  echo "Running SynthStrip on $t1 ..."
  mri_synthstrip -i "$t1" -o "$t1_input" -m "$brain_mask_out"
else
  t1_input="$t1"
fi

# Run registration
echo "Running SynthMorph registration..."
mri_synthmorph -o "$t1_mni_out" \
  -t "$t1_2_mni_warp" \
  -T "$mni_2_t1_warp" \
  "$t1_input" \
  "$mni_template" \
  -g

rm -rf "$temp_dir"

echo "Registration completed!"
