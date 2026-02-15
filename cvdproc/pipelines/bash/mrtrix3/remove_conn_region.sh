#!/usr/bin/env bash
set -euo pipefail

wm_mask="$1"
tck_file="$2"

out_tract_mask="$3"
out_tdi_norm="$4"
out_wm_mask="$5"

# make a temp dir
tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

# ----------------------------
# Basic input checks
# ----------------------------
if [[ ! -f "$wm_mask" ]]; then
  echo "ERROR: wm_mask not found: $wm_mask" >&2
  exit 1
fi
if [[ ! -f "$tck_file" ]]; then
  echo "ERROR: tck_file not found: $tck_file" >&2
  exit 1
fi

for cmd in mrconvert tckmap tckinfo mrcalc mrthreshold mrstats; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command not found in PATH: $cmd" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$out_tract_mask")"
mkdir -p "$(dirname "$out_tdi_norm")"
mkdir -p "$(dirname "$out_wm_mask")"

# ----------------------------
# Convert WM mask to .mif (template)
# ----------------------------
wm_mask_mif="${tmpdir}/wm_mask.mif"
mrconvert "$wm_mask" "$wm_mask_mif" -quiet -force

# ----------------------------
# Streamline count
# ----------------------------
N="$(tckinfo "$tck_file" | awk '/count:/ {print $2; exit}')"
if [[ -z "${N}" ]]; then
  echo "ERROR: Failed to parse streamline count from tckinfo output." >&2
  exit 1
fi
if ! [[ "${N}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Parsed streamline count is not an integer: ${N}" >&2
  exit 1
fi
if [[ "${N}" -le 0 ]]; then
  echo "ERROR: Streamline count must be > 0, got: ${N}" >&2
  exit 1
fi

# ----------------------------
# Create tract density image in WM mask grid and restrict to WM
# ----------------------------
tdi_mif="${tmpdir}/tract_density.mif"
tdi_wm_mif="${tmpdir}/tract_density_wm.mif"

tckmap "$tck_file" "$tdi_mif" \
  -template "$wm_mask_mif" \
  -precise \
  -quiet \
  -force

mrcalc "$tdi_mif" "$wm_mask_mif" -mult "$tdi_wm_mif" -quiet -force

# ----------------------------
# Normalize by total number of streamlines
# ----------------------------
tdi_norm_mif="${tmpdir}/tract_density_norm.mif"
mrcalc "$tdi_wm_mif" "$N" -div "$tdi_norm_mif" -quiet -force
mrconvert "$tdi_norm_mif" "$out_tdi_norm" -quiet -force

# ----------------------------
# Threshold to get tract mask
# Default threshold: 2e-4
# Override via env var THR, e.g. THR=1e-4 ./script.sh ...
# ----------------------------
THR="${THR:-2e-4}"

if ! python - <<PY >/dev/null 2>&1
import sys
try:
    float("${THR}")
except Exception:
    sys.exit(1)
PY
then
  echo "ERROR: THR is not a valid number: ${THR}" >&2
  exit 1
fi

tract_mask_mif="${tmpdir}/tract_mask.mif"
mrthreshold "$tdi_norm_mif" "$tract_mask_mif" -abs "$THR" -quiet -force
mrconvert "$tract_mask_mif" "$out_tract_mask" -quiet -force

# ----------------------------
# Exclude tract mask from WM mask:
# out_wm_mask = wm_mask * (1 - tract_mask)
# ----------------------------
wm_excluding_mif="${tmpdir}/wm_excluding_tract.mif"
mrcalc "$wm_mask_mif" 1 "$tract_mask_mif" -subtract -mult "$wm_excluding_mif" -quiet -force
mrconvert "$wm_excluding_mif" "$out_wm_mask" -quiet -force

# ----------------------------
# QC summary
# ----------------------------
wm_count="$(mrstats "$wm_mask_mif" -output count -quiet -force)"
tract_count="$(mrstats "$tract_mask_mif" -output count -quiet -force)"
wm_excl_count="$(mrstats "$wm_excluding_mif" -output count -quiet -force)"

echo "Done: Streamlines (N): ${N}; Threshold (THR): ${THR}"