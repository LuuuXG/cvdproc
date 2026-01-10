#!/usr/bin/env bash
set -euo pipefail

#############################################
# Usage:
#   bash repair_symlinks.sh <subject_dir>
#
# Example:
#   bash repair_symlinks.sh /mnt/e/NewSubjects/sub-SSI0008_lesionfilled
#
# The script will fix symlinks in:
#   <subject_dir>/surf
#############################################

if [ "$#" -ne 1 ]; then
    echo "ERROR: One argument required."
    echo "Usage: bash $0 <subject_dir>"
    exit 1
fi

SUBJECT_DIR="$1"
SURF_DIR="${SUBJECT_DIR}/surf"

# Check directories
if [ ! -d "$SUBJECT_DIR" ]; then
    echo "ERROR: Subject directory does not exist:"
    echo "  $SUBJECT_DIR"
    exit 1
fi

if [ ! -d "$SURF_DIR" ]; then
    echo "ERROR: surf directory does not exist:"
    echo "  $SURF_DIR"
    exit 1
fi

echo "=== Repairing FreeSurfer symlinks in: $SURF_DIR ==="
echo

# Declare symlink mapping
declare -A LINKS
LINKS["lh.fsaverage.sphere.reg"]="lh.sphere.reg"
LINKS["lh.pial"]="lh.pial.T1"
LINKS["lh.white.H"]="lh.white.preaparc.H"
LINKS["lh.white.K"]="lh.white.preaparc.K"
LINKS["rh.fsaverage.sphere.reg"]="rh.sphere.reg"
LINKS["rh.pial"]="rh.pial.T1"
LINKS["rh.white.H"]="rh.white.preaparc.H"
LINKS["rh.white.K"]="rh.white.preaparc.K"

for link_name in "${!LINKS[@]}"; do
    target_name="${LINKS[$link_name]}"
    link_path="${SURF_DIR}/${link_name}"
    target_path="${target_name}"  # relative path inside surf/

    echo "Processing: $link_name -> $target_name"

    # Remove existing file / symlink
    if [ -L "$link_path" ] || [ -e "$link_path" ]; then
        echo "  Removing existing: $link_path"
        rm -f "$link_path"
    fi

    # Create new correct symlink
    (
        cd "$SURF_DIR"
        ln -s "$target_path" "$link_name"
    )
    echo "  Created symlink: $link_name -> $target_name"

    # Warning if target doesn't exist
    if [ ! -e "${SURF_DIR}/${target_name}" ]; then
        echo "  WARNING: target does not exist: ${SURF_DIR}/${target_name}"
    fi

    echo
done

echo "=== Symlink repair completed successfully! ==="
