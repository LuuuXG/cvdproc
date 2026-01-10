#!/usr/bin/env bash
set -euo pipefail

############################################################
# Usage:
#   bash fix_fsaverage_symlink.sh <subjects_root> <correct_fsaverage_path>
#
# Example:
#   bash fix_fsaverage_symlink.sh \
#       /mnt/f/BIDS/SVD_BIDS/derivatives/freesurfer \
#       /usr/local/freesurfer/7-dev/subjects/fsaverage
#
############################################################

fix_fsaverage_symlink() {
    local subjects_root="$1"
    local correct_target="$2"

    if [ ! -d "$subjects_root" ]; then
        echo "ERROR: subjects_root directory not found: $subjects_root"
        exit 1
    fi

    if [ ! -d "$correct_target" ]; then
        echo "ERROR: fsaverage directory not found: $correct_target"
        exit 1
    fi

    echo "=== Fixing fsaverage symlinks in: $subjects_root ==="
    echo "Correct target: $correct_target"
    echo

    for subdir in "$subjects_root"/sub-*; do
        [ -d "$subdir" ] || continue  # skip non-directories

        local link_path="$subdir/fsaverage"

        # Remove existing symlink
        if [ -L "$link_path" ]; then
            echo "Removing old symlink in $subdir:"
            echo "  $link_path -> $(readlink "$link_path")"
            rm -f "$link_path"
        elif [ -e "$link_path" ]; then
            echo "WARNING: $link_path exists but is not a symlink. Removing."
            rm -rf "$link_path"
        fi

        # Create new symlink
        ln -s "$correct_target" "$link_path"
        echo "Created new symlink:"
        echo "  $link_path -> $correct_target"
        echo
    done

    echo "=== fsaverage symlink repair finished ==="
}

############################################################
# Main entry
############################################################

if [ "$#" -ne 2 ]; then
    echo "ERROR: Two arguments are required."
    echo "Usage: bash $0 <subjects_root> <correct_fsaverage_path>"
    exit 1
fi

fix_fsaverage_symlink "$1" "$2"
