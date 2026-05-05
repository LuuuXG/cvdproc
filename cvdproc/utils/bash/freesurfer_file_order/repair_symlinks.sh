#!/usr/bin/env bash
set -euo pipefail

############################################################
# Usage:
#   bash repair_freesurfer_links.sh <bids_root>
#
# Example:
#   bash repair_freesurfer_links.sh /mnt/f/BIDS/WCH_AF_Project
############################################################

if [ "$#" -ne 1 ]; then
    echo "ERROR: One argument required."
    echo "Usage: bash $0 <bids_root>"
    exit 1
fi

BIDS_ROOT="$1"
FS_ROOT="${BIDS_ROOT}/derivatives/freesurfer"

if [ ! -d "$BIDS_ROOT" ]; then
    echo "ERROR: BIDS root does not exist:"
    echo "  $BIDS_ROOT"
    exit 1
fi

if [ ! -d "$FS_ROOT" ]; then
    echo "ERROR: FreeSurfer derivatives directory does not exist:"
    echo "  $FS_ROOT"
    exit 1
fi

detect_fsaverage() {
    local candidates=()

    if [ -n "${SUBJECTS_DIR:-}" ]; then
        candidates+=("${SUBJECTS_DIR}/fsaverage")
    fi

    if [ -n "${FREESURFER_HOME:-}" ]; then
        candidates+=("${FREESURFER_HOME}/subjects/fsaverage")
    fi

    candidates+=(
        "/usr/local/freesurfer/7-dev/subjects/fsaverage"
        "/usr/local/freesurfer/subjects/fsaverage"
        "/opt/freesurfer/subjects/fsaverage"
    )

    local p
    for p in "${candidates[@]}"; do
        if [ -d "$p" ]; then
            echo "$p"
            return 0
        fi
    done

    return 1
}

repair_fsaverage_symlink() {
    local sub_root="$1"
    local correct_target="$2"
    local link_path="${sub_root}/fsaverage"

    echo "=== Repairing fsaverage symlink in: $sub_root ==="

    if [ -L "$link_path" ]; then
        echo "  Removing old symlink: $link_path -> $(readlink "$link_path")"
        rm -f "$link_path"
    elif [ -e "$link_path" ]; then
        echo "  WARNING: $link_path exists but is not a symlink. Removing."
        rm -rf "$link_path"
    fi

    ln -s "$correct_target" "$link_path"
    echo "  Created: $link_path -> $correct_target"
    echo
}

repair_surf_symlinks() {
    local subject_dir="$1"
    local surf_dir="${subject_dir}/surf"

    if [ ! -d "$surf_dir" ]; then
        return 0
    fi

    echo "=== Repairing surf symlinks in: $surf_dir ==="

    declare -A LINKS
    LINKS["lh.fsaverage.sphere.reg"]="lh.sphere.reg"
    LINKS["lh.pial"]="lh.pial.T1"
    LINKS["lh.white.H"]="lh.white.preaparc.H"
    LINKS["lh.white.K"]="lh.white.preaparc.K"
    LINKS["rh.fsaverage.sphere.reg"]="rh.sphere.reg"
    LINKS["rh.pial"]="rh.pial.T1"
    LINKS["rh.white.H"]="rh.white.preaparc.H"
    LINKS["rh.white.K"]="rh.white.preaparc.K"

    local link_name
    local target_name
    local link_path

    for link_name in "${!LINKS[@]}"; do
        target_name="${LINKS[$link_name]}"
        link_path="${surf_dir}/${link_name}"

        echo "  Processing: $link_name -> $target_name"

        if [ -L "$link_path" ] || [ -e "$link_path" ]; then
            rm -f "$link_path"
        fi

        (
            cd "$surf_dir"
            ln -s "$target_name" "$link_name"
        )

        if [ ! -e "${surf_dir}/${target_name}" ]; then
            echo "    WARNING: target does not exist: ${surf_dir}/${target_name}"
        else
            echo "    OK"
        fi
    done

    echo
}

CORRECT_FSAVERAGE="$(detect_fsaverage || true)"

if [ -z "${CORRECT_FSAVERAGE:-}" ]; then
    echo "ERROR: Could not auto-detect fsaverage."
    echo "Checked SUBJECTS_DIR, FREESURFER_HOME, and common install paths."
    exit 1
fi

echo "BIDS root: $BIDS_ROOT"
echo "FreeSurfer root: $FS_ROOT"
echo "Detected fsaverage: $CORRECT_FSAVERAGE"
echo

for sub_root in "${FS_ROOT}"/sub-*; do
    [ -d "$sub_root" ] || continue

    echo "############################################################"
    echo "Processing subject root: $sub_root"
    echo "############################################################"

    repair_fsaverage_symlink "$sub_root" "$CORRECT_FSAVERAGE"

    for child_dir in "$sub_root"/*; do
        [ -e "$child_dir" ] || continue

        # fsaverage is a template symlink, not a real subject folder
        if [ "$(basename "$child_dir")" = "fsaverage" ]; then
            continue
        fi

        # skip symlink entries
        if [ -L "$child_dir" ]; then
            continue
        fi

        [ -d "$child_dir" ] || continue

        if [ -d "${child_dir}/surf" ]; then
            repair_surf_symlinks "$child_dir"
        fi
    done
done

echo "=== All repairs completed successfully ==="