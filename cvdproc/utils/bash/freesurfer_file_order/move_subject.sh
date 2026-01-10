#!/usr/bin/env bash
set -euo pipefail

#############################################
# Usage:
#   bash move_subject.sh <source_subject_dir> <destination_subject_dir>
#
# Example:
#   bash move_subject.sh \
#      /usr/local/freesurfer/7-dev/subjects/sub-SSI0008_lesionfilled \
#      /mnt/e/NewSubjects/sub-SSI0008_lesionfilled
#############################################

if [ "$#" -ne 2 ]; then
    echo "ERROR: Two arguments required."
    echo "Usage: bash $0 <source_subject_dir> <destination_subject_dir>"
    exit 1
fi

SRC_SUBJECT="$1"
DEST_SUBJECT="$2"

# Check source existence
if [ ! -d "$SRC_SUBJECT" ]; then
    echo "ERROR: Source subject directory does not exist:"
    echo "  $SRC_SUBJECT"
    exit 1
fi

# Ensure destination parent directory exists
DEST_PARENT="$(dirname "$DEST_SUBJECT")"
mkdir -p "$DEST_PARENT"

# Do not overwrite an existing subject folder
if [ -e "$DEST_SUBJECT" ]; then
    echo "ERROR: Destination directory already exists:"
    echo "  $DEST_SUBJECT"
    echo "Please remove or rename it first."
    exit 1
fi

echo "=== Moving FreeSurfer subject ==="
echo "FROM: $SRC_SUBJECT"
echo "TO  : $DEST_SUBJECT"
echo

mv "$SRC_SUBJECT" "$DEST_SUBJECT"

echo "Move completed successfully!"
echo "New subject location:"
echo "  $DEST_SUBJECT"
