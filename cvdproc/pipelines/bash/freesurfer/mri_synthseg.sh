#!/usr/bin/env bash
[ ! -e "$FREESURFER_HOME" ] && echo "error: freesurfer has not been properly sourced" && exit 1
exec python3 $FREESURFER_HOME/python/scripts/mri_synthseg "$@"
