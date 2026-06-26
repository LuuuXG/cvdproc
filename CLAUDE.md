# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

CVDProc is a research neuroimaging processing package for cerebrovascular disease studies (stroke, CSVD, atrial fibrillation, community cohorts). It wraps external tools (FSL, FreeSurfer, ANTs, MATLAB, MRtrix3, Docker) into Nipype workflows, operating on BIDS-structured datasets.

**Python 3.10** is the target version. The codebase runs on Linux (tested on WSL2/Ubuntu 22.04).

## Common Commands

```bash
# Install in editable mode with all optional deps
pip install -e ".[preprocess,quality,visualization,analysis]"

# Build docs (MkDocs Material)
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build            # static site into site/

# CLI entry points (registered in pyproject.toml [project.scripts])
cvdproc --help                                   # main pipeline runner
cvdproc_tool list                                # list available utility scripts
cvdproc_tool python/some_script.py [args...]     # run a utility by relative path
```

### Running a pipeline

Pipelines are driven by a YAML config file and run through `cvdproc`:

```bash
cvdproc --config_file config.yaml --run_pipeline --pipeline wmh_quantification \
        --subject_id 001 002 --session_id 01 02
```

Key `cvdproc` flags:
- `--run_initialization` — scaffold a BIDS directory
- `--run_dcm2bids` — convert DICOM → BIDS (requires `dcm2bids` config section)
- `--run_pipeline --pipeline <name>` — execute a named pipeline
- `--extract_results --pipeline <name>` — aggregate population-level results
- `--check_data` — verify BIDS data presence

## Architecture

### BIDS data layer (`cvdproc/bids_data/`)

`BIDSSubject` and `BIDSSession` model the BIDS directory tree. `BIDSSession` auto-discovers files per modality (`anat`, `dwi`, `func`, `fmap`, `perf`, plus non-standard `swi`, `qsm`, `pwi`) and derivative directories (`freesurfer`, `lesion_mask`, `fsl_anat`, `xfm`). These objects are passed into every pipeline constructor.

### Pipeline system (`cvdproc/pipelines/`)

All pipelines follow the same contract — a constructor receiving `(subject, session, output_path, **config_kwargs)` and implementing:
- `create_workflow()` → returns a `nipype.Workflow` object
- `extract_results()` (optional mixin) → aggregate outputs across subjects

`PipelineManager` (`cvdproc/controllers/pipeline_manager.py`) is the central registry — a single `if/elif` chain mapping pipeline name strings to their classes. When adding a new pipeline, register it there.

Pipelines are organized by modality:
- `smri/` — structural MRI (FreeSurfer, CAT12, FSL_ANAT, lesion analysis, brain age, HIPSTA, SCN, anatomical segmentation, CSVD markers like WMH/PVS/CMB)
- `dmri/` — diffusion MRI (full DWI preprocessing/analysis, NEMO postprocessing, LQT)
- `perfusion/` — ASL processing
- `qmri/` — QSM pipelines
- `pwi/` — DSC-MRI perfusion
- `fmri/` — functional MRI
- `dcm2bids/` — DICOM→BIDS conversion
- `common/` — reusable Nipype custom interfaces (image calc, file ops, registration utilities)
- `bash/`, `matlab/`, `r/` — shell/MATLAB/R scripts invoked as external tools by pipelines

Many pipelines wrap Docker/Singularity containers; see `docs/containers.md` for the image list.

### Nipype patterns

Custom Nipype interfaces in `pipelines/common/` extend `BaseInterface` (define `input_spec`/`output_spec`, implement `_run_interface`/`_list_outputs`). Workflows use `IdentityInterface` for inputs, `DataSink` for outputs, and `MapNode` for iteration over subjects/sessions.

### Config (`cvdproc/config/`)

- `paths.py` — package root resolution and shared path constants (fsaverage directory, medial wall labels, FreeSurfer qcache metric pair discovery)
- `logger_config.py` — single `setup_logger()` function returning a configured `logging.Logger`

### Utilities (`cvdproc/utils/`)

`cvdproc_tool` (`tool_manager.py`) is a CLI dispatcher that runs scripts from `utils/python/`, `utils/bash/`, or `utils/matlab/` by relative path. Each subdirectory contains standalone scripts for one-off tasks: DICOM sorting, Excel merging, surface file conversion, atlas comparison, etc.

### External data (`cvdproc/data/`)

Excluded from the repo via `.gitignore`. Contains atlas files, model weights, DSI Studio configs, MATLAB toolboxes, BIDS config templates. Must be downloaded separately (Google Drive link in README).

## Key constraints

- **No test suite exists.** There are ad-hoc test scripts under `pipelines/nipype_test/` that serve as Nipype workflow smoke tests, but no pytest/unittest framework.
- `cvdproc/data/` and `cvdproc/trash/` are git-ignored. `trash/` contains deprecated/archived code.
- BIDS handling is non-standard (requires session level, uses non-standard suffixes like `swi`, `qsm`, `pwi`). The README recommends starting from DICOM for reproducibility.
- Many pipelines depend on external tools (MATLAB, Docker, FreeSurfer license) that must be pre-installed and on `PATH`.
- Optional dependency groups in `pyproject.toml` isolate large packages (TensorFlow, PyTorch, MONAI) behind the `[preprocess]` extra.
