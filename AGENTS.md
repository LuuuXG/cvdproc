# CVDProc agent instructions

This file is the single source of repository guidance for coding agents. It consolidates the former AGENTS.md and CLAUDE.md. Apply these instructions to all future development in this repository.

## Working modes

- **Development mode:** reviewing or changing source code, interfaces, pipelines, tests, documentation, configuration templates, or packaging. Follow all development rules below. A review request authorizes inspection and reporting; implement fixes only when requested or otherwise included in the task.
- **Application mode:** using an existing CVDProc pipeline on a specified dataset, checking inputs, preparing run configuration, or extracting results. Follow the requested dataset, analysis settings, and output scope. Do not silently change source code or scientific definitions to make a run succeed. For DICOM conversion, read [the application guide](cvdproc/guides/application/dcm2bids.md). Dataset-level rules are supplied by [the application entry template](cvdproc/guides/application/dataset_agents.md); these are separate from repository development rules. Initialization installs copies in the dataset; `python -m cvdproc.guides --bids_dir <BIDS>` adds missing instructions to existing datasets without reinitializing them.
- Infer the mode from the requested work. When both are needed, state which part changes the software and which part runs an analysis. Do not assume that development validation authorizes a full dataset run.

Application procedures also include [pipeline execution](cvdproc/guides/application/pipelines.md) and [result extraction](cvdproc/guides/application/extract_results.md). Read the relevant guide and the shared workflow in [the dataset entry template](cvdproc/guides/application/dataset_agents.md) when applying CVDProc to a dataset. Maintain these packaged files as the source of application instructions.

## Language and coding style

- All content written into project files must be in English, including code comments, docstrings, documentation, configuration, and scripts. Translate existing Chinese text into English when editing the file containing it. Conversation may use the user's language.
- Preserve the surrounding coding style. Use compact, readable formatting and concise comments only where useful.
- Keep short or moderately long calls, constructors, trait definitions, dictionaries, and path expressions on one line when readable. Avoid aggressive Black-style wrapping. Split genuinely long or complex statements when needed for clarity.
- Do not reformat unrelated code, perform opportunistic refactors, or introduce implementation-only user configuration.

Preferred examples:

```python
subject_id = Str(mandatory=True, argstr="%s", position=0, desc="Subject ID without the sub- prefix.")
outputs["all_fa"] = os.path.join(output_dir, "FA_processing", "tbss", "stats", "all_FA.nii.gz")
```

## Project and runtime

CVDProc is a research neuroimaging package for cerebrovascular disease studies, including stroke, CSVD, atrial fibrillation, and community cohorts. It wraps FSL, FreeSurfer, ANTs, MATLAB, MRtrix3, and containers in Nipype workflows over BIDS-structured datasets.

- Python 3.10 is the development target. The current packaging metadata advertises `>=3.7`; do not assume that this establishes tested compatibility with older Python versions.
- The intended processing environment is Linux, with WSL2 / Ubuntu 22.04 documented by the project. Windows source inspection does not validate Linux tools or end-to-end workflows.
- External tools, MATLAB, Docker/Singularity, and required licenses must be configured separately. See `docs/containers.md` for containers.
- Optional dependency groups in `pyproject.toml` include `preprocess`, `quality`, `visualization`, and `analysis`. Heavy dependencies such as TensorFlow, PyTorch, and MONAI belong to optional preprocessing dependencies. Preserve lazy pipeline imports in the manager.

## Architecture and ownership

- `cvdproc/main.py`: CLI argument handling, YAML configuration, subject/session iteration, workflow execution, and population extraction.
- `cvdproc/bids_data/`: `BIDSSubject`, `BIDSSession`, file discovery, and BIDS filename helpers. Sessions discover `anat`, `dwi`, `func`, `fmap`, `perf`, and project extensions `swi`, `qsm`, `pwi`, plus derivative directories such as `freesurfer`, `lesion_mask`, `fsl_anat`, and `xfm`.
- `cvdproc/controllers/pipeline_manager.py`: central lazy `if/elif` registry mapping pipeline names to classes. Register new pipelines here.
- `cvdproc/pipelines/`: `smri`, `dmri`, `perfusion`, `qmri`, `pwi`, and `multi` pipelines; shared interfaces in `common`; DICOM conversion in `dcm2bids`; external command scripts in `bash`, `matlab`, and `r`. Follow the actual neighboring modules when extending an area.
- `cvdproc/config/paths.py`: package resource resolution, fsaverage / medial-wall paths, and FreeSurfer qcache metric-pair discovery. `logger_config.py` provides `setup_logger()`.
- `cvdproc/utils/tool_manager.py`: the `cvdproc_tool` dispatcher for utility scripts in `utils/python`, `utils/bash`, and `utils/matlab`.
- `cvdproc/pipelines/external/`: bundled external implementations. Prefer changing project-owned integration code; modify bundled implementations only when the task requires it.
- `cvdproc/data/`: separately downloaded atlases, model weights, templates, toolboxes, and configurations; see README for download instructions. `cvdproc/trash/` contains archived code. Both are git-ignored.

## Pipeline and interface design

- Keep pipeline classes centered on `__init__`, `check_data_requirements`, `create_workflow`, and `extract_results` where supported. Constructors normally accept `(subject, session, output_path, **config_kwargs)`; respect existing subject-level exceptions such as longitudinal processing.
- `create_workflow()` returns a Nipype `Workflow`; execution belongs to the caller. Keep one-off discovery, input selection, and output naming in `create_workflow`, following neighboring pipelines, rather than introducing many trivial private methods.
- Ensure required validation actually runs before processing. A `check_data_requirements` method that is not called does not validate a workflow.
- For pipelines supporting population extraction, allow construction with `subject=None` and `session=None`. Keep result-source paths distinct from summary-output paths and retain subject/session and analysis identity in aggregated results.
- Search existing project and Nipype interfaces before implementing a wrapper or helper. Check `pipelines/common/` and the relevant modality first. Reuse `MRIConvertApplyWarp`, Nipype `FLIRT`, and suitable existing statistics/file-operation interfaces; do not duplicate their command construction inside analysis interfaces.
- Consolidate interfaces for one analysis in one `<analysis>_nipype.py` module. Share common operations; retain helpers only when reuse or a distinct scientific algorithm warrants them.
- Custom interfaces use Nipype input/output specs and the appropriate `BaseInterface` or `CommandLine` lifecycle. Keep declared traits, workflow connections, generated filenames, and `_list_outputs()` consistent. Use `isdefined` where Nipype optional traits require it.
- Use `IdentityInterface`, `DataSink`, and `MapNode` where appropriate to the neighboring workflow and iteration needs. They are existing patterns, not requirements to add unnecessary nodes.

## Scientific correctness, inputs, and outputs

- Preserve scientific definitions when reusing an interface. A generic voxel mean cannot replace a ratio of summed streamline weights. Keep units, numerator/denominator definitions, zero handling, and aggregation order explicit.
- Check shape and spatial geometry before voxelwise combinations. Make reference space, transform direction, coordinate convention, interpolation, and atlas-label identity explicit. Use label-preserving interpolation for masks and discrete labels.
- Do not infer acquisition identity from image dimensions alone. Match images and sidecars using acquisition/BIDS entities or explicit provenance. Reject ambiguous selections instead of silently taking the first match.
- Validate configuration and parallel subject/session/input lists before starting a batch. Do not silently truncate a batch with `zip` over unequal lists. Handle missing and subject-level sessions explicitly.
- Preserve the project's session-oriented BIDS conventions and existing non-standard modalities. Do not silently generalize them to arbitrary BIDS layouts. The README recommends starting from DICOM for reproducibility.
- Reuse BIDS naming helpers and package path helpers. Child path components must not unexpectedly become absolute paths and discard the intended parent directory.
- Keep outputs separated by subject, session, and analysis. Pass explicit working directories and output paths to external interfaces. Check external command failures before cleanup, postprocessing, or reporting success.
- Keep original research inputs intact unless the requested operation explicitly entails conversion, renaming, or replacement. Do not use real datasets for disposable development checks.

## Development workflow and validation

1. Read this file and relevant local instructions; inspect the working-tree status and preserve existing user changes.
2. Inspect the entry point, neighboring pipeline, relevant interfaces, configuration, and documentation before editing. Search only the necessary source directories.
3. Implement the smallest coherent change, reusing existing abstractions and preserving scientific behavior. Update registration, configuration documentation, and output consumers when affected.
4. Validate at the level affected: syntax/import checks, interface trait and output checks, workflow construction, or small numerical/regression tests. Use synthetic inputs and follow the functional-test working-directory rule below. Do not blindly execute script collections or launch external processing as a smoke test.
5. Report findings or changes with file locations, verification performed, and any runtime/dependency limits. Distinguish static inspection, mocked checks, and actual tool execution.

- Do not recursively scan `cvdproc/data/`. Exclude `data`, `trash`, caches, generated `site/`, and bundled external assets from broad source searches. Inspect a specific resource only when the task needs it.
- In development mode, run functional tests in a dedicated temporary directory by default. If the user specifies a working directory for the task or tests, use that directory instead. Set the test process working directory and any interface/workflow working and output directories explicitly so generated files stay within the selected location.
- Keep disposable validation scripts, generated sidecars, workflow outputs, and crash files in the selected test working directory, not in the repository root unless the user explicitly selects it. Preserve existing files in user-specified directories; do not delete the directory as temporary cleanup. Durable tests belong in an appropriate test location.
- There is no established comprehensive test suite. `pipelines/nipype_test/` contains workflow examples and focused tests, including `test_fractional_length.py`. Inspect a test's dependencies and side effects before running it; do not describe all files there as pure unit tests.
- Run meaningful checks appropriate to the change. Documentation-only edits normally need content/link/diff checks, not new implementation-mirroring tests. Do not claim end-to-end verification without the required tools and inputs.
- Keep configuration choices meaningful to the user. Avoid adding flags solely to expose an implementation detail.

## Common commands

Run commands in an appropriately configured development environment; these examples do not authorize installing dependencies or running a dataset as part of every task.

```bash
pip install -e ".[preprocess,quality,visualization,analysis]"
mkdocs serve
mkdocs build
cvdproc --help
cvdproc_tool list
cvdproc_tool python/some_script.py [args...]
python -B -m unittest cvdproc.pipelines.nipype_test.test_fractional_length
```

Pipelines use YAML configuration:

```bash
cvdproc --config_file config.yaml --run_pipeline --pipeline wmh_quantification \
        --subject_id 001 002 --session_id 01 02
```

Key actions: `--run_initialization` scaffolds a BIDS directory; `--run_dcm2bids` converts DICOM using the `dcm2bids` configuration section; `--run_pipeline --pipeline <name>` executes a workflow; `--extract_results --pipeline <name>` aggregates results; `--check_data` checks configured data presence. Consult the current CLI and relevant pipeline documentation for required arguments.
