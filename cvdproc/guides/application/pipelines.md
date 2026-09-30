# Running CVDProc analysis pipelines

Follow the dataset AGENTS.md shared application workflow. This guide defines the common analysis procedure; consult the selected implementation for modality-specific requirements.

## 1. Translate the analysis goal into a supported workflow

Ask what the user wants to measure or produce, which visits to include, and what preprocessing or analysis outputs already exist. Reuse known answers. Explain the suitable registered pipeline and its expected outputs; if several methods answer the question, describe the differences relevant to the study and ask for the unresolved choice. Do not choose a scientific method solely because its dependencies are installed.

Use the installed `cvdproc/controllers/pipeline_manager.py` to resolve the actual pipeline name and class, then inspect its constructor, input selection, `check_data_requirements`, and `create_workflow`. Read relevant pipeline documentation and interfaces. A validation method may exist without being called; verify requirements directly before execution. If the requested method is unsupported, state the limitation rather than modifying source code in application mode.

## 2. Check inputs, dependencies, and resources

Prepare an explicit subject/session list and inspect the required modalities and prior derivatives for those visits. Resolve ambiguous input selections using acquisition entities and provenance. Check reference spaces, transforms, masks, atlases, and units when relevant; do not substitute a similarly named file or infer identity from dimensions.

Check only dependencies needed by the selected method: Python packages, command-line tools, MATLAB/toolboxes or runtime, container engines/images, model files, atlases, licenses, and configured paths. Report each missing or unverified prerequisite and the affected step. Use read-only availability/version checks rather than launching full processing as a dependency test. Offer setup help when needed; respect existing installation authorization and user environment choices.

Check CPU, memory, GPU, and disk requirements against the planned concurrency. Availability of a Python import does not establish that an external tool or licensed workflow can run.

## 3. Prepare configuration and execution scope

Read the existing study YAML and the relevant generated template. Confirm option names, accepted values, defaults, and interactions against the selected implementation, including how `**kwargs` are consumed. Present consequential choices such as segmentation method, image selection, reference space, resolution, atlas, and processing stages with their purpose. Ask only about choices that remain unresolved.

Set absolute `bids_dir` and `output_dir` values and place the supported options under `pipelines.<registered_name>`. Keep a configuration snapshot. The current CLI requires a nonempty configuration entry for the selected pipeline; if an empty entry is rejected, determine a supported explicit option rather than inventing a dummy setting.

The standard runner passes `<output_dir>/<pipeline>/sub-<id>/ses-<id>` to the manager, but particular pipelines can redirect outputs. Verify actual paths in the implementation. Workflow working directories are under `<BIDS>/derivatives/workflows/sub-<id>/ses-<id>`. The agent's auxiliary working directory is separate from these tool-managed directories.

## 4. Run a representative visit, then the requested cohort

Use the selected registered name in place of `PIPELINE_NAME`:

```bash
cvdproc --config_file /path/to/MyStudy/code/config.yml --run_pipeline \
  --pipeline PIPELINE_NAME --subject_id 001 --session_id baseline --n_jobs 1
```

Inspect the representative outputs before expanding a new configuration. Do not repeat a representative run that has already been validated with the same relevant settings. For a batch, provide equally sized subject/session lists in paired order; repeat shared session labels explicitly. Both IDs are required by the current CLI.

`--n_jobs` defaults to 1 and controls concurrent subject/session processes, not Nipype node parallelism or each tool's internal threads. Budget all levels together. Reject duplicate visit destinations; longitudinal FreeSurfer parallel runs require distinct subjects. Do not launch separate overlapping commands that share working/output directories.

Serial execution stops on the first failure. Parallel execution allows other submitted visits to finish and reports failures at the end. Check logs and crash files before retrying failed visits; do not clear successful work or assume cached outputs match changed settings.

## 5. Validate and hand off

Check expected files, readability, dimensions, geometry, labels, and scientific plausibility as appropriate to the analysis. Distinguish successful execution from quality approval. Record the configuration, relevant versions, visit status, and unresolved findings. If the user also requested tabular or population outputs, proceed with [the extraction guide](extract_results.md), checking that this pipeline implements extraction and that the requested source outputs are ready.
