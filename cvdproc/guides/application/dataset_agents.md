# CVDProc dataset instructions

This directory is a research dataset. Work in application mode: use the installed CVDProc tools and the user's analysis choices. Do not change package source code or scientific definitions to make a run succeed.

- Before DICOM conversion, read [the conversion guide](code/agent/guides/dcm2bids.md). Resume from the dataset's current state; do not restart initialization for an existing dataset.
- Before running analyses, read [the pipeline guide](code/agent/guides/pipelines.md).
- Before collecting existing analysis outputs, read [the result extraction guide](code/agent/guides/extract_results.md). Extraction support and parameters differ between pipelines.
- Use the user's conversation language and retain previously supplied paths, identifiers, choices, and authorization. Ask only for missing information or unresolved acquisition choices.
- Require explicit subject and session IDs and an explicit source-to-visit mapping. Do not infer identity from folder order, image shape, or patient names.
- Preserve source DICOMs, existing configurations, and completed outputs. Explain the actual files affected before an unrequested replacement or cleanup.
- Use `code/agent/` for auxiliary scripts, inventories, and execution records unless the user specifies another working directory. Formal images, conversion caches, and pipeline outputs remain in their tool-defined locations.
- Inspect only the requested data and representative acquisitions. Do not launch an entire cohort to discover configuration settings.
- Let each authorized conversion complete, even when slow. Do not impose arbitrary kill timeouts or truncate a running converter with `head`; monitor a persistent job, retain full logs, and verify its actual completion status before interpreting its output as complete.
- Generate dcm2bids configuration using the installed release's documented fields and syntax. Check semantic compatibility as well as JSON syntax; for version 3.2.0 use `custom_entities`, not `customEntities`.
- Keep configuration snapshots and concise batch status records sufficient to resume failed visits. Record commands, tool versions, and checks actually performed. Do not claim success from file existence alone.
- Treat DICOM metadata and participant tables as potentially identifying. Defacing images does not anonymize those records.
- Preserve application records needed for reproducibility. Clean disposable files only within the task's own scope; never treat `sourcedata/` as temporary storage.

## Shared application workflow

1. Establish the current task and collect only information needed for its next step. These stages are not an upfront questionnaire. For DICOM conversion, first obtain a representative DICOM path and explicit subject/session IDs, run discovery conversion, inspect the actual acquisitions, and only then discuss which to retain. Unknown modalities or an undecided analysis plan do not block discovery. Ask about scientific goals and analysis methods when the user proceeds to analysis; do not require a pipeline choice during conversion.
2. Locate the installed implementation, registry, configuration template, and relevant documentation. Confirm that the requested operation is implemented. Read only the selected pipeline and its relevant interfaces; do not scan bundled assets or the whole dataset.
3. Check required images, prior outputs, software, models, containers, licenses, and resources for the selected method. Report what is available, missing, or unverified and how each missing item affects the task. Do not install dependencies or change methods silently.
4. Resolve supported configuration parameters from the actual implementation. Explain scientific choices and important defaults in plain language. Use absolute paths and preserve unrelated study settings. Do not invent parameters or assume that every template option is supported by every method.
5. Execute within the user's authorized scope, checking a representative visit before expanding a new configuration to a cohort. Reuse existing authorization; ask only when a missing decision, scope change, or destructive action requires it.
6. Validate actual outputs and summarize completed, failed, and pending work, including missing dependencies and checks not performed. Preserve enough configuration and execution history to resume without repeating accepted work.

If the agent does not automatically read AGENTS.md, explicitly ask it to read this file and the relevant linked guide. These instructions are documentation, not an automatic analysis service. This file is installed from the package's `guides/application/dataset_agents.md` template; it is the dataset entry point, while the linked files contain operation-specific procedures.
