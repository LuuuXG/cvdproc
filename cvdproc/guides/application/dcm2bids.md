# DICOM to BIDS application guide

Use this guide when assisting a user with the installed CVDProc converter. It covers initialization, acquisition selection, one-visit validation, revisions, and batch conversion. It does not authorize software changes or a full dataset run beyond the user's request.

## 1. Establish the task and environment

Use the current conversation language. Reuse information already supplied. Obtain the desired dataset root and the scope of the next step. The user need not know which modalities are present or which sequences to retain. Do not ask for modality selection or an analysis pipeline before discovery conversion and inspection. Explain that source DICOMs do not need prior conversion or manual separation into sequences. SSD storage can improve conversion throughput; moving data into `sourcedata/` is optional.

### Interaction order: discover first, select afterward

For a new conversion, ask only for the next missing prerequisites: a complete representative DICOM visit path and its explicit subject/session IDs, plus the destination if not already known. Briefly explain IDs when needed. For example: "Please provide one representative DICOM visit and its subject/session IDs. I will first convert it for inspection, identify the available sequences, and then help you choose what to keep." Use the user's language.

Once these prerequisites are available, proceed with the discovery configuration below within the authorized conversion scope. Do not stop with a plan or send an upfront form combining paths, IDs, modality choices, and analysis pipelines. An empty `sourcedata/` folder is not a blocker when the user supplies an external DICOM path. If valid discovery outputs already exist, inspect them instead of repeating conversion.

After inspecting actual JSON/NIfTI outputs, present a short, understandable inventory and a proposed mapping. Ask focused questions only about the observed alternatives or unresolved selections. Accept already supplied preferences, but verify them against the actual data. Keep downstream analysis choices for the analysis stage unless the user explicitly raises them now.

Check that `cvdproc`, `dcm2bids`, `dcm2bids_scaffold`, and `dcm2niix` are available in the same processing environment. Record installed versions using package metadata or supported version commands. CVDProc declares dcm2bids as a dependency; dcm2niix and optional FSL tools still need separate installation. Offer installation help for missing tools instead of assuming they exist. Linux/WSL is the intended processing environment; use paths understood by that environment, such as `/mnt/e/...` within WSL.

Resolve ambiguous destination paths before creating a dataset: `/mnt/d/BIDS` and `/mnt/d/BIDS/MyStudy` are different dataset roots. Set the process working directory to `<BIDS>/code/agent/` once available, or the user's explicit working directory. Use absolute paths in configuration files. Auxiliary records belong in that working directory; images and conversion caches use CVDProc's normal locations.

## 2. Initialize a new dataset or resume an existing one

For a new dataset only:

```bash
cvdproc --run_initialization --bids_dir /path/to/MyStudy
```

Inspect the command result and the dataset structure, including `dataset_description.json`, `participants.tsv`, `code/config_template.yml`, `.bidsignore`, `AGENTS.md`, and `code/agent/guides/dcm2bids.md`. Directory existence alone does not prove scaffold success.

CVDProc treats scaffold software-update check failures as warnings and continues initialization. Actual scaffold failures still stop initialization before CVDProc writes participant tables or reports completion. A previous failed attempt can leave partial files; inspect them before retrying rather than assuming the dataset is complete or deleting it wholesale.

Do not rerun initialization to install these instructions in an existing dataset: the current initializer resets participant tables. Instead use:

```bash
python -m cvdproc.guides --bids_dir /path/to/MyStudy
```

This command only adds missing guide files and the `AGENTS.md` ignore entry. Existing instructions are preserved, including older guide copies; it is not an automatic guide updater. If a root AGENTS.md already exists, incorporate the guide reference without replacing project-specific rules.

Create `code/config.yml` from the generated template, preserving an existing configuration. Set `bids_dir` and `dcm2bids.config_file` to the actual dataset and JSON configuration paths. Start with optional modifications disabled:

```yaml
bids_dir: /path/to/MyStudy
dcm2bids:
  config_file: /path/to/MyStudy/code/dcm2bids_config.json
  ignore: []
  keep_filtered_dicom: false
  dwi_fix_bvecbval: []
  perf_fix_aslcontext: []
  deface_anat: false
  fix_intendedfor: false
  resample_to_iso: []
```

For initial acquisition discovery, a deliberately unmatched description can leave converted files available for inspection:

```json
{
  "descriptions": [
    {
      "datatype": "anat",
      "suffix": "T1w",
      "criteria": {"SeriesDescription": "CVDPROC_DISCOVERY_NO_MATCH_7F93A2"}
    }
  ]
}
```

This is a discovery configuration, not a working T1w mapping. Verify the sentinel does not match any acquired series; replace it with verified criteria before final conversion. Check the installed dcm2bids behavior if discovery reports no matches; distinguish that condition from a failed dcm2niix conversion. Never overwrite an established study mapping with this placeholder.

### Configure dcm2niix options, including automatic 3D cropping

Set `dcm2niixOptions` as a string at the top level of `code/dcm2bids_config.json`, alongside `descriptions`. It belongs neither inside a description nor in CVDProc's YAML `dcm2bids` section. CVDProc passes this JSON file to dcm2bids, which supplies the options to dcm2niix; no extra CVDProc command-line argument is needed.

For automatic cropping, retain the baseline conversion options and add `-x y`:

```json
{
  "dcm2niixOptions": "-b y -ba y -z y -f '%3s_%f_%p_%t' -x y",
  "descriptions": [
    {
      "datatype": "anat",
      "suffix": "T1w",
      "criteria": {"SeriesDescription": "CVDPROC_DISCOVERY_NO_MATCH_7F93A2"}
    }
  ]
}
```

The description remains a discovery placeholder; use the study's reviewed descriptions for actual selection. When editing an existing JSON file, preserve its descriptions and other settings. A custom options string replaces the default string rather than appending to it: do not specify only `-x y`. The baseline above follows [dcm2bids 3.2.0](https://unfmontreal.github.io/Dcm2Bids/3.2.0/how-to/use-advanced-commands/#dcm2niixoptions); check the installed version before adapting it. These options apply to the dcm2niix invocation, not to one description selected afterward. Let dcm2bids manage input and output directories rather than adding `-o` or positional paths here.

Relevant [dcm2niix options](https://github.com/rordenlab/dcm2niix/blob/master/docs/source/dcm2niix.rst):

| Option | Meaning |
| --- | --- |
| `-b y` | Write JSON sidecars needed for matching. |
| `-ba y` | Anonymize identifying fields in generated sidecars; this does not anonymize CVDProc participant tables or source DICOMs. |
| `-z y` | Write compressed NIfTI output. |
| `-f '%3s_%f_%p_%t'` | Set intermediate filenames; preserve the established pattern for traceability. |
| `-x y` | Attempt automatic cropping of eligible 3D acquisitions to remove excess neck. It is not skull stripping and does not guarantee a crop for every series. |
| `-x n` | Disable cropping; dcm2niix's documented default. |
| `-x i` | Disable both cropping and canonical rotation of 3D acquisitions; not equivalent to `-x n`. |

Check `dcm2niix -h` for the installed executable's supported options. Use an already established crop choice; otherwise leave the default unchanged for discovery rather than requiring another upfront choice. If inspection shows a reason to change it, discuss that choice before final conversion and retain it for the cohort. Do not add a separate image-cropping or header-editing step to reproduce this built-in feature.

When dcm2niix emits orphaned `*_Crop_<number>.nii[.gz]` files, CVDProc's existing matching step can associate them with the uncropped BIDS outputs even after JSON files have moved. Section 6 describes unique matching, backups, and the report. Inspect actual crop coverage and matching outcomes; enabling `-x y` alone does not prove that the final BIDS image was replaced.

Changing this string does not retroactively change cached NIfTI files. Inspect the conversion log to verify that dcm2niix actually ran with the requested options. If an earlier uncropped conversion is being reused, regenerate the affected visit from DICOM in a separate staging dataset as described in section 6. Do not delete accepted outputs or caches merely to force cropping.

## 3. Inspect representative visits

### Wait for the complete conversion

Discovery means completing DICOM-to-NIfTI conversion for the entire selected representative visit before selecting BIDS outputs. It is not a time-limited sample. Apply these execution rules to both discovery and final conversion:

- Do not terminate conversion because it takes several minutes or produces no recent output. Do not add arbitrary process-killing limits such as `timeout 300`, or pipe the running converter into `head` or another consumer that closes the output stream early.
- Use a persistent execution session that survives individual tool waits. If the tool has a hard execution limit, arrange a durable job with a recorded process identity, full log, and final exit status before launching. Poll that same job and give progress updates; a polling timeout is not a reason to stop or relaunch it.
- Save complete stdout/stderr in the selected working directory. Read snapshots with `tail` or `head` from the saved log, not by truncating the converter's live output. Preserve the converter's own exit status if using a logging pipeline.
- Wait for the actual completion status and inspect the full conversion log. A disappearing process, a quiet log, or a few generated files does not establish success. Compare the converted inventory with the selected visit's source series and account for exclusions, unsupported series, and reported failures; series and output counts need not be one-to-one.
- If conversion is interrupted, label the outputs incomplete and confirm that the original job and its children have stopped before retrying. Do not treat partial caches as a complete inventory or proceed to final sequence selection. Preserve the diagnostic log and use an isolated staging conversion when cache completeness cannot be established.
- Stop only for a user cancellation, an actual failure, or a diagnosed condition requiring intervention; elapsed time alone is not such a condition. If the environment cannot sustain the job, report that limitation rather than presenting partial results as a completed conversion.

Ask for a complete representative DICOM visit and its explicit subject/session IDs. IDs omit `sub-` and `ses-`, use alphanumeric labels, and must remain consistent across the study. Even a single acquisition requires a session label. Do not infer IDs from clinical identifiers or directory names. Use one example per scanner/protocol variant where necessary; the first subject may not represent the cohort.

```bash
cvdproc --config_file /path/to/MyStudy/code/config.yml --run_dcm2bids \
  --subject_id 001 --session_id baseline --dicom_dir /path/to/dicom/visit001
```

Inspect the exit status, logs, and `tmp_dcm2bids/sub-001_ses-baseline/`. Successful matches may already have moved to `sub-001/ses-baseline/`; inspect both locations as needed. CVDProc invokes dcm2bids with `--auto_extract_entities`, then handles orphaned crops and participant metadata. The conversion also records identifying DICOM fields in the participant table; this is not an anonymization workflow.

Create a compact acquisition inventory in the selected working directory. For each candidate record its original converted filename, relevant JSON metadata, image dimensions/voxel sizes, intended modality, and proposed target entities. Use `SeriesDescription` first, then observed fields such as `ProtocolName`, `ImageType`, echo metadata, and acquisition parameters to disambiguate. Do not invent fields absent from the JSON. Dimensions support inspection but cannot establish acquisition identity or prove reconstruction provenance.

## 4. Build and review the mapping

Begin this stage only after inspecting the representative conversion. Present the acquisitions actually found, with plain-language interpretations, proposed destinations, and any uncertainty. Do not present a generic menu of T1w, T2w, FLAIR, diffusion, ASL, BOLD, SWI, and QSM as if all were available. When the user is unsure, explain the observed choices and propose a mapping for review rather than asking them to identify sequences themselves. Resolve competing acquisitions with focused questions, then confirm the selected mapping before final conversion. A target family can legitimately contain several echoes, parts, directions, or repeated runs; each selected input must have an unambiguous intended destination.

Use the [dcm2bids configuration documentation](https://unfmontreal.github.io/Dcm2Bids/3.2.0/how-to/create-config-file/) matching the installed release. Criteria combine observed metadata; default string patterns are shell-style wildcards, not regular expressions. Avoid overly broad patterns and verify their matches against the complete representative inventory. Keep scanner-specific SeriesNumber criteria out of a cohort-wide rule unless their stability has been established.

### Follow the installed dcm2bids configuration specification

Before writing or changing the JSON configuration, identify the installed dcm2bids version and verify field names, nesting, value types, and matching behavior against that release's official documentation or installed parser. Do not mix examples from different releases or invent spelling variants. For dcm2bids 3.2.0, custom filename entities use `custom_entities` inside a description, not the legacy `customEntities` spelling. The top-level `dcm2niixOptions` field retains its documented camel-case spelling; do not mechanically convert all keys to snake_case.

After writing, parse the file as JSON and review its fields against the supported configuration format. Valid JSON alone does not establish a valid dcm2bids configuration: an unknown key may be ignored rather than rejected. Use a release-supported validator if available; otherwise inspect the relevant parser and verify the resulting filenames and sidecars in the completed representative conversion. Do not invent a validation command. Check that intended custom entities and related-image references actually appear in the outputs before applying the configuration to the cohort.

Use the [BIDS MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html) for modality metadata and filename entities. Check the version used by the project and downstream tools. Use `acq-` for distinct acquisitions and appropriate run/echo/part/direction entities when supported; inspect auto-extracted entities rather than assuming they are correct. Ask about competing clinical, high-resolution, or reconstructed acquisitions when metadata does not settle which should be retained.

Preserve CVDProc project extensions: SWI uses `swi/`; QSM uses `qsm/`, commonly with a `GRE` suffix and explicit echo/part entities. Document these as project conventions, not claims of standard BIDS compliance. Confirm other custom datatype/suffix choices with the user and add narrowly scoped `.bidsignore` patterns. Ignoring an extension does not validate its scientific content or ensure downstream compatibility.

For related field maps and their targets, use the installed dcm2bids `id` and `sidecar_changes.IntendedFor` mechanism where appropriate. Verify emitted references resolve to the intended acquisitions. Do not assume that similar sequence names establish a field-map relationship; inspect phase-encoding and acquisition metadata. Validate JSON syntax before running.

## 5. Select optional CVDProc operations

Read the installed `cvdproc/pipelines/dcm2bids/dcm2bids_processor.py` and CLI when behavior is uncertain. The current order is conversion/crop handling, participant update, gradient replacement, ASL context replacement, defacing, IntendedFor rewriting, and resampling.

| Configuration | Current behavior and decision rule |
| --- | --- |
| `ignore` | Excludes DICOM series by SeriesDescription before conversion. Use only for explicitly unwanted series. Check implementation for pattern syntax; do not confuse these patterns with dcm2bids JSON criteria. |
| `keep_filtered_dicom` | Retains the filtered DICOM working copy; otherwise it is removed after successful external conversion. It does not control the NIfTI conversion cache or source DICOMs. |
| `dwi_fix_bvecbval` | List of `match`, `bvec`, `bval` entries. Filename substring matches select `.nii.gz` images; existing gradient files are replaced from supplied files. Require known correct gradient provenance, volume counts, and coordinate convention. Missing targets can be skipped, so inspect actual results. |
| `perf_fix_aslcontext` | List of `match`, `aslcontext` entries; copies or creates the context table beside matched `.nii.gz` ASL images. Verify row order/count against volumes and the known acquisition, not a guessed alternation. |
| `deface_anat` | Uses FSL to overwrite selected T1w/T2w/FLAIR images. Enable for the requested data handling policy with recoverable originals. It does not remove identifying DICOM or participant metadata. |
| `fix_intendedfor` | Rewrites existing references containing a session component to start with `ses-`. It does not create missing relationships. Enable only when required by the actual downstream consumer; inspect resulting references. |
| `resample_to_iso` | List of `match`, positive numeric `resolution` in mm, and `interp` entries. Current code overwrites the matched image despite the function docstring describing a new `res-` file. Obtain the user's choice and preserve an original before enabling. Do not automatically reduce high-resolution acquisitions to 1 mm. Review metadata consistency afterward. |

Avoid empty or broad filename match strings. Confirm FSL availability for operations using it. Several optional methods log errors or skip missing files instead of failing the whole command: a successful CLI exit alone does not establish that these operations succeeded.

## 6. Validate the representative conversion and revise safely

Save the reviewed JSON mapping, snapshot the configuration, and rerun the same one-visit command. dcm2bids can reuse available conversion files, but verify the installed version's cache behavior and actual files; do not promise every rerun is fast or complete. Optional CVDProc postprocessing can run again on existing images.

Inspect the destination visit for expected acquisitions, JSON/NIfTI pairs, dimensions, voxel sizes, entities, gradient counts, ASL context, and related-image references as applicable. Review image appearance when needed and ask the user to resolve remaining acquisition choices. Use a BIDS validator for the standard portion when available, record its version and findings, and separately check project extensions.

CVDProc may replace an uncropped target only when the orphan crop has a unique exact spatial/voxel match. It moves the crop unchanged, preserves the original under `tmp_dcm2bids/sub-<id>_ses-<id>/crop_backups/`, and records decisions in `crop_matching_*.jsonl`. Inspect skipped or ambiguous crops. Do not edit qform/sform or force a match from dimensions. Retain backups until outputs are accepted.

If a mapping must change, first list affected files and inspect conversion logs/provenance. Moving a BIDS-renamed file back into the cache under its final name is not a reliable restoration procedure. Restore the original converted basename and complete companion set only when that mapping is known and files are still suitable conversion inputs; keep unrelated accepted outputs intact. Account for crop replacements and any defacing/resampling before reuse. If original inputs cannot be recovered reliably, regenerate the affected visit from preserved DICOMs in a separate staging dataset with optional modifications disabled, verify the revised mapping, and replace only the agreed affected outputs. Do not blindly rerun initialization, clear caches, or overwrite a completed visit.

## 7. Convert the cohort

After representative mappings are accepted, prepare an explicit visit table with `subject_id`, `session_id`, `dicom_dir`, and the applicable configuration/protocol group. Validate all paths and labels before execution, reject duplicate destinations and unequal lists, and keep repeated session labels explicit. Ask about missing or unexpected acquisitions; do not silently substitute another series.

For visits sharing a configuration:

```bash
cvdproc --config_file /path/to/MyStudy/code/config.yml --run_dcm2bids \
  --subject_id 001 002 --session_id baseline baseline \
  --dicom_dir /path/to/dicom/visit001 /path/to/dicom/visit002
```

Alternatively use `--dicom_subdir` with paths relative to `<BIDS>/sourcedata/`; supply exactly one source option. Lists pair by position, with no broadcasting. Current `--run_dcm2bids` runs sequentially; `--n_jobs` applies only to `--run_pipeline`. Do not start concurrent conversion processes writing the shared participant table. A conversion exception stops the current CLI batch; retain completed results and resume only unresolved visits after diagnosis.

Record per-visit command/configuration, exit status, observed outputs, checks, and unresolved issues in the selected working directory. Separate missing input, conversion failure, unmatched acquisition, ambiguous crop, and optional-operation failure. Do not label a discovery-only run as a completed BIDS conversion.

Finish with a concise account of converted/failed/pending visits, configuration locations, retained backups, and any checks not performed. Preserve `sourcedata/`, configurations, and useful provenance. Remove only task-owned disposable files within the authorized cleanup scope.
