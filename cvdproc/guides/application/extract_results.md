# Extracting existing CVDProc results

Follow the dataset AGENTS.md shared application workflow. Result extraction summarizes existing outputs; it does not by itself authorize rerunning imaging analyses or filling missing results with another method.

## 1. Establish the requested result

Ask which measurements, pipeline outputs, subjects/sessions, and aggregation level the user needs when not already known. Clarify relevant analysis identity, such as method, atlas, resolution, or hemisphere. Identify the completed source analysis and its location before selecting an extractor.

Resolve the registered class through `cvdproc/controllers/pipeline_manager.py`. Inspect its constructor and `extract_results` implementation, including any called readers. Confirm that extraction is implemented, produces the requested measurements, and supports construction with `subject=None` and `session=None`. A registered pipeline is not necessarily extractable. If support is missing, explain that limitation instead of claiming extraction succeeded or silently implementing a replacement.

## 2. Check sources and extraction dependencies

Locate actual source files using the extractor's path logic. Some pipelines expose `extract_from`; others use different settings. Do not assume that a generic `extract_from` or a common folder layout works for all pipelines. Distinguish the input result tree from the destination summary directory and avoid reading an earlier summary as a fresh source.

Check the requested visits for missing, failed, incomplete, or incompatible analyses. Report gaps and determine whether partial extraction is acceptable when not already specified. Verify only dependencies used by the extractor and its constructor; do not require or install an entire processing stack without checking what this operation actually needs.

Inspect metric definitions, units, label mappings, and aggregation rules. Preserve subject/session and analysis identity. Do not turn missing values into zero, average incompatible runs, or mix atlases/methods without an explicit scientifically supported rule.

## 3. Configure scope and destination

Use the study YAML, preserve unrelated settings, and configure the selected pipeline's supported source parameters. Set an absolute `output_dir`; by default the CLI passes `<output_dir>/population/<pipeline>` as the summary destination. An explicit `--output_path` overrides that destination for extraction; verify whether the extractor honors it or adds its own subdirectories.

The current extraction branch does not pass CLI `--subject_id` or `--session_id` lists to the extractor. Do not promise these flags filter extraction. Determine whether the implementation supports a scope parameter. If it does not, explain its actual scope and, where appropriate, filter a separately saved summary by explicit identifiers after extraction, documenting that step and preserving the original summary.

Use a separate destination when needed to preserve an accepted summary. The CLI requires a nonempty `pipelines.<name>` configuration; use genuine supported options. Review configuration and dependencies before invoking the constructor, since implementations can differ in side effects.

## 4. Extract and validate

Use the selected registered name in place of `PIPELINE_NAME`:

```bash
cvdproc --config_file /path/to/MyStudy/code/config.yml \
  --extract_results --pipeline PIPELINE_NAME
```

Prefer a separate extraction command when reviewing preexisting or partly completed analyses. If combined with `--run_pipeline`, a processing failure prevents the later extraction branch from running. `--n_jobs` controls pipeline execution, not result extraction.

Check the return status and actual written outputs; a completion message does not prove that an extractor wrote meaningful data. Validate row counts against the expected scope and table grain, uniqueness of identifiers, metric columns and units, missingness, and any excluded visits. Trace representative values back to their source files and verify aggregation logic where relevant.

Report the summary paths, included and missing visits, measurement definitions needed to interpret the result, and any post-extraction filtering. Preserve the configuration and concise extraction record in the selected auxiliary working directory. Report unimplemented extraction, empty output, or partial coverage explicitly; do not describe them as a complete cohort result.
