# Usage

**Please refer to the different pages for specific functions:**

If you need to create a BIDS dataset from DICOM files, please refer to:

- [dcm2bids](dcm2bids/dcm2bids.md): Create BIDS dataset from DICOM files.
  
Or if you already have a dataset in BIDS format, you can use the following functions:

- [Pipelines](pipelines/index.md): Different pipelines for MRI preprocessing and analysis.

## Running visits in parallel

Use `--n_jobs` with `--run_pipeline` to set the maximum number of subject/session workflows running at once. The default is `1`, which retains sequential execution.

```bash
cvdproc --config_file config.yml --run_pipeline --pipeline wmh_quantification \
        --subject_id 001 002 003 --session_id 01 01 02 --n_jobs 2
```

This runs the explicitly paired visits `(001, 01)`, `(002, 01)`, and `(003, 02)`, with at most two active at once. Both ID lists are required and must have equal lengths; sessions are not inferred or repeated automatically.

Each concurrent visit constructs and runs its workflow in a separate process, with its own subject/session output and workflow directories. `--n_jobs` controls visit concurrency, not the threads, memory, or GPU resources used inside an individual pipeline. Choose the count to fit the combined resource needs of the selected pipeline.

Parallel batches reject duplicate subject/session pairs. For `freesurfer_longitudinal`, concurrent entries must also have distinct subjects because each entry processes that subject's longitudinal data.

In parallel mode, other submitted visits continue if one fails. The command reports the failed subject/session pairs and exits unsuccessfully after the batch finishes; subsequent result extraction is not started. Sequential mode continues to stop at the first failure. Console progress from concurrent visits can be interleaved and includes visit IDs; Nipype crash files are saved under the corresponding visit's workflow directory.
