# CAT12 Pipeline

::: cvdproc.pipelines.smri.cat12_pipeline.CAT12Pipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline runs [CAT12](https://neuro-jena.github.io/cat/) (Computational Anatomy Toolbox 12) for SPM-based structural MRI segmentation. CAT12 provides voxel-based morphometry (VBM) and surface-based morphometry (SBM) using a standalone MATLAB runtime, without requiring a full MATLAB installation.

### Processing Steps

1. T1w image is selected (filterable by `use_which_t1w`).
2. CAT12 standalone segmentation is run with Schaefer 2018 atlas ROIs at multiple parcellation resolutions (100, 200, 400, 600 parcels).
3. Registration voxel size is set to 1 mm isotropic.
4. Total intracranial volume (TIV), CSF, GM, and WM volumes are extracted from the CAT12 XML report.

### Dependencies

- MATLAB Runtime v93 (R2017b)
- CAT12 standalone package (`cat12_standalone_path`)

### Configuration Example

```yaml
pipelines:
    cat12:
        cat12_standalone_path: "/path/to/cat12_standalone"
        use_which_t1w: "acq-highres"
        job: "segmentation"
```

### Parameters

- `use_which_t1w`: Select a specific T1w image by matching a substring in the filename.
- `matlab_path`: Path to MATLAB executable (default: `'matlab'`).
- `job`: Processing job type (default: `'segmentation'`).
- `cat12_path`: Path to CAT12 installation.
- `cat12_standalone_path`: Path to CAT12 standalone package (required for standalone mode).
- `extract_from`: Path to extract results from for population-level summary.
