# SynthSR Pipeline

::: cvdproc.pipelines.smri.freesurfer_pipeline.SynthSRPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline runs [SynthSR](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR) [@iglesias2023synthsr] to generate a synthetic 1 mm isotropic T1w image from a clinical-quality structural MRI. SynthSR is useful for standardizing lower-resolution clinical scans or heterogeneous acquisitions into a research-grade 1 mm isotropic T1w image.

### Modalities

- T1w (default): Uses a T1w image as input.
- FLAIR: Can optionally use a FLAIR image as input (set `input_type: 'FLAIR'`).

### Parameters

- `input_type`: Input image type, either `'T1w'` (default) or `'FLAIR'`. Only one type can be specified.
- `use_which_t1w`: Select a specific T1w image by matching a substring in the filename (when `input_type='T1w'`).
- `use_which_flair`: Select a specific FLAIR image by matching a substring in the filename (when `input_type='FLAIR'`).

!!! note
    SynthSR is also run automatically as part of `recon-all-clinical.sh` (see [Freesurfer](freesurfer.md)). This standalone pipeline is useful when you only need the synthetic 1 mm output without running the full clinical FreeSurfer pipeline.

### References

\bibliography
