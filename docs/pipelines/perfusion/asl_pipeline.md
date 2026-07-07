# ASL Pipeline

::: cvdproc.pipelines.perfusion.asl_pipeline.ASLPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline processes Arterial Spin Labeling (ASL) data using [ExploreASL](https://exploreasl.github.io/), a comprehensive ASL processing toolbox.

### Modalities

- ASL / perfusion (required)
- T1w (required for registration to MNI space)
- M0 scan (optional; auto-detected from ASL context file or separate `_m0scan` NIfTI)

### Processing Steps

1. ASL and T1w images are selected (filterable by `use_which_asl` and `use_which_t1w`).
2. If an M0 scan is present, it is used for CBF quantification.
3. If a T1w-to-MNI non-linear warp does not already exist (from `xfm` derivatives), a SynthMorph registration is performed.
4. ExploreASL computes CBF maps in native space, which are then registered to T1w space and MNI space.
5. If multi-PLD data is available (detected automatically from the ASL JSON sidecar), arterial transit time (ATT) maps are also generated and normalized.

### Parameters

- `use_which_asl`: Select a specific ASL image by matching a substring in the filename.
- `use_which_t1w`: Select a specific T1w image by matching a substring in the filename.
- `preprocess_method`: Processing method. Currently only `'ExploreASL'` is supported (default).

### References

\bibliography
