# ARTS Pipeline

::: cvdproc.pipelines.multi.arts_pipeline.ARTSPipeline
    options:
      show_signature: false

----

## A more detailed description:

ARTS (Arteriolosclerosis Biomarker) is a multi-modality pipeline that computes a composite score reflecting cerebral arteriolosclerosis burden [@rudolph2026arts]. The pipeline integrates information from multiple MRI-derived biomarkers and is approximately 10x faster than the original implementation.

### Modalities and Dependencies

ARTS requires outputs from several other pipelines (must be run first):

- **anat_seg**: SynthSeg anatomical segmentation
- **dwi_pipeline**: Native-space FA for `v1`, or an FA map in `MNI152NLin6Asym` space for `v2`
- **wmh_quantification**: WMH mask in T1w space
- **xfm** (only for `v1`): T1w brain and FLAIR brain in T1w space

Additional inputs:
- `participants.tsv` for age and sex information
- ARTS data resources, including the IIT mean FA template, IIT atlas files, feature extractor, and classifier
- MNI152NLin6Asym-to-IIT warp distributed in the cvdproc data package (only for `v2`)

The pipeline locates the packaged ARTS data directory automatically; it is not a user-configurable parameter.

### Configuration Example

```yaml
pipelines:
    arts_pipeline:
        method: v2
```

### Parameters

- `method`: ARTS implementation to use. `v2` (default) uses an FA map already registered to `MNI152NLin6Asym` space and skips T1w, FLAIR, and SynthMorph registration. The legacy `v1` method requires `ARTS.sif`, which is no longer included; selecting `v1` raises an error that directs the user to `v2`.
- `extract_from`: Path to ARTS output directory from which to extract group-level score summaries.

### References

\bibliography
