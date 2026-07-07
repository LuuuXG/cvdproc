# ARTS Pipeline

::: cvdproc.pipelines.multi.arts_pipeline.ARTSPipeline
    options:
      show_signature: false

----

## A more detailed description:

ARTS (Arteriolosclerosis Biomarker) is a multi-modality pipeline that computes a composite score reflecting cerebral arteriolosclerosis burden [@rudolph2026arts]. The pipeline integrates information from multiple MRI-derived biomarkers and is approximately 10x faster than the original implementation.

### Modalities and Dependencies

ARTS requires outputs from several other pipelines (must be run first):

- **xfm** (T1w registration): T1w brain and FLAIR brain in T1w space
- **anat_seg**: SynthSeg anatomical segmentation
- **dwi_pipeline**: FA map from DTI tensor fitting
- **wmh_quantification**: WMH mask in T1w space

Additional inputs:
- `participants.tsv` for age and sex information
- `IITmean_FA.nii.gz` mean FA template (IIT space)
- ARTS Singularity image (`ARTS.sif`)

### Configuration Example

```yaml
pipelines:
    arts_pipeline: {}
```

### Parameters

- `extract_from`: Path to ARTS output directory from which to extract group-level score summaries.

### References

\bibliography
