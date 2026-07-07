# PVS Quantification

::: cvdproc.pipelines.smri.csvd_quantification.pvs_pipeline
    options:
      show_signature: false

----

## A more detailed description:

### Modalities

- T1w (required for both methods)
- FLAIR (optional, only used when `method='SHIVA'` and `modality='T1w+FLAIR'`)

### Methods

Two methods are available for perivascular space (PVS) segmentation:

#### SHIVA (`method='SHIVA'`)

Uses the [SHiVAi](https://github.com/pboutinaud/SHiVAi) toolbox for PVS detection [@boutinaud2021shiva]. Supports both T1w-only and T1w+FLAIR modalities.

!!! warning "SHiVAi Report Generation Issue"
    Current SHiVAi (2025-06) seems to have problems handling HTML/PDF reports (for example, when facing NaN values). Our pipeline works around this by disabling the report generation step, which is not critical for the pipeline to run.

#### SegCSVD (`method='segcsvd'`, default)

Uses the [segcsvd](https://github.com/AICONSlab/segcsvd) toolbox for PVS segmentation from T1w images [@gibson2026segcsvd]. This is currently the recommended method. The processing steps are:

1. **SynthSeg** — Anatomical segmentation of the T1w image [@billot2023synthseg] (uses existing results from `anat_seg` if available).
2. **SHiVA-based PVS parcellation** — A custom parcellation derived from SHiVA partitions the brain into regions relevant for PVS analysis (deep WM, basal ganglia, hippocampus, cerebellum, ventral DC, brainstem).
3. **Skull-stripping & N4 bias correction** — SynthStrip removes non-brain tissue, followed by N4 bias field correction.
4. **SegCSVD PVS segmentation** — Detects PVS from the bias-corrected T1w using the segcsvd model. If `use_wmh=True` and a pre-existing WMH mask is found in the `wmh_quantification` derivatives, WMH regions are masked out to avoid false positives.
5. **Regional volume calculation** — PVS volume is quantified within each parcellation region.

### Configuration Example

```yaml
pipelines:
    pvs_quantification:
        method: "segcsvd"
        use_which_t1w: "acq-highres"
        use_wmh: true
```

### References

\bibliography
