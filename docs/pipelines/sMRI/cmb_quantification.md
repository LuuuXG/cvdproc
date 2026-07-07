# CMB Quantification

::: cvdproc.pipelines.smri.csvd_quantification.cmb_pipeline.CMBSegmentationPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline performs cerebral microbleed (CMB) segmentation using the [SHiVAi](https://github.com/pboutinaud/SHiVAi) toolbox.

### Modalities

- SWI (required)
- T1w (required, used as reference space for SWI registration)

### Processing Steps

1. SWI and T1w images are selected (filterable by `use_which_swi` and `use_which_t1w`).
2. SWI is registered to T1w space using linear registration (FLIRT).
3. SHiVAi performs CMB detection on the T1w-registered SWI image.

### Configuration Example

```yaml
pipelines:
    cmb_quantification:
        method: "SHIVA"
        modality: "swi"
        threshold: 0.5
```

### Parameters

- `use_which_swi`: Select a specific SWI image by matching a substring in the filename.
- `use_which_t1w`: Select a specific T1w image by matching a substring in the filename.
- `method`: Currently only `'SHIVA'` is supported.
- `modality`: Currently only `'swi'` is supported.
- `threshold`: Detection threshold for CMB probability map (default: 0.5).
- `predictor_files`: List of predictor model files for SHiVAi.
- `crop_or_pad_percentage`: Padding percentage for SHiVAi input (default: `(0.5, 0.5, 0.5)`).
- `save_intermediate_image`: Whether to save intermediate images (default: `false`).
