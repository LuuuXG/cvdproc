# FSL Anat

::: cvdproc.pipelines.smri.fsl_anat_pipeline.FSLANATPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline runs [FSL's fsl_anat](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/fsl_anat) on a T1w image. fsl_anat is a wrapper script that performs standard structural MRI preprocessing including:

- Reorientation to standard (MNI) space
- Automatic cropping
- Bias field correction (FAST)
- Linear and non-linear registration to MNI152 space
- Segmentation (CSF, GM, WM)
- Subcortical structure segmentation (FIRST)

The outputs include a bias-corrected T1w image, brain-extracted images, tissue segmentation maps, and registration warps to MNI space.

### Modalities

- T1w (required)

### Parameters

- `use_which_t1w`: Select a specific T1w image by matching a substring in the filename (e.g., `'acq-highres'`). If not specified, the first T1w image found is used.
