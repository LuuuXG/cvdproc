# Lesion Analysis

::: cvdproc.pipelines.smri.lesion_analysis_pipeline

-----

## A more detailed description:

### Lesion Mask

A basic consideration is to use T1w structural MRI for the basic processing of lesions. In previous studies, we have tried using DWI for lesion delineation (as acute ischemic stroke lesions are more apparent on DWI), but subsequent neuroimaging analysis often involves image registration, and using low-resolution (and distorted) DWI for registration can introduce significant inaccuracies. Therefore, we currently prefer to manually delineate lesions in T1w space for subsequent analysis.

- if output contralateral lesions is desired (`out_contra_mask: true`): left-right flipped in 'MNI152NLin2009aSym' space and transformed to native space.

- if normalization is desired (`normalize: true`): the lesion mask is normalized to the standard 'MNI152NLin6Asym' space.
  This runs the same SynthStrip/SynthMorph processing as the default `t1_register` configuration (MNI152NLin6Asym, 1 mm, registration between skull-stripped images). It uses the lesion-filled T1w if one is generated, or the selected original T1w otherwise (including `lesion_fill: false`). The mask is transformed with nearest-neighbor interpolation.

  The subject/session `derivatives/xfm` directory receives all five default `t1_register` outputs: the MNI-space T1w, forward and inverse warp fields, native-space skull-stripped T1w, and native-space brain mask. Output names follow the selected T1w filename and the `t1_register` naming convention. A separate default `t1_register` run is therefore unnecessary. Other template spaces and resolutions still require `t1_register`.

  The registration node is included even if a forward warp already exists, so a partial previous result does not skip the remaining outputs. Running it may replace existing registration outputs; normal Nipype workflow caching still applies. Lesion size metrics continue to use the input lesion mask in native space.

### Lesion Filling

We currently use the [Lesion Inpainting Tool (LIT)](https://github.com/Deep-MI/LIT) developed by the Deep-MI team for lesion filling. This functionality can be enabled by setting `lesion_fill: true` and `lesion_fill_method: 'LIT'` (requires downloading the docker image: deepmi/lit:0.5.0).

Considerations for other lesion filling methods:

- [KUL_VBG](https://github.com/KUL-Radneuron/KUL_VBG): This method involves ANTs-based registration. It requires a longer time for processing compared to LIT.
- [FSL Lesion Filling](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/lesion_filling.html): Taking this as a representative example, we found that many other lesion filling methods are based on WMH lesions in MS patients and are not suitable for the brain infarction lesions studied in our research.

### Lesion Size

Three metrics are commonly used to quantify lesion size (only applicable for single-cluster lesion):

1. **Lesion Volume**: The total volume of the lesion in cubic millimeters (mm³).

2. **Max Diameter (Axial)**: The maximum diameter of the lesion in the axial plane, measured in millimeters (mm).

3. **Max Diameter (3D)**: The maximum diameter of the lesion in three-dimensional space, measured in millimeters (mm).
