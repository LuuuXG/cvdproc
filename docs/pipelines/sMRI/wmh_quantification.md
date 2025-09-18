# WMH Quantification

::: cvdproc.pipelines.smri.csvd_quantification.wmh_pipeline

-----

## A more detailed description:

### modalities to use

- T1w
- FLAIR

!!! warning "Note on T1w"
    If T1w is provided, make sure it is a 3D T1w image. In the following description, we will not specify whether T1w is 2D or 3D, as we assume T1w is 3D by default.

We handle T1w and FLAIR images using the following logic: You can provide either a T1w or a FLAIR image, or both (T1w/FLAIR/T1w+FLAIR). If T1w is provided, the segmentation will be performed on the T1w space (if FLAIR is also provided, FLAIR image will be registered to T1w space and then segmented). The concern here is that if the WMH is segmented on FLAIR image and then transformed to T1w space using a linear registration, the edges of the WMH mask may appear jagged or discontinuous.

### segmentation methods

Please refer to the table below for the available segmentation methods and their corresponding modalities and notes.

My advice on method selection:

  - The sensitivity of the methods (especially the ability to detect WMH lesions in the deep white matter). More generally, the nature of the WMH lesions in your dataset (CSVD-related WMH, MS lesions, etc.). For instance, LST-LPA and LST-AI are based on MS data.
  - Whether the method will identify WMH in corpus callosum or not
  - The modalities you have (T1w only, FLAIR only, or both T1w and FLAIR)
  
Currently, you can only choose one method for WMH segmentation. So a possible way to choose a method is to modify the config file and run the pipeline multiple times with different methods, then compare the results.

| method name | seg_method | modalities | notes |
|-------------|------------|------------|-------|
| LST-LPA | 'LST' | (2D/3D) FLAIR / (2D/3D) FLAIR + T1w | [LST](https://www.applied-statistics.de/lst.html). According to my experience, this method can effectively identify PWMH lesions, but may struggle with smaller DWMH. It is a simple and convenient method for WMH segmentation and is applied in various studies. |
| LST-AI | 'LSTAI' | (2D/3D) FLAIR + T1w | [LST-AI](https://github.com/CompImg/LST-AI). Please specify 'LSTAI' rather than 'LST-AI' in the config file (to avoid extra '-', which conflicts with BIDS style). This method is more sensitive to DWMH detection, but may not identify some PWMH. Additionally, attention should be paid to the selection of thresholds, as the resulting probmap is the average of three models, and many values close to 0.33 or 0.66 may appear in the probmap. |
| WMH-SynthSeg | 'WMHSynthSeg' | (2D/3D) FLAIR / T1w | [WMH-SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg). The advantages of this method include the ability to obtain 1mm resolution results, even when the input is 2D FLAIR. However, based on my experience, the segmentation performance varies across different FLAIR sequences, and it may exaggerate WMH. |
| FSL truenet | 'truenet' | 3D FLAIR / T1w / 3D FLAIR + T1w | [FSL truenet](https://github.com/v-sundaresan/truenet). The advantages of this method include the provision of multiple models, suitable for single-channel FLAIR and T1w or dual-channel FLAIR + T1w, especially FLAIR + T1w will get very good results, but the preprocessing time is relatively long (because it involves steps such as FAST segmentation). In addition, the preprocessing process seems to have some problems for the case of only FLAIR, and the results obtained without preprocessing when only FLAIR is used are consistent with the results obtained after running preprocessing. |

### bianca mask

If 'use_bianca_mask' is set to True, an inclusion mask is applied after the segmentation step (ref: 'The script below creates an example of inclusion mask from T1 images, which excludes cortical GM and the following structures: putamen, globus pallidus, nucleus accumbens, thalamus, brainstem, cerebellum, hippocampus, amygdala. The cortical GM is excluded from the brain mask by extracting the cortical CSF from single-subject’s CSF pve map, dilating it to reach the cortical GM, and excluding these areas. The other structures are identified in MNI space, non-linearly registered to the single-subjects’ images, and removed from the brain mask.' in [FSL BIANCA](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/BIANCA(2f)Userguide.html)).

Other WMH segmentation methods not included in the pipeline:

- [UBO detector](https://cns.readthedocs.io/en/latest/manual/quickstart.html)
- [SHiVAi](https://github.com/pboutinaud/SHiVAi)
- [segcsvd](https://github.com/AICONSlab/segcsvd)
- [MARS-WMH](https://github.com/miac-research/MARS-WMH)
- [FSL BIANCA](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/BIANCA(2f)Userguide.html): not included because it requires training data

### lateral ventricle mask

'ventmask_method' is used to specify the method for obtaining the lateral ventricle mask, which is required for the Fazekas location method. You can choose one of the following methods: 'SynthSeg'

- 'SynthSeg': use [SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg)

### location methods

You can choose one or more location methods to get the location information of WMH.

- 'Fazekas': classify WMH into periventricular WMH (PWMH) and deep WMH (DWMH) based on the distance to the lateral ventricles of each WMH cluster. WMH lesions totally outside the 10mm distance to the lateral ventricles are classified as DWMH. Those totally within 10mm distance or partially within 10mm distance are classified as PWMH (Theoretical should be divided into periventricular WMH and confluent WMH, but simplified to PWMH. See this [Keller et al.](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.13345) for more information). 
- 'bullseye': calculate WMH volume according a bullseye parcellation scheme (see [Sudre et al.](https://linkinghub.elsevier.com/retrieve/pii/S0150986117302249)). Do this classification need precomputed freesurfer recon-all results.
- 'shiva': calculate WMH volume according to the SHIVA parcellation scheme (Shallow, Deep, Perivetricular, Cerebellar, and Brainstem). See [shivai](https://github.com/pboutinaud/SHiVAi) for more information.
- 'McDonald': This utilizes the annotation of WMH clusters in [LST-AI](https://github.com/CompImg/LST-AI). It is more suitable for MS lesions.
  
### normalization to MNI space

If 'normalize_to_mni' is set to True, the WMH segmentation results will be transformed to MNI space using a non-linear registration (SynthMorph).

### shape features

If 'shape_features' is set to True, shape features of PWMH and DWMH will be calculated. We refer to the SMART-MR studies (e.g., [Keller et al.](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.13345), [Ghaznawi et al.](https://www.neurology.org/doi/10.1212/WNL.0000000000011827)) and [Han et al.](https://www.karger.com/Article/FullText/515836)

Methodological issues must be considered:

 - The simple hypothesis is that WMH lesions will be more complex and irregular as they grow larger. However, WMH in different locations may have different growth patterns. For example, a normal aging-related PWMH lesion may reveal a cap-like shape along the lateral ventricle (Fazekas score 1, assume have 4 WMH clusters in left/right anterior/posterior horn of lateral ventricle, respectively), these WMH clusters may grow and merge to a single large PWMH lesion (Fazekas score 3). So, compare the small and large PWMH lesions directly may not be appropriate (in another word, the shape features may dramatically change when two or more WMH clusters merge).
 - If you want to compare the shape features of WMH between different subjects, is it rational to simplely calculate the mean shape features of all WMH clusters in each subject? It seems OK with DWMH clusters, but not for PWMH clusters (see the above point).