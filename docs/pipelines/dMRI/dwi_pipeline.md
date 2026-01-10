# DWI Pipeline

::: cvdproc.pipelines.dmri.dwi_pipeline.DWIPipeline
    options:
      show_signature: false

----

## A more detailed description:

### DWI Preprocessing

#### Single-shell and Multi-shell DWI preprocessing

We use customized scripts to preprocess single-shell and multi-shell DWI data, because current popular DWI preprocessing tools (e.g., [MRtrix3](https://www.mrtrix.org/), [QSIPrep](https://qsiprep.readthedocs.io/en/latest/index.html)) do not provide an easy way to use [Synb0-DISCO](https://github.com/MASILab/Synb0-DISCO) for distortion correction when no field map is available.

The main steps of our DWI preprocessing pipeline are as follows:

1. If no reverse phase-encoding (PE) b0 image is available, we use Synb0-DISCO to synthesize the undistorted b0 image from T1w image. Otherwise, we directly use the reverse PE b0 image.
2. Denoise and Gibbs ringing removal (Gibbs ringing removal can be optionally applied).
3. FSL's TOPUP and EDDY for distortion and motion correction, using the real or synthesized reverse PE b0 image.
4. Resample the DWI data to a isotropic voxel size.

#### DSI preprocessing

We use [QSIPrep](https://qsiprep.readthedocs.io/en/latest/index.html) for DSI preprocessing. Our DWI Pipeline starts from QSIPrep preprocessed DSI data for further analysis. TOPUP and EDDY-based distortion and motion correction are not suitable for q-space sampling schemes like DSI, this is the reason why we do not employ [DSIStudio](https://dsi-studio.labsolver.org/doc/) for DSI preprocessing.

Concerning the DSI preprocessing pipeline implemented in QSIPrep is time-consuming (e.g., [Very long running time qsiprep processing DSI data](https://neurostars.org/t/very-long-running-time-qsiprep-processing-dsi-data/28972)), we make some modifications to speed up the process (by running the script [qsiprep_single.sh](../../../cvdproc/pipelines/bash/qsiprep_qsirecon/qsiprep_single.sh)):

1. Use `DRBUDDI_cuda` rather than `DRBUDDI` for distortion correction. Note that `DRBUDDI_cuda` requires a CUDA-capable GPU. Another change is to set `--DRBUDDI_disable_initial_rigid` flag to skip the initial rigid registration for a more stable run on PC.
2. Set `shoreline-iters` to `1`
3. Skip ANTs-based spatial normalization to MNI space by setting `--skip-anat-based-spatial-normalization` flag. To generate non-linear spatial warps to MNI space (needed for downstream QSIRecon processing), we run a separate registration using [mri_synthmorph](https://martinos.org/malte/synthmorph/) implemented in Freesurfer 7-dev version.
4. Only use T1w image as anatomical reference. This is ensured by excluding T2w and T2 FLAIR images with nipreps' `qsiprep_filter.json` file (see [How do I select only certain files to be input to fMRIPrep?](https://fmriprep.org/en/latest/faq.html#how-do-i-select-only-certain-files-to-be-input-to-fmriprep)).

For Chinese readers, notes on using QSIPrep for DSI preprocessing can be found [here](https://neuroimaging-notes-of-lxg.readthedocs.io/zh/latest/dMRI/qsiprep2.html).

If you are interested in replicating our DSI preprocessing pipeline: please first modify QSIPrep's docker file by using a new [tortoise.py](../../../cvdproc/utils/python/tortoise.py), then you can reference the script ([qsiprep_single.sh](../../../cvdproc/pipelines/bash/qsiprep_qsirecon/qsiprep_single.sh)) to run QSIPrep for DSI preprocessing (DWI and reverse b0 images in BIDS format, and put Freesurfer `license.txt`, `DRBUDDI_cuda.sh` [wrapper script](../../../cvdproc/pipelines/bash/qsiprep_qsirecon/DRBUDDI_cuda.sh), and `qsiprep_filter.json` in a folder named `code` in BIDS root directory). 

### Reconstruction of Diffusion Models

#### Tensor Model Fitting

FSL FDT `dtifit` is used to fit the diffusion tensor model and calculate DTI-derived metrics, including FA, MD, AD, RD, and tensor mode.

#### Free Water Elimination DTI Model Fitting

Three available options:
- Set 'markvcid_freewater' in `freewater` (**Recommended**): [MarkVCID2 MRI Free Water (FW)](https://markvcid.partners.org/sites/default/files/markvcid2/protocols/MarkVCID_FW_Kit_Protocol_v6.12.23.pdf). As it has been validated on MarkVCID2 study [@maillard2022mri] and external datasets [@lin2025longitudinal].

- Set 'single_shell_freewater' in `freewater`: [Single Shell Free Water Elimination Diffusion Tensor Model](https://github.com/sameerd/DiffusionTensorImaging)

- Set 'dti_freewater' in `freewater`: [DIPY free water elimination model](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_fwdti.html). Multi-shell DWI data is required for this method. For our clinical analysis, we mainly focus on the free water component rather than the free water-corrected DTI metrics, this method is not that practical as multi-shell DWI can directly use NODDI model to characterize extracellular water distribution (ISO/FW).

#### NODDI Model Fitting

[AMICO](https://github.com/daducci/AMICO) (Accelerated Microstructure Imaging via Convex Optimization) is used to fit the NODDI model and calculate NODDI-derived metrics, including ICVF, ISOVF, and ODI.

#### DSI Model Fitting

DSIStudio is used to reconstruct the diffusion ODFs and calculate DSI-derived metrics, including GFA, QA etc.

### Anatomical Processing

#### Registration between DWI and anatomiocal Images (fsnative and T1w space)

While QSIPrep register DWI to T1w space during preprocessing using b0 image, we prefer to use FA map for registration because FA map has better contrast for white matter structures.

If corresponding Freesurfer `recon-all` results are available, some extra processing will be performed for possible subsequent connectome analysis (see below).

### DWI-derived Metrics

#### PSMD

PSMD (Peak Width of Skeletonized Mean Diffusivity), considered a marker of white matter injury, is calculated using [psmd](https://github.com/isdneuroimaging/psmd) script [@zanon2023peak]. We use a previous version of the script, while the latest version prefers a Docker-based implementation.

#### DTI-ALPS

DTI-ALPS (Diffusion Tensor Image Analysis along the Perivascular Space) is calculated using a customized [script](https://github.com/gbarisano/alps).

The original method was proposed in [@taoka2017evaluation]. It is important to note that multiple studies have suggested that DTI-ALPS should not be simply interpreted as a reflection of glymphatic function. Instead, DTI-ALPS likely reflects complex changes in brain microstructure, pointing to a more comprehensive neurodegenerative mechanism [@li2025microstructural].

#### PVeD

periventricular diffusivity (PVeD) is proposed as a substitute marker reflecting the glymphatic function in the brain [@chen2025periventricular]. It is calculated using the official [EstPVeD](https://github.com/ChangleChen/EstPVeD) script.

Things to concern: A region growing method is used to determine the periventricular region. However, if there are lesions (such as lacunar infarcts with high MD values) in this region, these lesion areas will be ignored, leading to variability in the periventricular region among subjects.

### Connectome Analysis

!!! Note
    FreeSurfer `recon-all` results are required for this part of analysis.

#### MRtrix3-based Connectome Construction

Mrtrix3 is recommended for connectome construction. The main steps are as follows:

1. `dwi2response dhollander/tournier` to estimate the response functions for different tissue types (for single-shell data, only WM response function is estimated).

2. `dwi2fod msmt_csd/csd` to calculate the fiber orientation distributions (FODs) using constrained spherical deconvolution (CSD) (for multi-shell/single-shell data, respectively).

3. `mtnormalise` to perform multi-tissue informed log-domain intensity normalization on FODs.

4. `5ttgen freesurfer` to generate the 5-tissue-type (5TT) image from Freesurfer `recon-all` results. That's why Freesurfer results are required for this part of analysis. It is convenient to use Freesurfer parcellation as brain atlas for connectome construction, and outperforms `5ttgen fsl` in our experience (especially for subjects with severe WMH and Lacunes).

5. `5tt2gmwmi` to generate the gray matter-white matter interface (GM-WM interface) from the 5TT image.

6. `tckgen` to generate 1000000 streamlines using the iFOD2 probabilistic tracking algorithm with anatomically constrained tractography (ACT) framework. The GM-WM interface is used as seeding mask.

7. `tcksift2` to compute the cross-sectional area multipliers for each streamline using SIFT2 method.

8. `tck2connectome` to generate the connectome matrix using Freesurfer parcellation as brain atlas (Currently using aparc and aparc+aseg). The connectome weights are set as SIFT2-weighted streamline count.

### References

\bibliography