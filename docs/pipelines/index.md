# Pipelines Overview

## Command to Run a Pipeline
```bash
cvdproc --config_file <path/to/your/config/file> --run_pipeline --pipeline <pipeline name> --subject_id <subject id (with out -sub prefix)> --session_id <session id (with out -ses prefix)>
```

Generally, parameters concerning the pipeline should be set in the configuration file (this configuration file has been mentioned in the [dcm2bids](../dcm2bids/dcm2bids.md) section). As we already set the `bids_dir` parameter in the configuration file, we will need to add some new parameters:
```yaml
# For example
bids_dir: /mnt/f/BIDS/demo_wmh # we have done it in the dcm2bids section
dcm2bids: 
  config_file: /mnt/f/BIDS/demo_wmh/code/dcm2bids_config.json
# New things
output_dir: /mnt/f/BIDS/demo_wmh/derivatives # output directory, please use bids::/derivatives
pipelines:
    wmh_quantification: # the pipeline name, can be changed to any available pipeline (see below)
        # Then the pipeline-specific parameters go here
        use_which_flair: "acq-tra"
        use_which_t1w: 'acq-highres'
        seg_threshold: 0.5
        seg_method: "LST"
        location_method: ["Fazekas", "shiva", "McDonald"]
        ventmask_method: 'SynthSeg'
        use_bianca_mask: false
        normalize_to_mni: true
```
The `--pipeline` parameter is used to specify which pipeline to run. The `--subject_id` and `--session_id` parameters are used to specify the subject and session to be processed. You can run multiple subjects and sessions serially by specifying multiple `--subject_id` and `--session_id` parameters. For example, if you want to run the `wmh_quantification` pipeline for subjects `SUB0001` and `SUB0002`, both at session `01`, you can run: `cvdproc --config_file /mnt/f/BIDS/demo_wmh/code/config.yml --run_pipeline --pipeline wmh_quantification --subject_id 0001 0002 --session_id 01 01`.

The detailed parameters for each pipeline can be found in the respective documentation pages (see below). Parameters except for `subject`, `session`, and `output_path` should be set in the configuration file.

## Currently Available Pipelines

### Structural MRI (sMRI) Pipelines
- [x] Lesion Preprocess ([lesion_analysis](./sMRI/lesion_analysis.md))
- [x] Freesurfer recon-all/recon-all-clinical.sh ([freesurfer](./sMRI/freesurfer.md))
- [x] FSL anat (fsl_anat)
- [x] T1w Registration to MNI space (t1_register)
- [x] Anatomical Segmentation ([anat_seg](./sMRI/anat_seg.md))
- [x] WMH Quantification ([wmh_quantification](./sMRI/wmh_quantification.md))
- [x] PVS Quantification ([pvs_quantification](./sMRI/pvs_quantification.md))
- [x] CMB Quantification (cmb_quantification)

### Diffusion MRI (dMRI) Pipelines
- [x] General DWI Processing ([dwi_pipeline](./dMRI/dwi_pipeline.md))
- [x] Lesion Quantification Toolkit (LQT) Pipeline ([lqt_pipeline](./dMRI/lqt_pipeline.md))

### Functional MRI (fMRI) Pipelines
- [ ] fMRI Pipeline (fmri_pipeline)

### Arterial Spin Labeling (ASL) Pipelines
- [ ] ASL Pipeline (asl_pipeline)

### Quantitative MRI (qMRI) Pipelines
- [x] QSM Pipeline ([qsm_pipeline](./qMRI/qsm_pipeline.md))
- [x] SEPIA QSM (deprecated, archived as a record of the processing used in our paper) ([sepia_qsm](./qMRI/sepia_qsm.md))

### DSC-MRI (PWI) Pipelines
- [x] PWI Pipeline ([pwi_pipeline](./pwi/pwi_pipeline.md))
  