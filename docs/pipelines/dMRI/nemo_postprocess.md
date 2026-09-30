# NeMo Postprocessing Pipeline

::: cvdproc.pipelines.dmri.nemo_postprocess_pipeline.NemoPostprocessPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline postprocesses outputs from the [NeMo](https://github.com/KUL-Radneuron/NeMo) toolbox, which computes network-based measures from structural connectivity data. Postprocessed files use BIDS-style subject/session, atlas, tractography-model, space, hemisphere, density, description, and statistic entities.

The unsmoothed mean ChacoVol image is always projected from MNI152 to the left and right fsaverage 164k surfaces. Both `ifod2act` and `sdstream` outputs are discovered automatically; either method can be present on its own. Smoothed maps are not projected.

### Dependencies

- NeMo output directory (auto-discovered via BIDS derivatives)
- FreeSurfer `recon-all`, `recon-all-clinical.sh`, or longitudinal FreeSurfer results only when weighted cortical metric extraction is enabled

### Processing Options

#### Surface Projection (always enabled)

Projects each available unsmoothed mean ChacoVol map to fsaverage 164k and masks the medial wall. This step does not require FreeSurfer subject results.

#### FreeSurfer Metrics (`cortical_metrics=True`)

Extracts ChacoVol-weighted metrics from the optional FreeSurfer outputs. Surface projection still runs when this option is `False`.

#### ChacoVol / ChacoConn (`results_to_csv=True`)

Converts every discovered ChacoVol atlas and ChacoConn atlas to CSV independently. Atlas counts do not need to match. Packaged labels are used for AAL116, FreeSurfer86, and Shen268; other atlases use a matching CSV/TSV label table when available and otherwise receive numeric ROI labels.

### Parameters

- `use_freesurfer_clinical`: Set to `True` if using `recon-all-clinical.sh` outputs instead of standard `recon-all`.
- `use_freesurfer_longitudinal`: Set to `True` if using longitudinal FreeSurfer outputs.
- `cortical_metrics`: Whether to extract optional ChacoVol-weighted FreeSurfer metrics (default: `False`).
- `results_to_csv`: Whether to convert ChacoVol/ChacoConn results to CSV (default: `False`).
- `extract_from`: Path to extract results from for population-level summary.
