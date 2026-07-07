# NeMo Postprocessing Pipeline

::: cvdproc.pipelines.dmri.nemo_postprocess_pipeline.NemoPostprocessPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline postprocesses outputs from the [NeMo](https://github.com/KUL-Radneuron/NeMo) toolbox, which computes network-based measures from structural connectivity data. The pipeline extracts and aggregates cortical metrics, subcortical volumes, and structural connectivity matrices for population-level analysis.

### Dependencies

- NeMo output directory (auto-discovered via BIDS derivatives)
- FreeSurfer `recon-all`, `recon-all-clinical.sh`, or longitudinal FreeSurfer results for cortical metrics extraction

### Processing Options

#### Cortical Metrics (`cortical_metrics=True`)

Extracts weighted cortical metrics (weighted degree, betweenness centrality, clustering coefficient, etc.) from NeMo's output. Requires FreeSurfer results.

#### ChacoVol / ChacoConn (`results_to_csv=True`)

Converts NeMo's ChacoVol (subcortical volumes per atlas) and ChacoConn (structural connectivity matrices) outputs to CSV format for group-level analysis.

### Parameters

- `use_freesurfer_clinical`: Set to `True` if using `recon-all-clinical.sh` outputs instead of standard `recon-all`.
- `use_freesurfer_longitudinal`: Set to `True` if using longitudinal FreeSurfer outputs.
- `cortical_metrics`: Whether to extract cortical metrics from NeMo outputs (default: `False`).
- `results_to_csv`: Whether to convert ChacoVol/ChacoConn results to CSV (default: `False`).
- `extract_from`: Path to extract results from for population-level summary.
