# Structural Covariance Network

::: cvdproc.pipelines.smri.scn_pipeline.SCNPipeline
    options:
      show_signature: false

----

## A more detailed description:

This pipeline computes Structural Covariance Network (SCN) measures from FreeSurfer-derived cortical features.

### Methods

Currently, only the **MIND** (Morphometric Inverse Divergence) method is supported. MIND computes structural covariance between cortical regions using multiple morphometric features from FreeSurfer's `aparc` parcellation, including:

- Cortical thickness (CT)
- Mean curvature (MC)
- Gray matter volume (Vol)
- Surface area (SD)
- Sulcal depth (SA)

### Dependencies

- FreeSurfer `recon-all` or `recon-all-clinical.sh` results are required.
- Set `use_freesurfer_clinical=True` if using `recon-all-clinical.sh` outputs.

### References

\bibliography
