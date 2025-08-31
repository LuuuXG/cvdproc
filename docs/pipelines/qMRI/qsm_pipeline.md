# QSM Pipeline

::: cvdproc.pipelines.qmri.qsm_pipeline.QSMPipeline
    options:
      show_signature: false

-----

## A more detailed description:

This pipeline contains the processing steps to compute Quantitative Susceptibility Mapping (QSM) and other scalar maps from multi-echo gradient echo (GRE) data by the following steps:

1. QSM reconstruction using a combined approach from the SEPIA toolbox and Chisep toolbox, including:
    - Phase unwrapping (SEPIA: ROMEO total field calculation)
    - Background field removal using V-SHARP (SEPIA)
    - Dipole inversion using iLSQR (Chisep)