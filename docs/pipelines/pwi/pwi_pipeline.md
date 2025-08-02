# PWI Pipeline

::: cvdproc.pipelines.pwi.pwi_pipeline.PWIPipeline
    options:
      show_signature: false
    
-----

## A more detailed description:

1. Strip the PWI data using SynthStrip to create a brain mask.

2. Calculate the PWI concentration map (a 4D image).

3. Auto select the AIF (arterial input function) from the concentration data. (Please refer to [Auto AIF Selection](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/blob/develop/src/original/JBJA_GUSahlgrenska_SWE/AIF_selection_auto/AIF_selection_automatic.py))

4. Generate the perfusion maps (rCBF, rCBV, MTT, TTP, K2) using the MATLAB-based [dsc-mri-toolbox](https://github.com/FAIR-Unipd/dsc-mri-toolbox).