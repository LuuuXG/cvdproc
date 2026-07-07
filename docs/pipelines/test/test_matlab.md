# MATLAB Test Pipeline

::: cvdproc.pipelines.nipype_test.test_matlab.TestMatlabPipeline
    options:
      show_signature: false

----

## A more detailed description:

A test pipeline for running a MATLAB script via Nipype. This pipeline demonstrates the pattern for integrating MATLAB-based processing steps into a Nipype workflow.

### Processing Steps

1. A T1w image is selected from the BIDS session.
2. A MATLAB script (`matlab/test/matlab_test.m`) is executed with the T1w image as input.
3. The placeholder paths in the MATLAB script are replaced with the actual NIfTI file paths at runtime.

### Dependencies

- MATLAB installed and available on `PATH` (or set `matlab_path`)

### Parameters

- `use_which_t1w`: Select a specific T1w image by matching a substring in the filename.
- `matlab_path`: Path to MATLAB executable (default: `'matlab'`).
