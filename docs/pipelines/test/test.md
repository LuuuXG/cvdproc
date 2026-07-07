# Test Pipeline

::: cvdproc.pipelines.nipype_test.test.TestPipeline
    options:
      show_signature: false

----

## A more detailed description:

A minimal Nipype pipeline for testing purposes. This pipeline creates a simple workflow that outputs a text file containing the subject and session IDs. It is useful for:

- Verifying that the Nipype environment is correctly configured
- Testing new BIDS data ingestion
- Debugging workflow connection issues

### Modalities

- None required (always passes `check_data_requirements`)
