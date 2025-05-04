# dcm2bids

We are using the [dcm2bids](https://unfmontreal.github.io/Dcm2Bids/3.1.1/) to convert DICOM files to BIDS format. Before running the code, please make sure you have installed the `dcm2bids` (it should have been installed when you installed the cvdproc) and `dcm2niix` (you can install it via `apt install dcm2niix`)

## Create a new BIDS dataset

If you want to create a new BIDS dataset, you can use the following command:

```bash
cvdproc --run_initialization <path/to/the/folder/you/want/to/create>
```

You don't need to create the folder manually, the code will create it for you.

## Convert DICOM to BIDS

If you already have a BIDS root folder or just created one with the command above, you can follow the steps below to convert DICOM files to BIDS format.

### Create a dcm2bids configuration file

Please refer to the official dcm2bids documentation [How to create a configuration file](https://unfmontreal.github.io/Dcm2Bids/3.2.0/how-to/create-config-file/) for the most detailed guidance.

Here we take the example of converting a 3D T1w image to BIDS format.

Create a file named `dcm2bids_config.json` in the `code` folder of your BIDS root directory (the file name and location can be changed, but we name it this way for convenience). The content of the file is as follows:

```json
{
  "descriptions": [
    {
      "datatype": "anat",
      "suffix": "T1w",
      "criteria": {
        "SeriesDescription": "*mprage*",
      }
    }
  ]
}
```

The most important part of the configuration file is the `criteria` field, which specifies how to match the DICOM files. In this case, we are matching the `SeriesDescription` field with a regular expression `*mprage*`. If your DICOM files do not have this field or have a different value, you can try to use the `dcm2bids_helper` command to get the information, or you can use `dcm2niix` (which can also be found in MRIcroGL) to get the information in the JSON file. **You can also skip this step (just copy the content above) and wait for the next step to see how we solve it.**

### Create a cvdproc configuration file

!!! info "cvdproc config file"
    This step is very important because it is the core of the `cvdproc` command to specify parameters.

