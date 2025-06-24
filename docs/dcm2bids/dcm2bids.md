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

Create a `config.yml` file in the `code` folder (note that we use the yaml format), and the content is as follows:

```yml
# You need to specify the BIDS root folder here
bids_dir: /mnt/f/BIDS/demo_wmh
## You need to specify the dcm2bids configuration file here
dcm2bids: /mnt/f/BIDS/demo_wmh/code/dcm2bids_config.json
```

Remember to change the `bids_dir` and `dcm2bids` paths to your own paths. The `bids_dir` is the root folder of your BIDS dataset, and the `dcm2bids` is the path to the dcm2bids configuration file you just created. And then you need to move or copy the folder containing the DICOM files for a single subject into the `sourcedata` folder of the BIDS root directory. For example, if you have a folder named `DICOM_01` containing the DICOM files for a single subject's baseline acquisition, you need to move or copy it to `/mnt/f/BIDS/demo_wmh/sourcedata/DICOM_01`. The final folder structure should look like this:

!!! info DICOM folder
    Because the original DICOM images obtained in actual research may have different structures, the file structure under the subject's DICOM folder may vary. However, it should be noted that there is no need to preprocess the subfolders under the DICOM folder in advance (for example, a common practice is to make one subfolder correspond to one scanning sequence). This is because `dcm2bids` will convert all DICOM files found under the folder, even if only a few sequences are specified in the json file (so, for example, when EPI sequences with DWI or multi-echo GRE sequences are included, the conversion time may take several minutes, but fortunately, theoretically, such conversion only needs to be done once).

After the above preparation, you need to specify the subject ID and session ID for the converted subject. For example, if the DICOM files are stored in `DICOM_01`, you need to set the subject ID to `SUB0001` and the session ID to `01` to indicate baseline data. Run:

```
cvdproc --config_file /mnt/f/BIDS/demo_wmh/code/config.yml \
  --run_dcm2bids \
  --subject_id SUB0001 --session_id 01 \
  --dicom_subdir DICOM_01
```

Theoretically, the folder `/mnt/f/BIDS/demo_wmh/sub-SUB0001` should be created to store the subject's data. However, since we did not check the `SeriesDescription` field of the DICOM files in advance, it should prompt that no matching files were found, and the subject folder was not created. Next, we can open the `tmp_dcm2bids` folder to check the output of `dcm2bids`, find the json file of the image of interest, and then find the `SeriesDescription` field (or other fields you want to match) to modify the corresponding content in `dcm2bids_config.json`, and run the above command again.

At this time, it should be able to successfully obtain the `sub-SUB0001` folder, which contains the `ses-01` subfolder. Because `dcm2bids` will automatically look for images that meet the criteria under `tmp_dcm2bids`, instead of converting all images again.

## .bidsignore

We can notice that there is a `.bidsignore` file in the bids root directory, which is used to ignore files that do not belong to the BIDS format. If we open it, we will find that it contains the `tmp_dcm2bids` folder, which is the temporary folder generated when we run the dcm2bids command. The `dcm2bids` will automatically add it to the `.bidsignore`.

It is worth noting that `dcm2bids` itself is very flexible and allows the generation of folders that do not meet the BIDS requirements (for example, I want to change the datatype and suffix in the json file to qsm and GRE respectively, which I believe is not currently specified in the BIDS specification, but this is more convenient for organizing data). In this case, we need to manually add these folders to the `.bidsignore` (for example: sub-\*/ses-\*/qsm/ to ignore each qsm folder). This is very necessary because various nipreps (such as fmriprep, qsiprep) include a check bids validation step (although it can be skipped, it is not recommended to do so, otherwise it is easy to have no error but the running process has problems), and these folders that do not meet the standard definition will cause errors.

In addition, the BIDSLayout function of nipype seems to automatically exclude folders that do not meet the BIDS definition, regardless of whether they are added to the `.bidsignore`, which is also why we did not use it.