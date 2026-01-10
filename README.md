# CVDProc: CerebroVascular Disease imaging Processing

> **⚠️ Important Notice**
>
> This package is intended for research purposes only, to facilitate reproducibility of neuroimaging analyses in our center.
> The authors do **NOT** guarantee the correctness or clinical validity of all processing workflows.

Here is the documentation for processing codes used in various research projects.

----

## About
Brain Imaging Processing for the Cerebrovascular Disease Research Cohort of the Department of Neurology, West China Hospital, Sichuan University (Stroke, Cerebral Small Vessel Disease, Atrial Fibrillation, Community Cohort, etc.)

This repository currently serves as a public display for MRI image preprocessing and analysis, aimed at enhancing the transparency and reproducibility of research conducted at our center.

Our imaging data involves:
- Lesion present/lesion absent
- Clinical imaging/research imaging
- 3T/7T

## Installation
### 0. System Requirements and Dependencies:

#### Required

   - **Linux** environment (This package has been tested on WSL2 with Ubuntu22.04)
   - **Python 3.7** (Due to the features of the [SHIVA model](https://github.com/pboutinaud/SHIVA_PVS), higher versions of Python are not supported. If you wish to run the parts of the code related to PVS and CMB segmentation, we recommend creating a Python environment with version 3.7.)

As an example using WSL2 with Ubuntu 22.04, you can create a new conda environment using the following commands (assuming that conda or miniconda is already installed):
```bash
conda create -n <env_name> python=3.7 openssl=1.1.1
conda activate <env_name>
sudo apt update
sudo apt install build-essential python3-dev python3-pip
```

#### Optional
   - **Matlab**
   - **Docker**
   - Common neuroimaging tools: **Freesurfer**, **FSL**, **SPM**, **ANTS**.

These tools are optional but **strongly recommended**. Although installing the software listed above is not required for installing this package via `pip`, many workflows rely on them — especially FSL and Freesurfer. Different pipelines may depend on different tools, so having them pre-installed will ensure full functionality.

### 1. Clone the repository or download it directly

### 2. Install via `pip`:
```
# Use -e to allow modification of the code without needing to reinstall.
# Here </path/to/cvdproc> is the folder containing `setup.py`
pip install -e /path/to/cvdproc

# If it is necessary to use mirror
pip install -e /path/to/cvdproc -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. Data Files

The `cvdproc/data/` directory is excluded from the repository due to its size. Please download the required data files from the following link: [Download data.zip](https://drive.google.com/file/d/1VkCxPL4eNn8vWRegL4IaBbfHsNymNF-7/view?usp=sharing)

After downloading, unzip and place the contents into `./cvdproc/data/` (the same level with `pipelines`)

## Usage
Please refer to the [documentation](https://LuuuXG.github.io/cvdproc) (🚧 under construction).

If you are interested in reproducing our analysis process, we recommend starting from DICOM data. This is because the handling of BIDS format in the code may not be fully standard and compliant (for example, we require a session level and include non-standard suffixes).

## Citation
Please cite the corresponding papers mentioned in the documentation when using this package for your research.