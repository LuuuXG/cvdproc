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
   - **Python 3.10** is recommended for general use. Some legacy or deep-learning based pipelines may require separate environments with their own Python/CUDA/TensorFlow/PyTorch versions.
   - External neuroimaging tools: **FSL**, **FreeSurfer**, **ANTs**, **MATLAB**, **Docker**, and **MRtrix3**.

As an example using WSL2 with Ubuntu 22.04, you can create a new conda environment using the following commands (assuming that conda or miniconda is already installed):
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
sudo apt update
sudo apt install build-essential python3-dev python3-pip
```

These external tools are not installed by `pip`. Please install them separately and make sure their command-line tools are available in your shell environment. Some specific pipelines may require additional tools, such as **SPM**, **DSI Studio**, **dcm2niix**, or Singularity/Apptainer.

### 1. Clone the repository or download it directly

### 2. Install via `pip`:
```bash
# Use -e to allow modification of the code without needing to reinstall.
# Here </path/to/cvdproc> is the folder containing `pyproject.toml`
pip install -e /path/to/cvdproc

# If it is necessary to use mirror
pip install -e /path/to/cvdproc -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Optional dependency groups can be installed when needed. For example:

```bash
pip install -e "/path/to/cvdproc[preprocess]"
pip install -e "/path/to/cvdproc[quality]"
pip install -e "/path/to/cvdproc[analysis,visualization]"
```

Available optional groups include `preprocess`, `quality`, `visualization`, and `analysis`. See the installation documentation for details.

### 3. Data Files

The `cvdproc/data/` directory is excluded from the repository due to its large size. Please download the required data files from the following link: [Download data.zip](https://drive.google.com/file/d/1hFVFhlc0BE4_db81LN7yEU-JIhXySN85/view?usp=sharing)

After downloading, unzip and place the contents into `./cvdproc/data/` (the same level with `pipelines`)

## Usage
Please refer to the [documentation](https://LuuuXG.github.io/cvdproc) (🚧 under construction).

If you are interested in reproducing our analysis process, we recommend starting from DICOM data. This is because the handling of BIDS format in the code may not be fully standard and compliant (for example, we require a session level and include non-standard suffixes).

## Citation
Please cite the corresponding papers mentioned in the documentation when using this package for your research.
