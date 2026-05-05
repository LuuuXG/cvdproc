# Installation

## Download

You can download the source code by cloning the GitHub repository:
```bash
git clone https://github.com/LuuuXG/cvdproc.git
```
Or download it manually from the [GitHub homepage](https://github.com/LuuuXG/cvdproc).

The `cvdproc/data/` directory is excluded from the repository due to its large size. Please download the required data files from the following link: [Download data.zip](https://drive.google.com/file/d/1hFVFhlc0BE4_db81LN7yEU-JIhXySN85/view?usp=sharing)

After downloading, unzip and place the contents into `./cvdproc/data/` (the same level with `pipelines`)

## Installation
Please create a new conda environment, which is more suitable for this package.

```bash
# Please replace <env_name> with the name you want for the environment
conda create -n <env_name> python=3.10
conda activate <env_name>
```

!!! note "Python Version"
    Python 3.10 is recommended for general use. Some legacy deep-learning models or external tools may require their own Python environment. For example, early versions of the [SHIVA model](https://github.com/pboutinaud/SHIVA_PVS) used older TensorFlow/Python combinations, while newer tools such as [SHiVAi](https://github.com/pboutinaud/SHiVAi) and [LST-AI](https://github.com/CompImg/LST-AI) may require different versions. If a pipeline has strict requirements, please create a separate environment for that pipeline.

### Required System Dependencies

CVDProc wraps many external neuroimaging tools. The following software should be installed and configured before running most pipelines:

- **FSL**
- **FreeSurfer**
- **ANTs**
- **MATLAB**
- **Docker**
- **MRtrix3**

These tools are not installed by `pip`. Please install them separately and make sure their command-line tools are available in your shell environment. For example, FSL, FreeSurfer, ANTs, and MRtrix3 should be available from `PATH`; FreeSurfer also requires a valid license file; MATLAB-based pipelines require a working MATLAB installation.

Some specific pipelines may require additional tools, such as **DSI Studio**, **dcm2niix**, **SPM**, or Singularity/Apptainer. Please check the documentation of the specific pipeline before running it.

Then, navigate to the directory where you downloaded the code (the folder containing `pyproject.toml`), and run the following command:

```bash
# Use -e to allow modification of the code without needing to reinstall.
# Here </path/to/cvdproc> is the folder containing `pyproject.toml`
pip install -e /path/to/cvdproc

# If it is necessary to use mirror
pip install -e /path/to/cvdproc -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Optional Python Dependencies

The base installation only includes dependencies that are commonly needed by the main package and most pipelines. Some functions depend on large or specialized packages, so they are provided as optional dependency groups in `pyproject.toml`.

Install optional groups only when you need the corresponding functions:

```bash
# Preprocessing, segmentation, image conversion, and additional model fitting
pip install -e "/path/to/cvdproc[preprocess]"

# QC tools, such as the FreeSurfer QC wrapper
pip install -e "/path/to/cvdproc[quality]"

# Visualization and surface/brain map plotting
pip install -e "/path/to/cvdproc[visualization]"

# Network and control-theory analysis
pip install -e "/path/to/cvdproc[analysis]"
```

You can install multiple groups at once:

```bash
pip install -e "/path/to/cvdproc[preprocess,quality,visualization]"
```

!!! note "Quotes around extras"
    The quotation marks are recommended, especially when using shells such as `zsh`, because square brackets may otherwise be interpreted by the shell.

!!! note "Large preprocessing dependencies"
    The `preprocess` group includes packages such as `tensorflow`, `torch`, and `monai`, because some preprocessing/segmentation workflows rely on deep-learning models. These packages are large and may need a CUDA-specific installation. For GPU workflows, it is often better to install the correct PyTorch/TensorFlow build manually according to your CUDA driver, then install `cvdproc` without reinstalling these packages.
