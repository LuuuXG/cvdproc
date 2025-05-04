# Installation

## Download

You can download the source code by cloning the GitHub repository:
```bash
git clone https://github.com/LuuuXG/cvdproc.git
```
Or download it manually from the [GitHub homepage](https://github.com/LuuuXG/cvdproc).

## Installation
Please create a new conda environment, which is more suitable for this package.

```bash
# Please replace <env_name> with the name you want for the environment
conda create -n <env_name> python=3.7 openssl=1.1.1
conda activate <env_name>
```

!!! note "Python Version"
    Due to the features of the early version of  [SHIVA model](https://github.com/pboutinaud/SHIVA_PVS), python 3.7 is used for tensorflow compatibility. We are working on updating the code to support higher versions of Python by using [SHiVAi](https://github.com/pboutinaud/SHiVAi). For example, [LST-AI](https://github.com/CompImg/LST-AI) for WMH segmentation needs python 3.8 or higher. Currently, the solution is to use a different environment for different pipelines if necessary.

Then, navigate to the directory where you downloaded the code (the folder containing `setup.py`), and run the following command:

```
# Use -e to allow modification of the code without needing to reinstall.
# Here </path/to/cvdproc> is the folder containing `setup.py`
pip install -e /path/to/cvdproc

# If it is necessary to use mirror
pip install -e /path/to/cvdproc -i https://pypi.tuna.tsinghua.edu.cn/simple
```

!!! note "tensorflow and torch"
    Because these two packages are large, we do not specify them in `setup.py`. However, they will be used in the subsequent code. Please install them as needed.