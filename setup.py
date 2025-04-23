from setuptools import setup, find_packages

setup(
    name="cvdproc",
    version="0.0.1",
    author="Youjie Wang",
    author_email="wangyoujie2002@163.com",
    description="CVDProc: CerebroVascular Disease imaging Processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    #url="https://github.com/your-repo/neurovasc",  # TODO
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas",
        "scipy>=1.7.0",
        "matplotlib",
        "scikit-image>=0.18.0",
        "nipype",
        "packaging",
        "dipy",
        "pybids",
        "nibabel>=3.2.0",
        #"nilearn",
        "pyyaml>=5.4.0",
        "xlwt>=1.3.0",
        "scienceplots>=1.0.9",
        #"tensorflow=2.13.1",
        #"torch",
        #"dcm2niix",
        "dcm2bids",
        "openpyxl"
    ],
    entry_points={
        "console_scripts": [
            "cvdproc=cvdproc.main:main_entry", # main entry
            "cvdproc_tool=cvdproc.utils.python.tool_manager:main", # tools
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
