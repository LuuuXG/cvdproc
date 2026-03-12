# Hipsta

::: cvdproc.pipelines.smri.hipsta_pipeline

-----

## A more detailed description:

[Hipsta](https://deep-mi.org/hipsta/dev/index.html)

### Installation

```
pip install hipsta
```

Other considerations:

- only gmsh version 2.x is supported (do not use gmsh 4.x or later).

```
cd ~
wget https://gmsh.info/bin/Linux/gmsh-2.16.0-Linux64.tgz

tar -xzf gmsh-2.16.0-Linux64.tgz

echo 'export PATH=$HOME/gmsh-2.16.0-Linux/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

- Newer versions of scipy may cause issues: `ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 0 dimension(s)`. A python script to fix this issue is provided in `<cvdproc>/pipelines/smri/hipsta/patch_hipsta_mode.py`.