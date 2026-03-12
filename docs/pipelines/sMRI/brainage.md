# Brain Age Estimation

::: cvdproc.pipelines.smri.brainage_pipeline

-----

## A more detailed description:

Currently, only the `brainageR` method is supported for brain age estimation. This choice is based on Dorfel et al. [@dorfel2023prediction] which reported that 'pyment and
brainageR consistently showed the highest accuracy and test–retest reliability'.

### installation of brainageR

`brainageR` can be installed following the [official documentation](https://github.com/james-cole/brainageR). It is also included in the `cvdproc` package (in `data` folder, including the rds files), but the lines 66-70 in the brainageR folder need to be modified to specify the correct paths.

### References

\bibliography