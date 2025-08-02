# LQT Pipeline

::: cvdproc.pipelines.dmri.lqt_pipeline.LQTPipeline
    options:
      show_signature: false

----

## A more detailed description:

Source: [LQT](https://github.com/jdwor/LQT)

!!! Note
    LQT is implemented in R, so you need to have R installed on your system. Using R in WSL seems to have some instability, try changing the cores=4 parameter in 'cvdproc/pipelines/r/lqt_single_subject.R'.
  
I recommend using the [RStudio](https://www.rstudio.com/products/rstudio/download/) IDE (because we use rstudioapi in the script) and open 'cvdproc/pipelines/r/lqt_initialize.r' to initialize the LQT package: As the package is no longer being actively maintained, we included the package in 'cvdproc/pipelines/external/LQT' and we can install it from there. Then we need to copy some extension data to the package directory (data is in 'cvdproc/data/lqt/extdata'), which is done in the script. Doing this by following instructions in the [LQT README](https://github.com/jdwor/LQT) seems incorrect as it will download a MacOS version of DSI Studio, which is not compatible with WSL Linux.