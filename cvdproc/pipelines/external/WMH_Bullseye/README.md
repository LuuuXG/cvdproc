# Introduction

This is an example code to apply Bullseye WMH parcellation as explained here: 

- 10.1016/j.neurad.2017.10.001 
- 10.1038/s41598-022-06019-8 

I used this code to generate the segmentations in the following manuscript:

Jiménez-Balado J, Corlier F, Habeck C, Stern Y, Eich T. Effects of white matter hyperintensities distribution and clustering on late-life cognitive impairment. Sci Rep. 2022 Feb 4;12(1):1955. doi: 10.1038/s41598-022-06019-8. PMID: 35121804; PMCID: PMC8816933.

# Dependencies

## Software

You need a UNIX environment with the following softwares: 

- Freesurfer.
- Python 3
- R (only for obtaining summary statistics)

## Python dependencies 

- os 
- numpy
- nibabel
- sys
- scipy
- matplotlib

## R dependencies: 

- oro.nifti
- neurobase

# Use

## Previous steps

Before running the code you need to do the following steps: 

1. Running Freesurfer on your subjects. 
2. Using some algorithm to segment WMH (binary mask). 

## Main script

The main script is located at the project's root (`bullseye_pipeline.sh`). And only takes two parameters: 

- 1: Text file containing a list of patients to be processed. 
- 2: Source location

`bash bullseye_pipelines.sh my_list_subjects.txt /my/folder/to/Bullseye_WMH`

## Steps in the pipeline

### Concentric parcellation

This step depends on the: cortex ribbon segmentation (`ribbon.nii.gz`) and FS' aseg (`aseg.nii.gz'). The software calculates the distance from the ventricles to cortex. 
Basically, it calculates the euclidean distance transform from ventricles and cortex mask, and then calculates: `dist_orig / (dist_orig + dist_dest)`.

After this, you should see a gradient of intensities fron ventricles to cortex. 

### Lobal parcellation

This step uses several freesurfer functions. First merges aparc parcellation to lobes (frontal, temporal, occipital and parietal). Then extend these cortical segmentations to the white matter using `mri_aparc2aseg`. Finally, it uses a python script to assign some pending regions. 

### Intersection

After these steps, intersect both masks.

 


