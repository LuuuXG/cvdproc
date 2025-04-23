import os
import subprocess
import nibabel as nib
import numpy as np
import json

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function

from cvdproc.pipelines.dmri.freewater.single_shell_freewater import SingleShellFW

single_shell_fw = SingleShellFW()

input_directory = '/mnt/e/Codes/cvdproc/cvdproc/pipelines/external/DiffusionTensorImaging/data/input'
fdwi = input_directory + "/dti.nii.gz"
fmask = input_directory + "/nodif_brain.nii.gz"
fbval = input_directory + "/bvals"
fbvec = input_directory + "/bvecs"
working_directory = '/mnt/e/Codes/cvdproc/cvdproc/pipelines/external/DiffusionTensorImaging/data/output'
output_directory = '/mnt/e/Codes/cvdproc/cvdproc/pipelines/external/DiffusionTensorImaging/data/output'

single_shell_fw.inputs.fdwi = fdwi
single_shell_fw.inputs.fbval = fbval
single_shell_fw.inputs.fbvec = fbvec
single_shell_fw.inputs.mask_file = fmask
single_shell_fw.inputs.working_directory = working_directory
single_shell_fw.inputs.output_directory = output_directory

single_shell_fw.run()