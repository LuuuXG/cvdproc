import os
import subprocess
import shutil
import nibabel as nib
import time
import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str

import logging
logger = logging.getLogger(__name__)

class Synb0InputSpec(BaseInterfaceInputSpec):
    t1w_img = File(desc="Path to the T1-weighted image")
    dwi_img = File(desc="Path to the DWI image. First volume is b0 image")
    output_path_synb0 = Directory(desc="Output directory for the synb0 image")
    phase_encoding_number = Str(desc="Phase encoding number")
    total_readout_time = Str(desc="Total readout time")

class Synb0OutputSpec(TraitedSpec):
    acqparam = File(desc="Path to the acqparam.txt file")
    b0_all = File(desc="Path to the b0_all image")

class Synb0(BaseInterface):
    input_spec = Synb0InputSpec
    output_spec = Synb0OutputSpec

    def _run_interface(self, runtime):
        synb0_input_path = os.path.join(self.inputs.output_path_synb0, "INPUTS")
        synb0_output_path = os.path.join(self.inputs.output_path_synb0, "OUTPUTS")
        b0_all_path = os.path.join(synb0_output_path, "b0_all.nii.gz")

        if not os.path.exists(b0_all_path):
            os.makedirs(synb0_input_path, exist_ok=True)
            os.makedirs(synb0_output_path, exist_ok=True)

            # mri_synthstrip t1w_img
            subprocess.run(['mri_synthstrip', '-i', self.inputs.t1w_img, '-o', os.path.join(synb0_input_path, 'T1.nii.gz')], check=True)

            # extract b0 image from dwi_img
            subprocess.run(['fslroi', self.inputs.dwi_img, os.path.join(synb0_input_path, 'b0.nii.gz'), '0', '1'], check=True)

            # Create a acqparam.txt file in INPUTS
            with open(os.path.join(synb0_input_path, 'acqparam.txt'), 'w') as f:
                f.write(self.inputs.phase_encoding_number + ' ' + str(self.inputs.total_readout_time) + '\n')
                f.write(self.inputs.phase_encoding_number + ' 0')
            
            # synb0-disco
            # first check the docker image is available
            docker_image = "leonyichencai/synb0-disco:v3.1"
            try:
                subprocess.run(['docker', 'inspect', docker_image], check=True)
            except subprocess.CalledProcessError:
                logger.error(f"Docker image {docker_image} not found. Please pull the image first. \n docker pull {docker_image}")
                return runtime
            
            fs_license = os.environ.get("FS_LICENSE")
            if not fs_license:
                raise ValueError("FS_LICENSE environment variable is not set.")
            
            subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{synb0_input_path}:/INPUTS",
                "-v", f"{synb0_output_path}:/OUTPUTS",
                "-v", f"{fs_license}:/extra/freesurfer/license.txt",
                "leonyichencai/synb0-disco:v3.1",
                "--user", "1000:1000",
                "--stripped",
                "--notopup"
            ], check=True)
        else:
            logger.info(f"b0_all.nii.gz already exists at {b0_all_path}. Skipping synb0-disco.")
        
        b0_u = os.path.join(synb0_output_path, 'b0_u.nii.gz')
        b0_d_smooth = os.path.join(synb0_output_path, 'b0_d_smooth.nii.gz')

        # merge the synthetic b0 image with the original dwi image
        subprocess.run(['fslmerge', '-t', os.path.join(self.inputs.output_path_synb0, 'b0_all.nii.gz'), b0_d_smooth, b0_u], check=True)

        # output
        self._acqparam = os.path.join(synb0_input_path, 'acqparam.txt')
        self._b0_all = os.path.join(self.inputs.output_path_synb0, 'b0_all.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["acqparam"] = os.path.abspath(self._acqparam)
        outputs["b0_all"] = os.path.abspath(self._b0_all)
        
        return outputs