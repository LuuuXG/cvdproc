import os
import subprocess
import nibabel as nib
import time
import shutil
import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str

#############################################
# SHiVAi input preparation (Type: standard) #
#############################################

class PrepareShivaInputInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(desc='subject id', mandatory=True)
    # here we use full <subject_id> rather than sub-<subject_id>
    flair_path = Str(desc='flair path', mandatory=False)
    t1_path = Str(desc='t1 path', mandatory=False)
    swi_path = Str(desc='swi path', mandatory=False)
    output_dir = Str(desc='output directory (have many sub-* subfolders)', mandatory=True)

class PrepareShivaInputOutputSpec(TraitedSpec):
    shiva_input_dir = Directory(desc='shiva input directory') # same as output_dir

class PrepareShivaInput(BaseInterface):
    input_spec = PrepareShivaInputInputSpec
    output_spec = PrepareShivaInputOutputSpec

    def _run_interface(self, runtime):
        subject_id = self.inputs.subject_id
        flair_path = self.inputs.flair_path
        t1_path = self.inputs.t1_path
        swi_path = self.inputs.swi_path
        output_dir = self.inputs.output_dir

        # make sure output_dir exists
        os.makedirs(output_dir, exist_ok=True)

        # create subject folder in output_dir
        subject_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        # COPY corresponding files (if is not None) to subject_dir
        # subfolder: <subject_id>/flair, <subject_id>/t1, <subject_id>/swi
        if flair_path is not None and os.path.exists(flair_path):
            flair_dir = os.path.join(subject_dir, 'flair')
            os.makedirs(flair_dir, exist_ok=True)
            shutil.copy2(flair_path, os.path.join(flair_dir, 'flair.nii.gz'))
        if t1_path is not None and os.path.exists(t1_path):
            t1_dir = os.path.join(subject_dir, 't1')
            os.makedirs(t1_dir, exist_ok=True)
            shutil.copy2(t1_path, os.path.join(t1_dir, 't1.nii.gz'))
        if swi_path is not None and os.path.exists(swi_path):
            swi_dir = os.path.join(subject_dir, 'swi')
            os.makedirs(swi_dir, exist_ok=True)
            shutil.copy2(swi_path, os.path.join(swi_dir, 'swi.nii.gz'))

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['shiva_input_dir'] = self.inputs.output_dir

        return outputs

#######################
# SHiVAi Segmentation #
#######################

# class ShivaSegmentationInputSpec(CommandLineInputSpec):
#     shiva_input_dir = Str(desc='shiva input directory', mandatory=True, argstr='--in %s')
#     shiva_output_dir = Str(desc='shiva output directory', mandatory=True, argstr='--out %s')
#     input_type = Str(desc='input type', mandatory=True, argstr='--input_type %s')
#     prediction = traits.List(desc='prediction', mandatory=True, argstr='--prediction %s...')
#     shiva_config = Str(desc='shiva config file', mandatory=True, argstr='--config %s')
#     brain_seg = Str(desc='brain segmentation', mandatory=True, argstr='--brain_seg %s')

# class ShivaSegmentationOutputSpec(TraitedSpec):
#     shiva_output_dir = Directory(desc='shiva output directory')

# class ShivaSegmentation(CommandLine):
#     _cmd = 'shiva'
#     input_spec = ShivaSegmentationInputSpec
#     output_spec = ShivaSegmentationOutputSpec
#     terminal_output = 'allatonce'

#     def _list_outputs(self):
#         outputs = self.output_spec().get()
#         outputs['shiva_output_dir'] = self.inputs.shiva_output_dir

#         return outputs
    
class ShivaSegmentationInputSpec(BaseInterfaceInputSpec):
    shiva_input_dir = Str(desc='shiva input directory', mandatory=True)
    shiva_output_dir = Str(desc='shiva output directory', mandatory=True)
    input_type = Str(desc='input type', mandatory=True)
    prediction = traits.List(desc='prediction', mandatory=True)
    shiva_config = Str(desc='shiva config file', mandatory=True)
    brain_seg = Str(desc='brain segmentation', mandatory=True)

class ShivaSegmentationOutputSpec(TraitedSpec):
    shiva_output_dir = Directory(desc='shiva output directory')

class ShivaSegmentation(BaseInterface):
    input_spec = ShivaSegmentationInputSpec
    output_spec = ShivaSegmentationOutputSpec

    def _run_interface(self, runtime):
        shiva_input_dir = self.inputs.shiva_input_dir
        shiva_output_dir = self.inputs.shiva_output_dir
        input_type = self.inputs.input_type
        prediction = self.inputs.prediction
        shiva_config = self.inputs.shiva_config
        brain_seg = self.inputs.brain_seg

        # make sure shiva_output_dir exists
        os.makedirs(shiva_output_dir, exist_ok=True)

        # run shiva
        cmd = f'shiva --in {shiva_input_dir} --out {shiva_output_dir} --input_type {input_type} --prediction {" ".join(prediction)} --config {shiva_config} --brain_seg {brain_seg}'
        subprocess.run(cmd, shell=True, check=True)

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['shiva_output_dir'] = self.inputs.shiva_output_dir

        return outputs