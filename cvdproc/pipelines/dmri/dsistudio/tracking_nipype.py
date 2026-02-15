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
from traits.api import Bool, Int, Str, Float, Either

from cvdproc.config.paths import get_package_path

dsi_studio_path = get_package_path('data', 'dsi-studio', 'dsi_studio')

class DSIstudioTrackingInputSpec(CommandLineInputSpec):
    # core
    source = File(exists=True, desc='Path to your .fib.gz (or .fz) file.', argstr='--source=%s', mandatory=True)
    output = Str(desc='Output tractography file (e.g. --output=tract.tt.gz).', argstr='--output=%s')
    track_id = Str(desc='Track ID used for autotrack', argstr='--track_id=%s')
    thread_count = Int(1, desc='Number of threads to use', argstr='--thread_count=%d')
    # tracking parameters
    method = Int(desc='Tracking algorithm. 0=streamline, 1=RK4, etc.', argstr='--method=%d')
    tract_count = Int(desc='Number of tracts to generate', argstr='--tract_count=%d')
    seed_count = Int(desc='Number of seeds to use', argstr='--seed_count=%d')
    track_voxel_ratio = Float(desc='Seeding density as a ratio of total voxel count', argstr='--track_voxel_ratio=%f')
    turning_angle = Float(desc='Maximum turning angle in degrees', argstr='--turning_angle=%f')
    step_size = Float(desc='Step size in mm', argstr='--step_size=%f')
    smoothing = Float(desc='Smoothing factor', argstr='--smoothing=%f')
    min_length = Float(desc='Minimum fiber length in mm', argstr='--min_length=%f')
    max_length = Float(desc='Maximum fiber length in mm', argstr='--max_length=%f')
    otsu_threshold = Float(desc='Default threshold for FA-based seeding.', argstr='--otsu_threshold=%f')
    threshold_index = Str(desc='Use a different diffusion index (e.g., QA) for termination instead of FA.', argstr='--threshold_index=%s')
    check_ending = Bool(desc='Off for whole-brain tracking', argstr='--check_ending')
    parameter_id = Str(desc='Parameter ID from previous tracking session to replicate parameters.', argstr='--parameter_id=%s')
    random_seed = Int(desc='Random seed for reproducibility', argstr='--random_seed=%d')
    # region options
    seed = Str(desc='Seeding region file (e.g., --seed=seed.nii.gz)', argstr='--seed=%s')
    roi = Str(desc='ROI file for inclusion (e.g., --roi=roi.nii.gz)', argstr='--roi=%s')
    roa = Str(desc='ROA file for exclusion (e.g., --roa=roa.nii.gz)', argstr='--roa=%s')
    end = Str(desc='End region file (e.g., --end=end.nii.gz)', argstr='--end=%s')
    ter = Str(desc='Termination region file (e.g., --ter=ter.nii.gz)', argstr='--ter=%s')
    nend = Str(desc='N-end region file (e.g., --nend=nend.nii.gz)', argstr='--nend=%s')
    lim = Str(desc='Limiting region file (e.g., --lim=lim.nii.gz)', argstr='--lim=%s')
    # additional args
    args = Str(desc='Additional command-line arguments', argstr='%s')

class DSIstudioTrackingOutputSpec(TraitedSpec):
    output = Str(desc='Output tractography file.')

class DSIstudioTracking(CommandLine):
    _cmd = dsi_studio_path + ' --action=trk'
    input_spec = DSIstudioTrackingInputSpec
    output_spec = DSIstudioTrackingOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output'] = self.inputs.output
        
        return outputs
