import os
import shutil
import subprocess
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge
from nipype.interfaces.base import (CommandLine, traits, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec, File)

class TestMatlabInputSpec(BaseInterfaceInputSpec):
    input_file = File(exists=True, mandatory=True, desc='Input file for the MATLAB script')
    output_dir = traits.Str(mandatory=True, desc='Output directory for the MATLAB script results')
    script_path = File(exists=True, desc='Path to the MATLAB script to be executed')

class TestMatlabOutputSpec(TraitedSpec):
    output_dir = traits.Str(exists=True, desc='Output directory containing the results of the MATLAB script')

class TestMatlab(BaseInterface):
    input_spec = TestMatlabInputSpec
    output_spec = TestMatlabOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(self.inputs.output_dir, exist_ok=True)
        # Load script
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()

        # Replace the placeholders in the script
        script_path = os.path.join(self.inputs.output_dir, 'test.m')
        script_content = script_content.replace('placeholder/for/nipype/input_image', self.inputs.input_file)
        script_content = script_content.replace('placeholder/for/nipype/output_dir', self.inputs.output_dir)
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)
        
        cmd_str = f"run('{script_path}'); exit;"
        mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"{cmd_str}\"", terminal_output='stream')

        result = mlab.run()

        return result.runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_dir'] = self.inputs.output_dir
        return outputs

class TestMatlabPipeline:
    def __init__(self, subject, session, output_path, matlab_path=None, **kwargs):
        """
        Test pipeline for running a MATLAB script using Nipype.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.matlab_path = matlab_path if matlab_path else 'matlab'

        self.use_which_t1w = kwargs.get('use_which_t1w', None)
    
    def check_data_requirements(self):
        # Check if the required data is available
        return self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        if self.session.get_t1w_files():
            t1w_files = self.session.get_t1w_files()
            if self.use_which_t1w:
                t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
                if len(t1w_files) != 1:
                    #raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
                    print('No T1w matched')
                    t1w_file = ''
                else:
                    t1w_file = t1w_files[0]
            else:
                t1w_files = [t1w_files[0]]
                t1w_file = t1w_files[0]
                print(f"No specific T1w file selected. Using the first one: {t1w_file}.")
        else:
            t1w_file = None
        
        test_matlab_wf = Workflow(name='test_matlab_wf')
        test_matlab_wf.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows')
        inputnode = Node(IdentityInterface(fields=['input_file', 'output_dir', 'script_path']), name='inputnode')
        inputnode.inputs.input_file = t1w_file
        inputnode.inputs.output_dir = self.output_path
        inputnode.inputs.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matlab', 'test', 'matlab_test.m'))

        test_matlab_node = Node(TestMatlab(), name='test_matlab_node')
        test_matlab_wf.connect(inputnode, 'input_file', test_matlab_node, 'input_file')
        test_matlab_wf.connect(inputnode, 'output_dir', test_matlab_node, 'output_dir')
        test_matlab_wf.connect(inputnode, 'script_path', test_matlab_node, 'script_path')

        return test_matlab_wf