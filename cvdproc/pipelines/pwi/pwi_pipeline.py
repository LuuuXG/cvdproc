import os
import pandas as pd
import json
import nibabel as nib
import numpy as np
import re

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function

from cvdproc.pipelines.smri.freesurfer.synthstrip import SynthStrip
from cvdproc.bids_data.rename_bids_file import rename_bids_file
from cvdproc.pipelines.pwi.aif.auto_aif_nipype import AutoAIFFromPWI
from cvdproc.pipelines.pwi.dsc_mri_toolbox.dsc_mri_nipype import DSCMRI, Conc

class PWIPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        PWI postprocessing pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_pwi = kwargs.get('use_which_pwi', 'pwi')
        self.dsc_mri_toolbox_path = kwargs.get('dsc_mri_toolbox_path', None)

        self.extract_from = kwargs.get('extract_from', None)

    def check_data_requirements(self):
        """
        :return: bool
        """
        return self.session.get_pwi_files() is not None
    
    def create_workflow(self):
        # Get the PWI file (a .nii.gz and a .json)
        pwi_files = self.session.get_pwi_files()

        if len(pwi_files) == 1:
            pwi_path = pwi_files[0]
        else:
            # find all files contain self.use_which_pwi
            # if more than one PWI file, use the first one
            pwi_path = next((f for f in pwi_files if self.use_which_pwi in f), None)

        if pwi_path is None:
            raise FileNotFoundError(f"No PWI file found with keyword '{self.use_which_pwi}' in session {self.session.session_id}.")
        pwi_path = os.path.abspath(pwi_path)
        print('Using PWI file:', pwi_path)

        pwi_img = pwi_path
        pwi_json = pwi_path.replace('.nii.gz', '.json')

        # get EchoTime from the JSON file
        with open(pwi_json, 'r') as f:
            pwi_json_data = json.load(f)
        if 'EchoTime' in pwi_json_data and 'RepetitionTime' in pwi_json_data:
            echo_time = pwi_json_data['EchoTime']
            repetition_time = pwi_json_data['RepetitionTime']
            print(f"EchoTime: {echo_time}, RepetitionTime: {repetition_time}")
        else:
            raise KeyError(f"EchoTime not found in JSON file {pwi_json}. Please ensure the PWI JSON file contains 'EchoTime'.")

        #################
        # Main Workflow #
        #################
        pwi_workflow = Workflow(name='pwi_workflow')
        pwi_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}")

        # Input node
        input_node = Node(IdentityInterface(fields=['pwi_img', 'echo_time', 'repetition_time', 'dsc_mri_toolbox_path']),
                         name='input_node')
        input_node.inputs.pwi_img = pwi_img
        input_node.inputs.echo_time = echo_time
        input_node.inputs.repetition_time = repetition_time
        input_node.inputs.dsc_mri_toolbox_path = self.dsc_mri_toolbox_path
        
        # Get pwi mask
        synthstrip_node = Node(SynthStrip(), name='synthstrip_node')
        pwi_workflow.connect(input_node, 'pwi_img', synthstrip_node, 'image')
        synthstrip_node.inputs.four_d = True  # Process as 4D image
        synthstrip_node.inputs.mask_file = os.path.join(self.output_path, rename_bids_file(pwi_path, {'space': 'pwi'}, 'brainmask', '.nii.gz'))

        # Calculate concentration
        conc_node = Node(Conc(), name='conc_node')
        pwi_workflow.connect(input_node, 'dsc_mri_toolbox_path', conc_node, 'toolbox_dir')
        pwi_workflow.connect(input_node, 'pwi_img', conc_node, 'pwi_path')
        pwi_workflow.connect(synthstrip_node, 'mask_file', conc_node, 'mask_path')
        pwi_workflow.connect(input_node, 'echo_time', conc_node, 'echo_time')
        pwi_workflow.connect(input_node, 'repetition_time', conc_node, 'repetition_time')
        conc_node.inputs.output_path = self.output_path
        conc_node.inputs.output_conc_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'conc', ''))
        conc_node.inputs.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matlab', 'dsc_mri_toolbox', 'conc.m'))

        # Auto AIF calculation
        auto_aif_node = Node(AutoAIFFromPWI(), name='auto_aif_node')
        pwi_workflow.connect(conc_node, 'output_conc_path', auto_aif_node, 'conc_path')
        pwi_workflow.connect(synthstrip_node, 'mask_file', auto_aif_node, 'mask_path')
        auto_aif_node.inputs.time_echo = echo_time
        #auto_aif_node.inputs.output_conc = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'deltaR2s', '.nii.gz'))
        auto_aif_node.inputs.output_aif_vec = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'AIF', '.mat'))
        auto_aif_node.inputs.output_aif_roi = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'AIFmask', '.nii.gz'))
        auto_aif_node.inputs.baseline_range = [0, 15]

        # Calculate PWI map
        pwi_map_node = Node(DSCMRI(), name='pwi_map_node')
        pwi_workflow.connect(input_node, 'dsc_mri_toolbox_path', pwi_map_node, 'toolbox_dir')
        pwi_workflow.connect(input_node, 'pwi_img', pwi_map_node, 'pwi_path')
        pwi_workflow.connect(auto_aif_node, 'output_aif_vec', pwi_map_node, 'aif_conc')
        pwi_workflow.connect(synthstrip_node, 'mask_file', pwi_map_node, 'mask_path')
        pwi_workflow.connect(input_node, 'echo_time', pwi_map_node, 'echo_time')
        pwi_workflow.connect(input_node, 'repetition_time', pwi_map_node, 'repetition_time')
        pwi_map_node.inputs.output_path = self.output_path
        pwi_map_node.inputs.output_cbv_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'CBV', ''))
        pwi_map_node.inputs.output_cbv_lc_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {'desc': 'LeakageCorrection'}, 'CBV', ''))
        pwi_map_node.inputs.output_k2_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'K2map', ''))
        pwi_map_node.inputs.output_cbf_svd_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {'desc': 'SVD'}, 'CBF', ''))
        pwi_map_node.inputs.output_cbf_csvd_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {'desc': 'CSVD'}, 'CBF', ''))
        pwi_map_node.inputs.output_cbf_osvd_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {'desc': 'OSVD'}, 'CBF', ''))
        pwi_map_node.inputs.output_mtt_svd_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {'desc': 'SVD'}, 'MTT', ''))
        pwi_map_node.inputs.output_mtt_csvd_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {'desc': 'CSVD'}, 'MTT', ''))
        pwi_map_node.inputs.output_mtt_osvd_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {'desc': 'OSVD'}, 'MTT', ''))
        pwi_map_node.inputs.output_ttp_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'TTP', ''))
        pwi_map_node.inputs.output_s0_path = os.path.join(self.output_path, rename_bids_file(pwi_path, {}, 'S0map', ''))
        pwi_map_node.inputs.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matlab', 'dsc_mri_toolbox', 'pwimap.m'))

        return pwi_workflow