import os
import pandas as pd
import json
import nibabel as nib
import numpy as np
import re

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from cvdproc.pipelines.dmri.lqt.lqt_nipype import LQT

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class LQTPipeline:
    """
    Using the Lesion Quantification Toolkit (LQT) to quantify lesion disconnection.

    Last updated: 2025-08-02, WYJ
    """
    def __init__(self, 
                 subject, 
                 session, 
                 output_path,
                 seed_mask: str = 'lesion_mask', 
                 use_which_mask: str = 'infarction',
                 extract_from: str = None,
                 **kwargs):
        """
        LQT Pipeline for lesion disconnection analysis.

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Directory to save outputs.
            seed_mask (str, optional): Name of the seed mask folder in 'derivatives'. For example, if the ROI mask path is 'derivatives/lesion_mask/sub-XXX/ses-XXX/*infarction.nii.gz', then seed_mask='lesion_mask'.
            use_which_mask (str, optional): Keyword to select the desired lesion mask. Default is 'infarction'. For example, if the lesion mask is 'derivatives/lesion_mask/sub-XXX/ses-XXX/*infarction.nii.gz', then use_which_mask='infarction'.
            extract_from (str, optional): If extracting results, please provide it.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.seed_mask = seed_mask
        self.use_which_mask = use_which_mask

        self.extract_from = extract_from

    def check_data_requirements(self):
        """
        Will always return True, as the LQT pipeline will check the seed mask in MNI space during creation of the workflow.
        """
        return True
    
    def create_workflow(self):
        # Find the lesion file
        seed_mask_dir = os.path.join(self.subject.bids_dir, 'derivatives', self.seed_mask, 'sub-' + self.subject.subject_id, 'ses-' + self.session.session_id)
        # Search for the file containing the self.use_which_mask (if multiple, return the first one)
        lesion_files = [f for f in os.listdir(seed_mask_dir) if self.use_which_mask in f and f.endswith('.nii.gz')]
        if not lesion_files:
            raise FileNotFoundError(f"No lesion file found in {seed_mask_dir} containing '{self.use_which_mask}'")
        lesion_file = os.path.join(seed_mask_dir, lesion_files[0])

        lqt_workflow = Workflow(name='lqt_workflow')
        lqt_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', 'sub-' + self.subject.subject_id, 'ses-' + self.session.session_id)

        # Input node
        input_node = Node(IdentityInterface(fields=['patient_id', 'lesion_file', 'output_dir', 'parcel_path', 'lqt_script']),
                          name='input_node')
        input_node.inputs.patient_id = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}"
        input_node.inputs.lesion_file = lesion_file
        input_node.inputs.output_dir = self.output_path
        input_node.inputs.parcel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lqt', 'extdata', 'Schaefer_Yeo_Plus_Subcort', '100Parcels7Networks.nii.gz'))
        input_node.inputs.lqt_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'r', 'lqt', 'lqt_single_subject.R'))

        lqt_process_node = Node(LQT(), name='lqt_process')
        lqt_workflow.connect([
            (input_node, lqt_process_node, [('patient_id', 'patient_id'),
                                            ('lesion_file', 'lesion_file'),
                                            ('output_dir', 'output_dir'),
                                            ('parcel_path', 'parcel_path'),
                                            ('lqt_script', 'lqt_script')])
        ])
        lqt_process_node.inputs.dsi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lqt', 'extdata', 'DSI_studio', 'dsi-studio', 'dsi_studio'))

        return lqt_workflow
    
    def extract_results(self):
        print("TODO")
        return None