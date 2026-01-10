import os
import subprocess
import nibabel as nib
import numpy as np
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge

from ...bids_data.rename_bids_file import rename_bids_file

class SCNPipeline:
    def __init__(self,
                 subject,
                 session,
                 output_path,
                 method: list = ['MIND'],
                 use_freesurfer_clinical: bool = False,
                 **kwargs):
        """
        Structural Covariance Network (SCN) analysis pipeline.

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Output directory to save results.
            method (list, optional): List of methods to use for SCN analysis. Default is ['MIND'].
            use_freesurfer_clinical (bool, optional): Whether to use recon-all-clinical.sh outputs. Default is False.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.method = method
        self.use_freesurfer_clinical = use_freesurfer_clinical
    
    def check_data_requirements(self):
        # currently no specific data requirements
        return True
    
    def create_workflow(self):
        scn_workflow = Workflow(name='scn_workflow')

        # handle freesurfer outputs
        if self.use_freesurfer_clinical:
            fs_subjects_dir = os.path.dirname(self.session.freesurfer_clinical_dir)
            fs_subject_id = os.path.basename(self.session.freesurfer_clinical_dir)
            if self.session.freesurfer_clinical_dir is None:
                raise FileNotFoundError("[SCN Pipeline] Freesurfer clinical directory not found, but use_freesurfer_clinical is set to True.")
        else:
            fs_subjects_dir = os.path.dirname(self.session.freesurfer_dir)
            fs_subject_id = os.path.basename(self.session.freesurfer_dir)
            if self.session.freesurfer_dir is None:
                raise FileNotFoundError("[SCN Pipeline] Freesurfer directory not found.")
        
        inputnode = Node(IdentityInterface(fields=['fs_subjects_dir', 'fs_subject_id', 'fs_dir']),
                         name='inputnode')
        inputnode.inputs.fs_subjects_dir = fs_subjects_dir
        inputnode.inputs.fs_subject_id = fs_subject_id
        inputnode.inputs.fs_dir = self.session.freesurfer_dir

        if 'MIND' in self.method:
            from cvdproc.pipelines.smri.scn.scn_nipype import MINDCompute
            mind_scn_node = Node(MINDCompute(), name='mind_scn_node')
            scn_workflow.connect([(inputnode, mind_scn_node, [('fs_dir', 'surf_dir')])])
            mind_scn_node.inputs.features = ['CT','MC','Vol','SD','SA']
            mind_scn_node.inputs.parcellation = 'aparc'
            mind_output_csv = os.path.join(self.output_path, 'MIND', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_seg-DK_desc-MIND_connectivity.csv')
            mind_scn_node.inputs.output_csv = mind_output_csv
        
        return scn_workflow