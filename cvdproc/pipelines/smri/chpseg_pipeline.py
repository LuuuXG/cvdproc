import os
import pandas as pd
import subprocess
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function

from cvdproc.pipelines.smri.cp_seg.chpseg import ChPSeg

class ChPSegPipeline:
    """
    Choroid plexus segmentation pipeline.
    """
    def __init__(self, 
                 subject, 
                 session, 
                 output_path, 
                 use_which_t1w: str = "T1w",
                 method: str = "chpseg",
                 **kwargs):
        """
        chp_seg pipeline

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Directory to save outputs.
            use_which_t1w (str, optional): Keyword to select the desired T1w file.
            method (str, optional): Method to use for segmentation, default is "chpseg".
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.use_which_t1w = use_which_t1w
        self.method = method

    def check_data_requirements(self):
        """
        check if the required data is available
        :return: bool
        """
        return self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # Get the T1w file
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            # ensure that there is only 1 suitable file
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]
        print(f"[CP_SEG] Using T1w file: {t1w_file}")
        print(f"[CP_SEG] Using method: {self.method}")

        # Create the workflow
        cpseg_wf = Workflow(name='chpseg_workflow')
        cpseg_wf.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
        
        inputnode = Node(IdentityInterface(fields=['in_t1', 'output_dir']),
                         name='inputnode')
        inputnode.inputs.in_t1 = t1w_file
        inputnode.inputs.output_dir = self.output_path

        if self.method == "chpseg":
            chpseg_node = Node(ChPSeg(), name='chpseg_node')
            cpseg_wf.connect(inputnode, 'in_t1', chpseg_node, 'in_t1')
            cpseg_wf.connect(inputnode, 'output_dir', chpseg_node, 'output_dir')
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        return cpseg_wf