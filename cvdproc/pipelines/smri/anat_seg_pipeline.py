import os
import pandas as pd
import subprocess
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function

from cvdproc.pipelines.smri.anat_seg.cp_seg.chpseg import ChPSeg
from cvdproc.pipelines.smri.freesurfer.synthseg import SynthSeg

class AnatSegPipeline:
    """
    Generating anatomical segmentation atlas from sMRI
    """
    def __init__(self,
                 subject: object,
                 session: object,
                 output_path: str,
                 use_which_t1w: str = "T1w",
                 methods: list = ["synthseg", "chpseg"],
                 cpu_first: bool = False,
                 **kwargs):
        """
        Generating anatomical segmentation atlas from sMRI

        Args:
            subject: Subject object
            session: Session object
            output_path: Output directory
            use_which_t1w: specific string to select T1w image, e.g. 'acq-highres'. If None, T1w image is not used
            methods: List of methods to use. Options include 'synthseg' and 'chpseg'.
            cpu_first: Whether to use CPU first for SynthSeg (if available)
            **kwargs: Additional arguments
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.use_which_t1w = use_which_t1w
        self.methods = methods
        self.cpu_first = cpu_first
        self.kwargs = kwargs
    
    def check_data_requirements(self):
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
        print(f"[ANAT_SEG] Using T1w file: {t1w_file}")
        print(f"[ANAT_SEG] Using method: {self.methods}")

        # Create the workflow
        anatseg_workflow = Workflow(name='anatseg_workflow')
        anatseg_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')

        inputnode = Node(IdentityInterface(fields=['t1w_file']),
                         name='inputnode')
        inputnode.inputs.t1w_file = t1w_file

        # outputnode = Node(IdentityInterface(fields=['synthseg_out', 'chpseg_out']),
        #                   name='outputnode')

        if 'synthseg' in self.methods:
            synthseg = Node(SynthSeg(), name='synthseg')
            anatseg_workflow.connect(inputnode, 't1w_file', synthseg, 'image')
            synthseg.inputs.out = os.path.join(self.output_path, 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_synthseg.nii.gz')
            synthseg.inputs.vol = os.path.join(self.output_path, 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-synthseg_volume.csv')
            synthseg.inputs.robust = True
            synthseg.inputs.parc = True
            synthseg.inputs.keepgeom = True
            if self.cpu_first:
                synthseg.inputs.cpu = True
        
        if 'chpseg' in self.methods:
            chpseg = Node(ChPSeg(), name='chpseg')
            anatseg_workflow.connect(inputnode, 't1w_file', chpseg, 'in_t1')
            chpseg.inputs.output_dir = os.path.join(self.output_path, 'chpseg')
        
        return anatseg_workflow
