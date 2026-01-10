import os
import subprocess
import nibabel as nib
import numpy as np
from nipype.interfaces.freesurfer import ReconAll
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from cvdproc.pipelines.common.register import SynthmorphNonlinear

from ...bids_data.rename_bids_file import rename_bids_file

class T1RegisterPipeline:
    def __init__(self, subject, session, output_path, use_which_t1w: str = None, template_res: float = 1, **kwargs):
        """
        T1w registration pipeline to register T1w images to MNI space using SynthMorph.

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Output directory to save results.
            use_which_t1w (str, optional): Keyword to select the desired T1w image.
            template_res (float, optional): Resolution of the MNI template to use (1mm or 0.5mm). Default is 1mm.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = use_which_t1w
        self.template_res = template_res

    def check_data_requirements(self):
        return self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # get T1w image
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
        print(f"[T1_REGISTER] Using T1w file: {t1w_file}")

        # Create the workflow
        t1_register_wf = Workflow(name='t1_register_workflow')
        t1_register_wf.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')

        inputnode = Node(IdentityInterface(fields=['t1', 'mni_template', 't1_mni_out', 't1_2_mni_warp', 'mni_2_t1_warp']),
                         name='inputnode')

        if self.template_res == 1:
            mni_template = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'standard', 'MNI152', 'MNI152_T1_1mm_brain.nii.gz')
            inputnode.inputs.register_between_stripped = True
        elif self.template_res == 0.5:
            mni_template = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'standard', 'MNI152', 'MNI152_T1_0.5mm.nii.gz')
            inputnode.inputs.register_between_stripped = False

        inputnode.inputs.t1 = t1w_file
        inputnode.inputs.mni_template = mni_template
        inputnode.inputs.t1_mni_out = os.path.join(self.output_path, rename_bids_file(t1w_file, {'space': 'MNI152NLin6ASym'}, 'T1w', '.nii.gz'))
        inputnode.inputs.t1_2_mni_warp = os.path.join(self.output_path, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz')
        inputnode.inputs.mni_2_t1_warp = os.path.join(self.output_path, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')

        register_node = Node(SynthmorphNonlinear(), name='synthmorph_register')
        t1_register_wf.connect(inputnode, 't1', register_node, 't1')
        t1_register_wf.connect(inputnode, 'mni_template', register_node, 'mni_template')
        t1_register_wf.connect(inputnode, 't1_mni_out', register_node, 't1_mni_out')
        t1_register_wf.connect(inputnode, 't1_2_mni_warp', register_node, 't1_2_mni_warp')
        t1_register_wf.connect(inputnode, 'mni_2_t1_warp', register_node, 'mni_2_t1_warp')
        t1_register_wf.connect(inputnode, 'register_between_stripped', register_node, 'register_between_stripped')
        register_node.inputs.t1_stripped_out = os.path.join(self.output_path, rename_bids_file(t1w_file, {'space': 'T1w', 'desc': 'brain'}, 'T1w', '.nii.gz'))
        register_node.inputs.brain_mask_out = os.path.join(self.output_path, rename_bids_file(t1w_file, {'space': 'T1w', 'desc': 'brain'}, 'mask', '.nii.gz'))

        return t1_register_wf