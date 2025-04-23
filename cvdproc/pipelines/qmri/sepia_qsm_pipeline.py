import os
import subprocess
import nibabel as nib
import numpy as np
import json

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from .sepia_qsm.sepia_qsm_nipype import SepiaQSM, QSMRegister
from ..smri.fsl.fsl_anat_nipype import FSLANAT
from ..common.copy_to_rawdata import CopyToRawData

class SepiaQSMPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        Sepia QSM processing pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = kwargs.get('use_which_t1w', None)
        self.normalize = kwargs.get('normalize', False) # normalize to MNI space
        self.sepia_toolbox_path = kwargs.get('sepia_toolbox_path', None) # If you can type 'sepia' to open SEPIA GUI,
                                                                         # It is likely that the toolbox is already in the path.
                                                                         # So, you don't need to specify this.
                                                                         # Otherwise, you need to specify the path to the SEPIA toolbox.

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.sepia_qsm_script = os.path.join(base_dir, 'matlab', 'sepia_qsm', 'sepia_process.m')
        self.qsm_register_script = os.path.join(base_dir, 'bash', 'qsm_register.sh')
    
    def check_data_requirements(self):
        # We need T1w and QSM data
        return self.session.get_t1w_files() is not None and self.session.qsm_files is not None
    
    def create_workflow(self):
        os.makedirs(self.output_path, exist_ok=True)

        # get T1w image
        t1w_files = self.session.get_t1w_files()
        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]
        
        # get QSM image
        qsm_files = self.session.qsm_files
        
        # get the first echo magnitude image
        # have 'echo-1' and 'part-mag' in the file name
        mag_files = [f for f in qsm_files if 'echo-1_' in f and 'part-mag' in f]
        if len(mag_files) != 1:
            raise FileNotFoundError("No specific 1st echo magnitude image found.")
        first_mag_file = mag_files[0]

        sepia_qsm_wf = Workflow(name='sepia_qsm_wf', base_dir=self.output_path)

        inputnode = Node(IdentityInterface(fields=['t1w_file', 'first_mag_file', 'input_qsm_bids_dir', 'phase_image_correction', 'reverse_phase', 
                                                   'subject_output_folder', 'script_path', 'sepia_toolbox_path']), name='inputnode')
        inputnode.inputs.t1w_file = t1w_file
        inputnode.inputs.first_mag_file = first_mag_file
        inputnode.inputs.input_qsm_bids_dir = os.path.join(self.session.session_dir, 'qsm')
        inputnode.inputs.phase_image_correction = True
        inputnode.inputs.reverse_phase = 1
        inputnode.inputs.subject_output_folder = self.output_path
        inputnode.inputs.script_path = self.sepia_qsm_script
        inputnode.inputs.sepia_toolbox_path = self.sepia_toolbox_path

        sepia_qsm_node = Node(SepiaQSM(), name='sepia_qsm_node')
        sepia_qsm_wf.connect(inputnode, 'input_qsm_bids_dir', sepia_qsm_node, 'input_qsm_bids_dir')
        sepia_qsm_wf.connect(inputnode, 'phase_image_correction', sepia_qsm_node, 'phase_image_correction')
        sepia_qsm_wf.connect(inputnode, 'reverse_phase', sepia_qsm_node, 'reverse_phase')
        sepia_qsm_wf.connect(inputnode, 'subject_output_folder', sepia_qsm_node, 'subject_output_folder')
        sepia_qsm_wf.connect(inputnode, 'script_path', sepia_qsm_node, 'script_path')
        sepia_qsm_wf.connect(inputnode, 'sepia_toolbox_path', sepia_qsm_node, 'sepia_toolbox_path')

        copy_to_rawdata_node = Node(CopyToRawData(), name='copy_to_rawdata_node')
        sepia_qsm_wf.connect(sepia_qsm_node, 'swi', copy_to_rawdata_node, 'in_file')
        copy_to_rawdata_node.inputs.reference_file = ''
        copy_to_rawdata_node.inputs.output_dir = os.path.join(self.session.session_dir, 'swi')
        copy_to_rawdata_node.inputs.entities = {
            'sub': self.subject.subject_id,
            'ses': self.session.session_id,
            'desc': 'clearswi',
        }
        copy_to_rawdata_node.inputs.suffix = 'SWI'
        copy_to_rawdata_node.inputs.extension = '.nii.gz'

        if self.normalize:
            fsl_anat_output_path = os.path.join(self.session.bids_dir, 'derivatives', 'fsl_anat', f"sub-{self.session.subject_id}", f"ses-{self.session.session_id}")

            qsm_normalize_node = Node(QSMRegister(), name='qsm_normalize_node')
            sepia_qsm_wf.connect(inputnode, 't1w_file', qsm_normalize_node, 't1w_image')
            sepia_qsm_wf.connect(inputnode, 'first_mag_file', qsm_normalize_node, 'mag_image')
            sepia_qsm_wf.connect(inputnode, 'subject_output_folder', qsm_normalize_node, 'output_dir')
            
            if not os.path.exists(fsl_anat_output_path):
                os.makedirs(fsl_anat_output_path, exist_ok=True)

                fsl_anat_node = Node(FSLANAT(), name='fsl_anat_node')
                fsl_anat_node.inputs.input_image = t1w_file
                fsl_anat_node.inputs.output_directory = os.path.join(fsl_anat_output_path, 'fsl')

                sepia_qsm_wf.connect(fsl_anat_node, 'output_directory', qsm_normalize_node, 'fsl_anat_dir')
            else:
                qsm_normalize_node.inputs.fsl_anat_dir = os.path.join(self.session.bids_dir, 'derivatives', 'fsl_anat', f"sub-{self.subject}", f"ses-{self.session}", "fsl.anat")
            
            susceptibility_map = os.path.join(self.output_path, 'Sepia_Chimap.nii.gz')
            s0_map = os.path.join(self.output_path, 'Sepia_S0map.nii.gz')
            r2star_map = os.path.join(self.output_path, 'Sepia_R2starmap.nii.gz')
            t2star_map = os.path.join(self.output_path, 'Sepia_T2starmap.nii.gz')
            swi = os.path.join(self.output_path, 'Sepia_clearswi.nii.gz')
            mip = os.path.join(self.output_path, 'Sepia_clearswi-minIP.nii.gz')

            qsm_images = [susceptibility_map, s0_map, r2star_map, t2star_map, swi, mip]

            qsm_normalize_node.inputs.qsm_images = qsm_images
            qsm_normalize_node.inputs.qsm_register_script = self.qsm_register_script
        
        outputnode = Node(IdentityInterface(fields=['output_folder']), name='outputnode')
        sepia_qsm_wf.connect(sepia_qsm_node, 'output_folder', outputnode, 'output_folder')
        
        return sepia_qsm_wf