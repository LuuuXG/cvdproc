import os
import subprocess
import nibabel as nib
import numpy as np
import json

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from .qsm_pipeline_part1.qsm_pipeline_part1_nipype import QSMPipelinePart1
from .qqnet.qqnet_nipype import QQNet
from ..common.copy_to_rawdata import CopyToRawData
from ..common.register import ModalityRegistration
from cvdproc.pipelines.common.register import SynthmorphNonlinear
from cvdproc.pipelines.common.filter_existing import FilterExisting
from cvdproc.pipelines.qmri.qsm_register.qsm_register2_nipype import QSMRegister

from ...bids_data.rename_bids_file import rename_bids_file
from ...utils.python.basic_image_processor import extract_roi_means

class QSMPipeline:
    """
    QSM Processing Pipeline using Sepia and QQNet.
    """
    def __init__(
            self,
            subject,
            session,
            output_path,
            use_which_t1w: str = None,
            normalize: bool = False,
            phase_image_correction: bool = False,
            reverse_phase: int = 0,
            **kwargs
    ):
        """
        QSM processing pipeline.

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Output directory to save results.
            use_which_t1w (str, optional): Keyword to select the desired T1w image.
            normalize (bool, optional): If True, normalize QSM (and other scalar maps) to MNI space via T1w.
            phase_image_correction (bool, optional): If True, apply phase image correction for inter-slice phase polarity differences in the GE data. (https://github.com/kschan0214/sepia/discussions/93)
            reverse_phase (int, optional): Set to 1 to inverse phase polarity (for GE scanners).
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = use_which_t1w
        self.normalize = normalize
        self.phase_image_correction = phase_image_correction
        self.reverse_phase = reverse_phase
    
    def check_data_requirements(self):
        # We need T1w and QSM data
        return self.session.get_t1w_files() is not None and self.session.qsm_files is not None
    
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
        print(f"[QSM] Using T1w file: {t1w_file}")

        qsm_wf = Workflow(name='qsm_workflow')

        inputnode = Node(IdentityInterface(fields=['in_t1', 'bids_dir', 'subject_id', 'session_id', 'output_dir', 'phase_image_correction', 'reverse_phase']),
                         name='inputnode')
        inputnode.inputs.in_t1 = t1w_file
        inputnode.inputs.bids_dir = self.subject.bids_dir
        inputnode.inputs.subject_id = self.subject.subject_id
        inputnode.inputs.session_id = self.session.session_id
        inputnode.inputs.output_dir = self.output_path
        inputnode.inputs.phase_image_correction = self.phase_image_correction
        inputnode.inputs.reverse_phase = self.reverse_phase

        # QSM_pipeline_part1
        qsm_pipeline_part1_node = Node(QSMPipelinePart1(), name='qsm_pipeline_part1')
        qsm_pipeline_part1_node.inputs.cvdproc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        qsm_pipeline_part1_node.inputs.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matlab', "qsm_pipeline_part1", "QSM_pipeline_part1.m"))
        qsm_wf.connect(inputnode, 'bids_dir', qsm_pipeline_part1_node, 'bids_root_dir')
        qsm_wf.connect(inputnode, 'subject_id', qsm_pipeline_part1_node, 'subject_id')
        qsm_wf.connect(inputnode, 'session_id', qsm_pipeline_part1_node, 'session_id')
        qsm_wf.connect(inputnode, 'phase_image_correction', qsm_pipeline_part1_node, 'phase_image_correction')
        qsm_wf.connect(inputnode, 'reverse_phase', qsm_pipeline_part1_node, 'reverse_phase')

        # QQnet
        qqnet_node = Node(QQNet(), name='qqnet')
        qqnet_node.inputs.output_dir = os.path.join(self.output_path, 'qqnet_output')
        qqnet_node.inputs.prefix = f'sub-{self.subject.subject_id}_ses-{self.session.session_id}'
        qsm_wf.connect(qsm_pipeline_part1_node, 'processed_mag_path', qqnet_node, 'mag_4d_path')
        qsm_wf.connect(qsm_pipeline_part1_node, 'chisep_qsm_path', qqnet_node, 'qsm_path')
        qsm_wf.connect(qsm_pipeline_part1_node, 'qsm_mask_path', qqnet_node, 'mask_path')
        qsm_wf.connect(qsm_pipeline_part1_node, 'r2star_path', qqnet_node, 'r2star_path')
        qsm_wf.connect(qsm_pipeline_part1_node, 's0_path', qqnet_node, 's0_path')
        qsm_wf.connect(qsm_pipeline_part1_node, 'header_path', qqnet_node, 'header_path')

        # register to MNI
        if self.normalize:
            # 01: register to T1w
            qsm_files = self.session.qsm_files
            # find the first QSM file (.nii.gz) containing 'echo-1' and 'part-mag'
            mag_1stecho_file = None
            for f in qsm_files:
                if f.endswith('.nii.gz') and 'echo-1' in f and 'part-mag' in f:
                    mag_1stecho_file = f
                    break
            if mag_1stecho_file is None:
                raise FileNotFoundError("No suitable QSM magnitude file found (looking for 'echo-1' and 'part-mag').")
            print(f"[QSM] Using QSM magnitude file for registration: {mag_1stecho_file}")

            xfm_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
            os.makedirs(xfm_dir, exist_ok=True)

            mag_to_t1w_register_node = Node(ModalityRegistration(), name='mag_to_t1w_registration')
            qsm_wf.connect(inputnode, 'in_t1', mag_to_t1w_register_node, 'image_target')
            mag_to_t1w_register_node.inputs.image_target_strip = 0
            mag_to_t1w_register_node.inputs.image_source = mag_1stecho_file
            mag_to_t1w_register_node.inputs.image_source_strip = 0
            mag_to_t1w_register_node.inputs.flirt_direction = 1
            mag_to_t1w_register_node.inputs.output_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
            mag_to_t1w_register_node.inputs.registered_image_filename = rename_bids_file(mag_1stecho_file, {'space': 'T1w'}, 'GRE', '.nii.gz')
            mag_to_t1w_register_node.inputs.source_to_target_mat_filename = f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-QSM_to-T1w.mat'
            mag_to_t1w_register_node.inputs.target_to_source_mat_filename = f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-QSM.mat'
            mag_to_t1w_register_node.inputs.dof = 6

            # 02: register T1w to MNI
            target_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI_warp.nii.gz')
            t1_2_mni_warp_node = Node(IdentityInterface(fields=['t1_2_mni_warp']), name='t1_2_mni_warp_node')

            if not os.path.exists(target_warp):
                print(f"[QSM] T1w to MNI warp file not found. Running T1w to MNI registration (1mm resolution).")
                print(f"[QSM] If you want a different resolution, please run a separate T1 registration pipeline first.")
                t1w_to_mni_register_node = Node(SynthmorphNonlinear(), name='t1w_to_mni_registration')
                qsm_wf.connect(inputnode, 'in_t1', t1w_to_mni_register_node, 't1')
                t1w_to_mni_register_node.inputs.mni_template = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'standard', 'MNI152', 'MNI152_T1_1mm_brain.nii.gz')
                t1w_to_mni_register_node.inputs.t1_mni_out = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', rename_bids_file(t1w_file, {'space': 'MNI', 'desc':'brain'}, 'T1w', '.nii.gz'))
                t1w_to_mni_register_node.inputs.t1_2_mni_warp = target_warp
                t1w_to_mni_register_node.inputs.mni_2_t1_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI_to-T1w_warp.nii.gz')
                t1w_to_mni_register_node.inputs.t1_stripped = False
                t1w_to_mni_register_node.inputs.register_between_stripped = True

                qsm_wf.connect(t1w_to_mni_register_node, 't1_2_mni_warp', t1_2_mni_warp_node, 't1_2_mni_warp')
            else:
                print(f"[QSM] Found existing T1w to MNI warp file: {target_warp}")
                t1_2_mni_warp_node.inputs.t1_2_mni_warp = target_warp
            
            # 03: register QSM scalar maps to MNI
            # qsm_scalar_maps_node: merge different file paths to a list
            qsm_scalar_maps_node = Node(Merge(8), name='qsm_scalar_maps')
            qsm_wf.connect(qsm_pipeline_part1_node, 'chisep_qsm_path', qsm_scalar_maps_node, 'in1')
            qsm_wf.connect(qsm_pipeline_part1_node, 'r2star_path', qsm_scalar_maps_node, 'in2')
            qsm_wf.connect(qsm_pipeline_part1_node, 's0_path', qsm_scalar_maps_node, 'in3')
            qsm_wf.connect(qsm_pipeline_part1_node, 't2star_path', qsm_scalar_maps_node, 'in4')
            qsm_wf.connect(qsm_pipeline_part1_node, 'chidia_path', qsm_scalar_maps_node, 'in5')
            qsm_wf.connect(qsm_pipeline_part1_node, 'chipara_path', qsm_scalar_maps_node, 'in6')
            qsm_wf.connect(qsm_pipeline_part1_node, 'chitotal_path', qsm_scalar_maps_node, 'in7')
            qsm_wf.connect(qqnet_node, 'oef_path', qsm_scalar_maps_node, 'in8')

            # filter_existing_node = Node(FilterExisting(), name='filter_existing_qsm_scalars')
            # qsm_wf.connect(qsm_scalar_maps_node, 'out', filter_existing_node, 'input_file_list')

            output1_filenames = [
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-Chisep_Chimap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-R2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-S0map.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-T2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-Chidia.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-Chipara.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-Chitotal.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-OEF.nii.gz'
            ]

            output2_filenames = [
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-Chisep_Chimap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-R2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-S0map.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-T2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-Chidia.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-Chipara.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-Chitotal.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI_desc-OEF.nii.gz'
            ]

            qsm_to_mni_register_node = Node(QSMRegister(), name='qsm_to_mni_registration')
            qsm_wf.connect(inputnode, 'in_t1', qsm_to_mni_register_node, 't1w')
            qsm_wf.connect(t1_2_mni_warp_node, 't1_2_mni_warp', qsm_to_mni_register_node, 't1w_to_mni_warp')
            qsm_wf.connect(mag_to_t1w_register_node, 'source_to_target_mat', qsm_to_mni_register_node, 'qsm_to_t1w_affine')
            qsm_to_mni_register_node.inputs.output_dir = os.path.join(self.output_path, 'QSM_registered')
            qsm_wf.connect(qsm_scalar_maps_node, 'out', qsm_to_mni_register_node, 'input')
            qsm_to_mni_register_node.inputs.output1 = output1_filenames
            qsm_to_mni_register_node.inputs.output2 = output2_filenames

        return qsm_wf
