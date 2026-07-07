import os
import numpy as np
import glob

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from .qsm_pipeline_part1.qsm_pipeline_part1_nipype import QSMPipelinePart1
from .qqnet.qqnet_nipype import QQNet
from cvdproc.pipelines.common.files import FilterExisting
from cvdproc.pipelines.qmri.qsm_register.qsm_register2_nipype import QSMRegister
from cvdproc.pipelines.dmri.stats.dti_scalar_maps import CalculateScalarMaps
from cvdproc.pipelines.common.register import Tkregister2fs2t1w, MRIConvertApplyWarp, SynthmorphNonlinear, ModalityRegistration
from nipype.interfaces.fsl import FLIRT
from nipype.interfaces.freesurfer import MRIConvert

from ...bids_data.rename_bids_file import rename_bids_file
from ...utils.python.basic_image_processor import extract_roi_means
from cvdproc.config.paths import get_package_path

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
            qsm_metrics_stats: bool = False,
            skip_reconstruction: bool = False,
            extract_from: str = None,
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
            qsm_metrics_stats (bool, optional): If True, compute and save ROI statistics.
            skip_reconstruction (bool, optional): If True, skip the reconstruction step.
            extract_from (str, optional): Directory to extract results from (for population-level analysis).
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = use_which_t1w
        self.normalize = normalize
        self.phase_image_correction = phase_image_correction
        self.reverse_phase = reverse_phase
        self.qsm_metrics_stats = qsm_metrics_stats
        self.skip_reconstruction = skip_reconstruction
        self.extract_from = extract_from
    
    def check_data_requirements(self):
        # We need T1w and QSM data
        return self.session.get_t1w_files() is not None and self.session.qsm_files is not None
    
    def create_workflow(self):
        if self.qsm_metrics_stats and not self.normalize:
            self.normalize = True
            print("[QSM Pipeline] Warning: QSM metrics statistics computation requires registration to T1w space. Setting normalize=True.")

        # get T1w image
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            # ensure that there is only 1 suitable file
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"[QSM Pipeline] No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("[QSM Pipeline] No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]
        print(f"[QSM Pipeline] Using T1w file: {t1w_file}")

        fs_output = self.session.freesurfer_dir
        # if fs_output exists
        if fs_output is None:
            fs_subjects_dir = ''
            fs_subject_id = ''
            fs_output_process = False
            print("[QSM Pipeline] No FreeSurfer output found. Skipping related processing.")
        else:
            fs_subjects_dir = os.path.dirname(fs_output)
            fs_subject_id = os.path.basename(fs_output)
            # automatically do related anat preprocessing
            fs_output_process = True
            print(f"[QSM Pipeline] FreeSurfer output found: {fs_output}. Will do related processing.")

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
        inputnode.inputs.fs_subjects_dir = fs_subjects_dir
        inputnode.inputs.fs_subject_id = fs_subject_id

        qsm_metrics_node = Node(IdentityInterface(fields=['r2star_path', 's0_path', 't2star_path',
                                                              'chisep_qsm_path', 'chidia_path', 'chipara_path', 'chitotal_path',
                                                              'oef_path']),
                                     name='qsm_metrics')
        if self.skip_reconstruction:
            qsm_metrics_node.inputs.r2star_path = os.path.join(self.output_path, 'sepia_output', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_R2starmap.nii.gz')
            qsm_metrics_node.inputs.s0_path = os.path.join(self.output_path, 'sepia_output', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_S0map.nii.gz')
            qsm_metrics_node.inputs.t2star_path = os.path.join(self.output_path, 'sepia_output', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_T2starmap.nii.gz')
            qsm_metrics_node.inputs.chisep_qsm_path = os.path.join(self.output_path, 'QSM_reconstruction', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-QSMnet_Chimap.nii.gz')
            qsm_metrics_node.inputs.chidia_path = os.path.join(self.output_path, 'QSM_reconstruction', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_ChiDia.nii.gz')
            qsm_metrics_node.inputs.chipara_path = os.path.join(self.output_path, 'QSM_reconstruction', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_ChiPara.nii.gz')
            qsm_metrics_node.inputs.chitotal_path = os.path.join(self.output_path, 'QSM_reconstruction', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_ChiTotal.nii.gz')
            qsm_metrics_node.inputs.oef_path = os.path.join(self.output_path, 'qqnet_output', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-QQNET_OEF.nii.gz')
        else:
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

            qsm_wf.connect(qsm_pipeline_part1_node, 'chisep_qsm_path', qsm_metrics_node, 'chisep_qsm_path')
            qsm_wf.connect(qsm_pipeline_part1_node, 'r2star_path', qsm_metrics_node, 'r2star_path')
            qsm_wf.connect(qsm_pipeline_part1_node, 's0_path', qsm_metrics_node, 's0_path')
            qsm_wf.connect(qsm_pipeline_part1_node, 't2star_path', qsm_metrics_node, 't2star_path')
            qsm_wf.connect(qsm_pipeline_part1_node, 'chidia_path', qsm_metrics_node, 'chidia_path')
            qsm_wf.connect(qsm_pipeline_part1_node, 'chipara_path', qsm_metrics_node, 'chipara_path')
            qsm_wf.connect(qsm_pipeline_part1_node, 'chitotal_path', qsm_metrics_node, 'chitotal_path')
            qsm_wf.connect(qqnet_node, 'oef_path', qsm_metrics_node, 'oef_path')

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
                raise FileNotFoundError("[QSM Pipeline] No suitable QSM magnitude file found (looking for 'echo-1' and 'part-mag').")
            print(f"[QSM Pipeline] Using QSM magnitude file for registration: {mag_1stecho_file}")

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
            target_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz')
            target_inverse_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')
            t1_2_mni_warp_node = Node(IdentityInterface(fields=['t1_2_mni_warp']), name='t1_2_mni_warp_node')
            mni_2_t1w_warp_node = Node(IdentityInterface(fields=['mni_2_t1w_warp']), name='mni_2_t1w_warp_node')

            if not os.path.exists(target_warp):
                print(f"[QSM Pipeline] T1w to MNI warp file not found. Running T1w to MNI registration (1mm resolution).")
                print(f"[QSM Pipeline] If you want a different resolution, please run a separate T1 registration pipeline first.")
                t1w_to_mni_register_node = Node(SynthmorphNonlinear(), name='t1w_to_mni_registration')
                qsm_wf.connect(inputnode, 'in_t1', t1w_to_mni_register_node, 't1')
                t1w_to_mni_register_node.inputs.mni_template = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'standard', 'MNI152', 'MNI152_T1_1mm_brain.nii.gz')
                t1w_to_mni_register_node.inputs.t1_mni_out = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', rename_bids_file(t1w_file, {'space': 'MNI152NLin6ASym', 'desc':'brain'}, 'T1w', '.nii.gz'))
                t1w_to_mni_register_node.inputs.t1_2_mni_warp = target_warp
                t1w_to_mni_register_node.inputs.mni_2_t1_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')
                t1w_to_mni_register_node.inputs.register_between_stripped = True

                qsm_wf.connect(t1w_to_mni_register_node, 't1_2_mni_warp', t1_2_mni_warp_node, 't1_2_mni_warp')
            else:
                print(f"[QSM Pipeline] Found existing T1w to MNI warp file: {target_warp}")
                t1_2_mni_warp_node.inputs.t1_2_mni_warp = target_warp
                mni_2_t1w_warp_node.inputs.mni_2_t1w_warp = target_inverse_warp

            # 03: register QSM scalar maps to MNI
            # qsm_scalar_maps_node: merge different file paths to a list
            qsm_scalar_maps_node = Node(Merge(8), name='qsm_scalar_maps')
            qsm_wf.connect(qsm_metrics_node, 'chisep_qsm_path', qsm_scalar_maps_node, 'in1')
            qsm_wf.connect(qsm_metrics_node, 'r2star_path', qsm_scalar_maps_node, 'in2')
            qsm_wf.connect(qsm_metrics_node, 's0_path', qsm_scalar_maps_node, 'in3')
            qsm_wf.connect(qsm_metrics_node, 't2star_path', qsm_scalar_maps_node, 'in4')
            qsm_wf.connect(qsm_metrics_node, 'chidia_path', qsm_scalar_maps_node, 'in5')
            qsm_wf.connect(qsm_metrics_node, 'chipara_path', qsm_scalar_maps_node, 'in6')
            qsm_wf.connect(qsm_metrics_node, 'chitotal_path', qsm_scalar_maps_node, 'in7')
            qsm_wf.connect(qsm_metrics_node, 'oef_path', qsm_scalar_maps_node, 'in8')

            # filter_existing_node = Node(FilterExisting(), name='filter_existing_qsm_scalars')
            # qsm_wf.connect(qsm_scalar_maps_node, 'out', filter_existing_node, 'input_file_list')

            output1_filenames = [
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-QSMnet_Chimap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_R2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_S0map.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_T2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_Chidia.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_Chipara.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_Chitotal.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_OEF.nii.gz'
            ]

            output2_filenames = [
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_desc-QSMnet_Chimap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_R2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_S0map.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_T2starmap.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_Chidia.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_Chipara.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_Chitotal.nii.gz',
                f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_OEF.nii.gz'
            ]

            qsm_to_mni_register_node = Node(QSMRegister(), name='qsm_to_mni_registration')
            qsm_wf.connect(inputnode, 'in_t1', qsm_to_mni_register_node, 't1w')
            qsm_wf.connect(t1_2_mni_warp_node, 't1_2_mni_warp', qsm_to_mni_register_node, 't1w_to_mni_warp')
            qsm_wf.connect(mag_to_t1w_register_node, 'source_to_target_mat', qsm_to_mni_register_node, 'qsm_to_t1w_affine')
            qsm_to_mni_register_node.inputs.output_dir = os.path.join(self.output_path, 'QSM_registered')
            qsm_wf.connect(qsm_scalar_maps_node, 'out', qsm_to_mni_register_node, 'input')
            qsm_to_mni_register_node.inputs.output1 = output1_filenames
            qsm_to_mni_register_node.inputs.output2 = output2_filenames

            if self.qsm_metrics_stats:
                os.makedirs(os.path.join(self.output_path, 'qsm_metrics_stats'), exist_ok=True)

                # Check segs
                # synthseg_t1w: self.session.anat_seg_dir/synthseg/*_synthseg.nii.gz (should be 1 file)
                synthseg_t1w_file = glob.glob(os.path.join(self.session.anat_seg_dir, 'synthseg', '*_synthseg.nii.gz'))
                if not synthseg_t1w_file:
                    synthseg = False
                if len(synthseg_t1w_file) > 1:
                    raise ValueError("[QSM Pipeline] Multiple synthseg files found.")
                synthseg_t1w = synthseg_t1w_file[0]
                synthseg = True

                # dk_fs: self.freesurfer_dir/mri/aparc+aseg.mgz (should be 1 file)
                dk_fs_file = glob.glob(os.path.join(self.session.freesurfer_dir, 'mri', 'aparc+aseg.mgz'))
                if not dk_fs_file:
                    dk = False
                if len(dk_fs_file) > 1:
                    raise ValueError("[QSM Pipeline] Multiple aparc+aseg files found.")
                dk_fs = dk_fs_file[0]
                dk = True

                mni_to_t1w = True

                if synthseg:
                    scalar_maps_for_synthseg = Node(CalculateScalarMaps(), name='scalar_maps_for_synthseg')
                    qsm_wf.connect(qsm_to_mni_register_node, 'outputs_in_t1w', scalar_maps_for_synthseg, 'data_files')
                    scalar_maps_for_synthseg.inputs.mask_file = synthseg_t1w
                    scalar_maps_for_synthseg.inputs.colnames = ['Chi_QSMnet', 'R2star', 'S0', 'T2star', 'ChiDia', 'ChiPara', 'ChiTotal', 'OEF']
                    scalar_maps_for_synthseg.inputs.output_csv = os.path.join(self.output_path, 'qsm_metrics_stats', 'scalar_maps_for_synthseg.csv')
                    scalar_maps_for_synthseg.inputs.ignore_background = True
                    scalar_maps_for_synthseg.inputs.statistic = 'mean'
                
                if dk:
                    fs_to_t1w_xfm_node = Node(Tkregister2fs2t1w(), name='fs_to_t1w_xfm')
                    qsm_wf.connect(inputnode, 'fs_subjects_dir', fs_to_t1w_xfm_node, 'fs_subjects_dir')
                    qsm_wf.connect(inputnode, 'fs_subject_id', fs_to_t1w_xfm_node, 'fs_subject_id')
                    fs_to_t1w_xfm_node.inputs.output_matrix = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-fs_to-T1w_xfm.mat")
                    fs_to_t1w_xfm_node.inputs.output_inverse_matrix = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-fs_xfm.mat")

                    # aparc+aseg.mgz -> aparc+aseg.nii.gz -> T1w space
                    aseg_mgz_to_nifti_node = Node(MRIConvert(), name='aseg_mgz_to_nifti')
                    aseg_mgz_to_nifti_node.inputs.in_file = dk_fs
                    aseg_mgz_to_nifti_node.inputs.out_file = os.path.join(self.output_path, 'qsm_metrics_stats', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-fs_aparcaseg.nii.gz")

                    fs_aparcaseg_to_t1w_node = Node(FLIRT(), name='fs_aparcaseg_to_t1w')
                    qsm_wf.connect(aseg_mgz_to_nifti_node, 'out_file', fs_aparcaseg_to_t1w_node, 'in_file')
                    qsm_wf.connect(fs_to_t1w_xfm_node, 'output_matrix', fs_aparcaseg_to_t1w_node, 'in_matrix_file')
                    qsm_wf.connect(inputnode, 'in_t1', fs_aparcaseg_to_t1w_node, 'reference')
                    fs_aparcaseg_to_t1w_node.inputs.interp = 'nearestneighbour'
                    fs_aparcaseg_to_t1w_node.inputs.apply_xfm = True
                    fs_aparcaseg_to_t1w_node.inputs.out_file = os.path.join(self.output_path, 'qsm_metrics_stats', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_aparcaseg.nii.gz")

                    scalar_maps_for_dk = Node(CalculateScalarMaps(), name='scalar_maps_for_dk')
                    qsm_wf.connect(qsm_to_mni_register_node, 'outputs_in_t1w', scalar_maps_for_dk, 'data_files')
                    qsm_wf.connect(fs_aparcaseg_to_t1w_node, 'out_file', scalar_maps_for_dk, 'mask_file')
                    scalar_maps_for_dk.inputs.colnames = ['Chi_QSMnet', 'R2star', 'S0', 'T2star', 'ChiDia', 'ChiPara', 'ChiTotal', 'OEF']
                    scalar_maps_for_dk.inputs.output_csv = os.path.join(self.output_path, 'qsm_metrics_stats', 'scalar_maps_for_dk.csv')
                    scalar_maps_for_dk.inputs.ignore_background = True
                    scalar_maps_for_dk.inputs.statistic = 'mean'
                
                if mni_to_t1w:
                    jhu_to_t1w_transform = Node(MRIConvertApplyWarp(), name='jhu_to_t1w_transform')
                    qsm_wf.connect(mni_2_t1w_warp_node, 'mni_2_t1w_warp', jhu_to_t1w_transform, 'warp_image')
                    jhu_to_t1w_transform.inputs.input_image = get_package_path('data', 'standard', 'JHU', 'JHU-ICBM-labels-1mm.nii.gz')
                    jhu_to_t1w_transform.inputs.output_image = os.path.join(self.output_path, 'qsm_metrics_stats', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-JHU_atlas.nii.gz")
                    jhu_to_t1w_transform.inputs.interp = 'nearest'

                    scalar_maps_for_jhu = Node(CalculateScalarMaps(), name='scalar_maps_for_jhu')
                    qsm_wf.connect(qsm_to_mni_register_node, 'outputs_in_t1w', scalar_maps_for_jhu, 'data_files')
                    qsm_wf.connect(jhu_to_t1w_transform, 'output_image', scalar_maps_for_jhu, 'mask_file')
                    scalar_maps_for_jhu.inputs.colnames = ['Chi_QSMnet', 'R2star', 'S0', 'T2star', 'ChiDia', 'ChiPara', 'ChiTotal', 'OEF']
                    scalar_maps_for_jhu.inputs.output_csv = os.path.join(self.output_path, 'qsm_metrics_stats', 'scalar_maps_for_jhu.csv')
                    scalar_maps_for_jhu.inputs.ignore_background = True
                    scalar_maps_for_jhu.inputs.statistic = 'mean'

        return qsm_wf

    def extract_results(self):
        import os
        import re
        import xml.etree.ElementTree as ET
        import pandas as pd

        os.makedirs(self.output_path, exist_ok=True)
        qsm_output_path = self.extract_from

        if not qsm_output_path or not os.path.exists(qsm_output_path):
            raise FileNotFoundError(f"extract_from does not exist: {qsm_output_path}")

        # --- Parse FreeSurferColorLUT.txt ---
        fs_lut = {}
        lut_path = get_package_path('data', 'labelconvert_in', 'FreeSurferColorLUT.txt')
        with open(lut_path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                m = re.match(r'^(\d+)\s+(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', line)
                if m:
                    fs_lut[int(m.group(1))] = m.group(2).strip()

        # --- Parse JHU-labels.xml ---
        jhu_lut = {}
        xml_path = get_package_path('data', 'standard', 'JHU', 'JHU-labels.xml')
        tree = ET.parse(xml_path)
        for label_elem in tree.getroot().findall('.//label'):
            jhu_lut[int(label_elem.get('index'))] = label_elem.text.strip()

        metrics = ['Chi_QSMnet', 'R2star', 'S0', 'T2star', 'ChiDia', 'ChiPara', 'ChiTotal', 'OEF']

        # --- Helper: find all labels actually present across subjects for a given CSV type ---
        # We do two passes: first collect all unique labels, then read data with consistent columns.

        def _safe_read_csv(csv_path):
            if csv_path is None or not os.path.exists(csv_path):
                return None
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"[WARN] Failed to read CSV: {csv_path}. Error: {e}")
                return None

        def _collect_labels(csv_path):
            """Return sorted list of unique roi_labels found in a CSV."""
            df = _safe_read_csv(csv_path)
            if df is None or df.empty or 'roi_label' not in df.columns:
                return []
            return sorted(df['roi_label'].dropna().astype(int).unique().tolist())

        def _read_metrics(csv_path, lut, region_ids):
            """Read a CalculateScalarMaps CSV and return a flat dict {metric_region: value} for the given region_ids."""
            result = {f"{m}_{lut.get(rid, f'label_{rid}')}": None for rid in region_ids for m in metrics}
            df = _safe_read_csv(csv_path)
            if df is None or df.empty or 'roi_label' not in df.columns:
                return result
            for _, row in df.iterrows():
                rid = int(row['roi_label'])
                if rid not in region_ids:
                    continue
                name = lut.get(rid, f"label_{rid}")
                for m in metrics:
                    if m in df.columns:
                        val = row[m]
                        result[f"{m}_{name}"] = val if pd.notna(val) else None
            return result

        # --- Two-pass collection: first discover all labels, then read ---
        synthseg_all_labels = set()
        dk_all_labels = set()
        jhu_all_labels = set()

        subject_sessions = []
        for subj_folder in sorted(os.listdir(qsm_output_path)):
            subj_path = os.path.join(qsm_output_path, subj_folder)
            if not os.path.isdir(subj_path) or not subj_folder.startswith("sub-"):
                continue
            subj_id = subj_folder.replace("sub-", "", 1)
            ses_folders = sorted([f for f in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, f)) and f.startswith("ses-")])
            if not ses_folders:
                ses_folders = ["N/A"]
            for ses_folder in ses_folders:
                if ses_folder == "N/A":
                    ses_id = "N/A"
                    stats_dir = os.path.join(subj_path, 'qsm_metrics_stats')
                else:
                    ses_id = ses_folder.replace("ses-", "", 1)
                    stats_dir = os.path.join(subj_path, ses_folder, 'qsm_metrics_stats')

                synthseg_csv = os.path.join(stats_dir, 'scalar_maps_for_synthseg.csv')
                dk_csv = os.path.join(stats_dir, 'scalar_maps_for_dk.csv')
                jhu_csv = os.path.join(stats_dir, 'scalar_maps_for_jhu.csv')

                synthseg_all_labels.update(_collect_labels(synthseg_csv))
                dk_all_labels.update(_collect_labels(dk_csv))
                jhu_all_labels.update(_collect_labels(jhu_csv))

                subject_sessions.append((subj_id, ses_id, stats_dir))

        # Sort labels and build column lists
        synthseg_regions = sorted(synthseg_all_labels)
        dk_regions = sorted(dk_all_labels)
        jhu_regions = sorted(jhu_all_labels)

        synthseg_cols = ["Subject", "Session"] + [f"{m}_{fs_lut.get(rid, f'label_{rid}')}" for rid in synthseg_regions for m in metrics]
        dk_cols = ["Subject", "Session"] + [f"{m}_{fs_lut.get(rid, f'label_{rid}')}" for rid in dk_regions for m in metrics]
        jhu_cols = ["Subject", "Session"] + [f"{m}_{jhu_lut.get(rid, f'label_{rid}')}" for rid in jhu_regions for m in metrics]

        synthseg_rows = []
        dk_rows = []
        jhu_rows = []

        for subj_id, ses_id, stats_dir in subject_sessions:
            synthseg_csv = os.path.join(stats_dir, 'scalar_maps_for_synthseg.csv')
            dk_csv = os.path.join(stats_dir, 'scalar_maps_for_dk.csv')
            jhu_csv = os.path.join(stats_dir, 'scalar_maps_for_jhu.csv')

            row_base = {"Subject": f"sub-{subj_id}", "Session": f"ses-{ses_id}" if ses_id != "N/A" else "N/A"}

            sr = dict(row_base)
            sr.update(_read_metrics(synthseg_csv, fs_lut, synthseg_regions))
            synthseg_rows.append(sr)

            dr = dict(row_base)
            dr.update(_read_metrics(dk_csv, fs_lut, dk_regions))
            dk_rows.append(dr)

            jr = dict(row_base)
            jr.update(_read_metrics(jhu_csv, jhu_lut, jhu_regions))
            jhu_rows.append(jr)

        # --- Write xlsx ---
        synthseg_df = pd.DataFrame(synthseg_rows, columns=synthseg_cols)
        dk_df = pd.DataFrame(dk_rows, columns=dk_cols)
        jhu_df = pd.DataFrame(jhu_rows, columns=jhu_cols)

        synthseg_df.to_excel(os.path.join(self.output_path, "qsm_synthseg_summary.xlsx"), index=False)
        dk_df.to_excel(os.path.join(self.output_path, "qsm_dk_summary.xlsx"), index=False)
        jhu_df.to_excel(os.path.join(self.output_path, "qsm_jhu_summary.xlsx"), index=False)

        print(f"[QSM Pipeline] Results extracted successfully from: {qsm_output_path}")
        print(f"[QSM Pipeline] Saved {len(subject_sessions)} subjects to: {self.output_path}")
