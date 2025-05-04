import os
import shutil
import subprocess
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.ndimage import label, sum
from skimage.measure import marching_cubes, mesh_surface_area
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge
from ....bids_data.rename_bids_file import rename_bids_file
from ...common.binarize import threshold_binarize_nifti
from ...common.extract_region import extract_roi_from_nii
from ...common.copy_file import CopyFileCommandLine
from ...common.mri_synthstrip import MRISynthstripCommandLine
from ...common.delete_file import DeleteFileCommandLine

from .wmh.wmh_seg_nipype import LSTSegmentation, LSTAI, WMHSynthSeg, PrepareTrueNetData, TrueNetEvaluate, TrueNetPostProcess
from .wmh.wmh_location_nipype import Fazekas, Bullseyes
from .wmh.wmh_shape_nipype import WMHShape

from nipype.interfaces import fsl
from ...smri.fsl.fsl_anat_nipype import FSLANAT
from ...common.register import ModalityRegistration

import logging
logger = logging.getLogger(__name__)

class WMHSegmentationPipeline:
    def __init__(self, subject, session, output_path, matlab_path=None, **kwargs):
        """
        初始化 WMH 分割 pipeline
        :param subject: BIDSSubject 对象
        :param session: BIDSSession 对象
        :param output_path: str, 输出路径
        :param matlab_path: str, MATLAB 执行路径
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        if self.session is not None:
            if self.session.session_id is not None:
                session_part = f"ses-{self.session.session_id}"
            else:
                session_part = ''
            self.output_path_xfm = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', session_part)
            self.output_path_fslanat = os.path.join(self.subject.bids_dir, 'derivatives', 'fsl_anat', f'sub-{self.subject.subject_id}', session_part)

        self.use_which_t1w = kwargs.get('use_which_t1w', None)
        self.use_which_flair = kwargs.get('use_which_flair', None)
        self.matlab_path = matlab_path or kwargs.get('matlab_path', 'matlab')
        self.spm_path = kwargs.get('spm_path', None)
        self.threshold = kwargs.get('threshold', 0.5)
        self.seg_method = kwargs.get('seg_method', 'LST')
        self.location_method = kwargs.get('location_method', ['Fazekas'])
        if isinstance(self.location_method, str):
            self.location_method = [self.location_method]
        self.use_which_ventmask = kwargs.get('use_which_ventmask', 'fsl_anat')
        self.use_bianca_mask = kwargs.get('use_bianca_mask', False)
        self.shape_analysis = kwargs.get('shape_analysis', False)
        self.shape_analysis_voxel_thr = kwargs.get('shape_analysis_voxel_thr', 10)
        self.freesurfer_output_root_dir = kwargs.get('freesurfer_output_root_dir', os.path.join(os.path.dirname(self.output_path), "freesurfer"))
        self.skip_if_no_freesurfer = kwargs.get('skip_if_no_freesurfer', True)
        self.extract_from = kwargs.get('extract_from', None)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.script_path_lst = os.path.join(base_dir, 'matlab', 'wmh_seg_lst.m')
        self.script_path2 = os.path.join(base_dir, 'bash', 'wmh_fsl_preprocessing.sh')
        self.script_quantification = os.path.join(base_dir, 'bash', 'wmh_volume_quantification.sh')
        self.script_normalization = os.path.join(base_dir, 'bash', 'wmh_transform2mni.sh')
        self.script_path4 = os.path.join(os.path.dirname(__file__), 'bullseye_WMH', 'run_bullseye_pipeline.py')
        self.script_path5 = os.path.join(base_dir, 'bash', 'wmh_bullseyes_quantification.sh')
        self.script_path_register = os.path.join(base_dir, 'bash', 'register.sh')
        self.script_path_fslpreproc = os.path.join(base_dir, 'bash', 'wmh_fsl_preprocessing2.sh')

    def check_data_requirements(self):
        """
        检查数据需求
        :return: bool
        """
        return self.session.get_flair_files() is not None
    
    def create_workflow(self):
        flair_files = self.session.get_flair_files()
        if self.use_which_flair:
            flair_files = [f for f in flair_files if self.use_which_flair in f]
            if len(flair_files) != 1:
                raise FileNotFoundError(f"No specific FLAIR file found for {self.use_which_flair} or more than one found.")
            flair_file = flair_files[0]
        else:
            flair_files = [flair_files[0]]
            flair_file = flair_files[0]
            print(f"No specific FLAIR file selected. Using the first one: {flair_file}.")
        
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
            
            t1w_filename = os.path.basename(t1w_file).split(".")[0]
        
        if t1w_file is None:
            t1w_file = ''

        os.makedirs(self.output_path, exist_ok=True)

        flair_filename = os.path.basename(flair_file).split(".")[0]
        session_entity = f"_ses-{self.session.session_id}" if self.session.session_id else ""
        placeholder = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_Placeholder.nii.gz')

        wmh_workflow = Workflow(name="WMHSegmentationPipeline")
        wmh_workflow.base_dir = self.output_path

        inputnode = Node(IdentityInterface(fields=["flair_file", "t1w_file", "threshold", "output_path",
                                                   "matlab_path", "spm_path", "lst_seg_script"]), 
                         name="inputnode")
        
        inputnode.inputs.flair_file = flair_file
        inputnode.inputs.t1w_file = t1w_file
        inputnode.inputs.lst_seg_script = self.script_path_lst
        '''
        1. WMH Segmentation
        '''
        seg_threshold = self.threshold
        thr_string = f'{seg_threshold:.2f}'.replace('.', 'p')

        inputnode.inputs.threshold = seg_threshold
        inputnode.inputs.output_path = self.output_path
        inputnode.inputs.matlab_path = self.matlab_path
        inputnode.inputs.spm_path = self.spm_path

        wmh_filename_entities = {
            'space': 'FLAIR',
            'desc': f"{self.seg_method}Thr{thr_string}",
        }
        binarized_wmh_filename = rename_bids_file(placeholder, wmh_filename_entities, 'WMHmask', '.nii.gz')
        probmap_filename = rename_bids_file(placeholder, wmh_filename_entities, 'WMHprobmap', '.nii.gz')
        wmh_synthseg_filename = rename_bids_file(placeholder, wmh_filename_entities, 'WMHSynthSeg', '.nii.gz')

        wmh_mask_node = Node(IdentityInterface(fields=["wmh_mask", "wmh_prob_map"]), name="wmh_mask_node")

        if self.seg_method == 'LST':
            wmh_lst_seg_node = Node(LSTSegmentation(), name="wmh_lst_seg")
            wmh_lst_seg_node.inputs.output_mask_name = os.path.join(self.output_path, binarized_wmh_filename)
            wmh_lst_seg_node.inputs.output_prob_map_name = os.path.join(self.output_path, probmap_filename)

            wmh_workflow.connect([
                (inputnode, wmh_lst_seg_node, [("flair_file", "flair_img"),
                                               ("lst_seg_script", "seg_script"),
                                               ("threshold", "threshold"),
                                               ("output_path", "output_path"),
                                               ("matlab_path", "matlab_path"),
                                               ("spm_path", "spm_path")]),
                (wmh_lst_seg_node, wmh_mask_node, [("wmh_mask", "wmh_mask"),
                                                    ("wmh_prob_map", "wmh_prob_map")]),
            ])
        elif self.seg_method == 'LST-AI':
            wmh_lst_ai_node = Node(LSTAI(), name="wmh_lst_ai")
            wmh_lst_ai_node.inputs.output_dir = os.path.join(self.output_path, 'lst_ai_output')
            wmh_lst_ai_node.inputs.temp_dir = os.path.join(self.output_path, 'lst_ai_output')
            wmh_lst_ai_node.inputs.img_stripped = True
            wmh_lst_ai_node.inputs.save_prob_map = True

            lst_ai_t1w_mrisynthstrip_node = Node(MRISynthstripCommandLine(), name="lst_ai_t1w_mrisynthstrip")
            lst_ai_t1w_mrisynthstrip_node.inputs.output_file = os.path.join(self.output_path, 't1w_brain.nii.gz')

            lst_ai_flair_mrisynthstrip_node = Node(MRISynthstripCommandLine(), name="lst_ai_flair_mrisynthstrip")
            lst_ai_flair_mrisynthstrip_node.inputs.output_file = os.path.join(self.output_path, 'flair_brain.nii.gz')

            copy_lst_ai_probmap_node = Node(CopyFileCommandLine(), name="copy_lst_ai_probmap")
            copy_lst_ai_probmap_node.inputs.output_file = os.path.join(self.output_path, probmap_filename)

            copy_lst_ai_wmhmask_node = Node(CopyFileCommandLine(), name="copy_lst_ai_wmhmask")
            copy_lst_ai_wmhmask_node.inputs.output_file = os.path.join(self.output_path, binarized_wmh_filename)

            delete_lst_ai_t1w_mrisynthstrip_node = Node(DeleteFileCommandLine(), name="delete_lst_ai_t1w_mrisynthstrip")
            delete_lst_ai_flair_mrisynthstrip_node = Node(DeleteFileCommandLine(), name="delete_lst_ai_flair_mrisynthstrip")

            wmh_workflow.connect([
                (inputnode, lst_ai_t1w_mrisynthstrip_node, [("t1w_file", "input_file")]),
                (inputnode, lst_ai_flair_mrisynthstrip_node, [("flair_file", "input_file")]),
                (lst_ai_t1w_mrisynthstrip_node, wmh_lst_ai_node, [("output_file", "t1w_img")]),
                (lst_ai_flair_mrisynthstrip_node, wmh_lst_ai_node, [("output_file", "flair_img")]),
                (inputnode, wmh_lst_ai_node, [("threshold", "threshold")]),
                (wmh_lst_ai_node, copy_lst_ai_probmap_node, [("wmh_prob_map", "input_file")]),
                (wmh_lst_ai_node, copy_lst_ai_wmhmask_node, [("wmh_mask", "input_file")]),
                (wmh_lst_ai_node, delete_lst_ai_t1w_mrisynthstrip_node, [("input_t1w_img", "file")]),
                (wmh_lst_ai_node, delete_lst_ai_flair_mrisynthstrip_node, [("input_flair_img", "file")]),
                (copy_lst_ai_probmap_node, wmh_mask_node, [("output_file", "wmh_prob_map")]),
                (copy_lst_ai_wmhmask_node, wmh_mask_node, [("output_file", "wmh_mask")]),
            ])
        elif self.seg_method == 'WMHSynthSeg':
            wmh_filename_entities = {
                'space': 'FLAIR',
                'desc': f"{self.seg_method}",
            }
            binarized_wmh_filename = rename_bids_file(placeholder, wmh_filename_entities, 'WMHmask', '.nii.gz')
            probmap_filename = rename_bids_file(placeholder, wmh_filename_entities, 'WMHprobmap', '.nii.gz')
            wmh_synthseg_filename = rename_bids_file(placeholder, wmh_filename_entities, 'WMHSynthSeg', '.nii.gz')

            wmh_synthseg_node = Node(WMHSynthSeg(), name="wmh_synthseg")
            wmh_synthseg_node.inputs.output_mask_name = os.path.join(self.output_path, binarized_wmh_filename)
            wmh_synthseg_node.inputs.output_prob_map_name = os.path.join(self.output_path, probmap_filename)
            wmh_synthseg_node.inputs.seg_name = os.path.join(self.output_path, wmh_synthseg_filename)

            wmh_workflow.connect([
                (inputnode, wmh_synthseg_node, [("flair_file", "flair_img"),
                                                ("output_path", "output_dir")]),
                (wmh_synthseg_node, wmh_mask_node, [("wmh_mask", "wmh_mask"),
                                                     ("wmh_prob_map", "wmh_prob_map")]),
            ])
        elif self.seg_method == 'truenet':
            truenet_preprocess_dir = os.path.join(self.output_path, 'truenet_preprocess')
            os.makedirs(truenet_preprocess_dir, exist_ok=True)

            wmh_truenet_preprocess_node = Node(PrepareTrueNetData(), name="truenet_preprocess")
            wmh_truenet_preprocess_node.inputs.outname = os.path.join(truenet_preprocess_dir, 'truenet_preprocess')
            wmh_truenet_preprocess_node.inputs.verbose = True

            wmh_workflow.connect([
                (inputnode, wmh_truenet_preprocess_node, [("flair_file", "FLAIR"),
                                                ("t1w_file", "T1")]),
            ])

            truenet_evaluate_node = Node(TrueNetEvaluate(), name="truenet_evaluate")
            wmh_workflow.connect([
                (wmh_truenet_preprocess_node, truenet_evaluate_node, [("output_dir", "inp_dir")]),
            ])
            truenet_evaluate_node.inputs.model_name = 'mwsc'
            truenet_evaluate_node.inputs.output_dir = self.output_path

            truenet_postprocess_node = Node(TrueNetPostProcess(), name="truenet_postprocess")
            wmh_workflow.connect([
                (truenet_evaluate_node, truenet_postprocess_node, [("pred_file", "pred_file")]),
                (wmh_truenet_preprocess_node, truenet_postprocess_node, [("output_dir", "preprocess_dir")]),
                (truenet_postprocess_node, wmh_mask_node, [("wmh_mask", "wmh_mask")]),
            ])
            truenet_postprocess_node.inputs.output_dir = self.output_path
            truenet_postprocess_node.inputs.threshold = seg_threshold
            truenet_postprocess_node.inputs.output_mask_name = os.path.join(self.output_path, binarized_wmh_filename)
            truenet_postprocess_node.inputs.output_prob_map_name = os.path.join(self.output_path, probmap_filename)
        else:
            #raise ValueError(f"Unknown segmentation method: {self.seg_method} \n Please choose from ['LST', 'WMHSynthSeg', 'truenet'].")
            logger.warning('Only FLAIR to T1w registeration will perform if the two exist')

        '''
        2. WMH Location Quantification
        '''

        flair_in_t1w_entities = {
            'space': 'T1w',
        }
        flair_to_t1w_mat_entities = {
            'from': 'FLAIR',
            'to': 'T1w',
        }
        t1w_to_flair_mat_entities = {
            'from': 'T1w',
            'to': 'FLAIR',
        }
                
        if t1w_file is not None and t1w_file != '':
            xfm_output_path = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
            os.makedirs(xfm_output_path, exist_ok=True)                
            
            flair_to_t1w_xfm_node = Node(ModalityRegistration(), name="flair_to_t1w_xfm")
            # Here we assume T1w usually has a equal or better resolution than FLAIR
            wmh_workflow.connect([
                (inputnode, flair_to_t1w_xfm_node, [("flair_file", "image_source"),
                                                    ("t1w_file", "image_target")])
            ])
            flair_to_t1w_xfm_node.inputs.image_target_strip = 0
            flair_to_t1w_xfm_node.inputs.image_source_strip = 0
            flair_to_t1w_xfm_node.inputs.flirt_direction = 1
            flair_to_t1w_xfm_node.inputs.output_dir = xfm_output_path
            flair_to_t1w_xfm_node.inputs.registered_image_filename = rename_bids_file(flair_file, flair_in_t1w_entities, 'FLAIR', '.nii.gz')
            flair_to_t1w_xfm_node.inputs.source_to_target_mat_filename = rename_bids_file(placeholder, flair_to_t1w_mat_entities, 'xfm', '.mat')
            flair_to_t1w_xfm_node.inputs.target_to_source_mat_filename = rename_bids_file(placeholder, t1w_to_flair_mat_entities, 'xfm', '.mat')
            flair_to_t1w_xfm_node.inputs.dof = 6

        if self.location_method is not None:
            if 'Fazekas' in self.location_method:                
                fazekas_classification_node = Node(Fazekas(), name="fazekas_preprocess")
                fazekas_classification_node.inputs.use_which_ventmask = self.use_which_ventmask
                fazekas_classification_node.inputs.use_bianca_mask = self.use_bianca_mask
                wmh_filtered_filename_entities = {
                    'space': 'FLAIR',
                    'desc': 'FilteredByBianca'
                }
                fazekas_classification_node.inputs.wmh_mask_filename = rename_bids_file(placeholder, wmh_filtered_filename_entities, 'WMHmask', '.nii.gz')
                fazekas_classification_node.inputs.pwmh_mask_filename = rename_bids_file(placeholder, wmh_filename_entities, 'PWMHmask', '.nii.gz')
                fazekas_classification_node.inputs.dwmh_mask_filename = rename_bids_file(placeholder, wmh_filename_entities, 'DWMHmask', '.nii.gz')

                if self.use_which_ventmask == 'fsl_anat':
                    if t1w_file is None or t1w_file == '':
                        raise ValueError("Fazekas method requires T1w image. Please provide a T1w image or choose another method.")
                    
                    os.makedirs(self.output_path_fslanat, exist_ok=True)

                    fsl_anat_node = Node(FSLANAT(), name="fsl_anat")
                    wmh_workflow.connect(inputnode, "t1w_file", fsl_anat_node, "input_image")
                    fsl_anat_node.inputs.output_directory = os.path.join(self.subject.bids_dir, 'derivatives', 'fsl_anat', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'fsl')

                    wmh_workflow.connect([
                        (fsl_anat_node, fazekas_classification_node, [("output_directory", "fsl_anat_dir")]),
                        (inputnode, fazekas_classification_node, [("output_path", "output_dir"),
                                                            ("flair_file", "flair_img")]),
                        (wmh_mask_node, fazekas_classification_node, [("wmh_mask", "wmh_img")]),
                        (flair_to_t1w_xfm_node, fazekas_classification_node, [("target_to_source_mat", "t1_to_flair_xfm")]),
                    ])

                    fsl_anat_flair_entities = {
                        'space': 'FLAIR',
                        'desc': 'fslanat'
                    }

                    fazekas_classification_node.inputs.bianca_mask_filename = rename_bids_file(t1w_file, fsl_anat_flair_entities, 'BiancaMask', '.nii.gz')
                    fazekas_classification_node.inputs.vent_mask_filename = rename_bids_file(t1w_file, fsl_anat_flair_entities, 'VentMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_3mm_filename = rename_bids_file(t1w_file, fsl_anat_flair_entities, '3mmPeriventricularMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_10mm_filename = rename_bids_file(t1w_file, fsl_anat_flair_entities, '10mmPeriventricularMask', '.nii.gz')
                elif self.use_which_ventmask == 'WMHSynthSeg':
                    wmh_synthseg_flair_entities = {
                        'space': 'FLAIR',
                        'desc': 'WMHSynthSeg'
                    }

                    wmh_workflow.connect([
                        (inputnode, fazekas_classification_node, [("output_path", "output_dir"),
                                                            ("flair_file", "flair_img")]),
                        (wmh_mask_node, fazekas_classification_node, [("wmh_mask", "wmh_img")]),
                    ])

                    wmh_workflow.connect([
                        (wmh_synthseg_node, fazekas_classification_node, [("wmh_synthseg", "wmh_synthseg")]),
                    ])
                    fazekas_classification_node.inputs.vent_mask_filename = rename_bids_file(flair_file, wmh_synthseg_flair_entities, 'VentMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_3mm_filename = rename_bids_file(flair_file, wmh_synthseg_flair_entities, '3mmPeriventricularMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_10mm_filename = rename_bids_file(flair_file, wmh_synthseg_flair_entities, '10mmPeriventricularMask', '.nii.gz')
                elif self.use_which_ventmask == 'SynthSeg':
                    synthseg_flair_entities = {
                        'space': 'FLAIR',
                        'desc': 'SynthSeg'
                    }

                    wmh_workflow.connect([
                        (inputnode, fazekas_classification_node, [("output_path", "output_dir"),
                                                            ("flair_file", "flair_img")]),
                        (wmh_mask_node, fazekas_classification_node, [("wmh_mask", "wmh_img")]),
                    ])

                    fazekas_classification_node.inputs.vent_mask_filename = rename_bids_file(flair_file, synthseg_flair_entities, 'VentMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_3mm_filename = rename_bids_file(flair_file, synthseg_flair_entities, '3mmPeriventricularMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_10mm_filename = rename_bids_file(flair_file, synthseg_flair_entities, '10mmPeriventricularMask', '.nii.gz')
                
                fazekas_classification_node.inputs.wmh_mask_vol_filename = f'sub-{self.subject.subject_id}{session_entity}_TotalWMHVolume.csv'
                fazekas_classification_node.inputs.pwmh_mask_vol_filename = f'sub-{self.subject.subject_id}{session_entity}_PWMHVolume.csv'
                fazekas_classification_node.inputs.dwmh_mask_vol_filename = f'sub-{self.subject.subject_id}{session_entity}_DWMHVolume.csv'
            elif 'bullseyes' in self.location_method:
                bullseyes_node = Node(Bullseyes(), name="bullseyes")

                if self.session.freesurfer_dir is None:
                    raise ValueError("Freesurfer directory is not available. Please run Freesurfer first.")
                else:
                    bullseyes_node.inputs.fs_output_dir = os.path.dirname(self.session.freesurfer_dir)
                    bullseyes_node.inputs.fs_output_id = os.path.basename(self.session.freesurfer_dir)

                    wmh_workflow.connect([
                        (inputnode, bullseyes_node, [("flair_file", "flair_img"),
                                                     ("output_path", "output_dir")]),
                        (wmh_mask_node, bullseyes_node, [("wmh_mask", "wmh_img")]),
                        (flair_to_t1w_xfm_node, bullseyes_node, [("target_to_source_mat", "t1_to_flair_xfm")]),
                    ])

                    bullseye_wmparc_entities = {
                        'space': 'FLAIR'
                    }

                    fs_to_flair_xfm_entities = {
                        'from': 'fs',
                        'to': 'FLAIR'
                    }

                    fs_to_t1w_xfm_entities = {
                        'from': 'fs',
                        'to': 'T1w'
                    }

                    bullseyes_node.inputs.bullseye_wmparc_filename = rename_bids_file(flair_file, bullseye_wmparc_entities, 'BullseyesWMparc', '.nii.gz')
                    bullseyes_node.inputs.xfm_output_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}{session_entity}')
                    bullseyes_node.inputs.fs_to_flair_xfm_filename = rename_bids_file(placeholder, fs_to_flair_xfm_entities, 'xfm', '.mat')
                    bullseyes_node.inputs.fs_to_t1w_xfm_filename = rename_bids_file(placeholder, fs_to_t1w_xfm_entities, 'xfm', '.mat')
        
        '''
        3. Shape Analysis
        '''
        if self.shape_analysis:
            if t1w_file == '':
                raise ValueError("Shape analysis requires 3D T1w image. Please provide a T1w image or choose another method.")
            
            shape_features_subdir = os.path.join(self.output_path, 'shape_analysis')
            os.makedirs(shape_features_subdir, exist_ok=True)

            wmh_labeled_filename_entities = {
                'space': 'MNI152'
            }

            # transform WMH masks to MNI space
            # MNI template: $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz
            mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_1mm_brain.nii.gz')

            # First, we need to transform the WMH mask to T1w space
            wmh_in_t1w_entities = {
                'space': 'T1w',
            }

            wmh_in_mni_entities = {
                'space': 'MNI152',
            }

            transform_pwmh_to_t1w_node = Node(fsl.FLIRT(), name="transform_pwmh_to_t1w")
            wmh_workflow.connect([
                (fazekas_classification_node, transform_pwmh_to_t1w_node, [("pwmh_mask", "in_file")]),
                (flair_to_t1w_xfm_node, transform_pwmh_to_t1w_node, [("source_to_target_mat", "in_matrix_file")]),
                (inputnode, transform_pwmh_to_t1w_node, [("t1w_file", "reference")]),
            ])
            transform_pwmh_to_t1w_node.inputs.interp = 'nearestneighbour'
            transform_pwmh_to_t1w_node.inputs.apply_xfm = True
            transform_pwmh_to_t1w_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_in_t1w_entities, 'PWMHmask', '.nii.gz'))

            transform_pwmh_to_mni_node = Node(fsl.ApplyWarp(), name="transform_pwmh_to_mni")
            wmh_workflow.connect([
                (transform_pwmh_to_t1w_node, transform_pwmh_to_mni_node, [("out_file", "in_file")]),
                (fsl_anat_node, transform_pwmh_to_mni_node, [("t1w_to_mni_nonlin_field", "field_file")]),
            ])
            transform_pwmh_to_mni_node.inputs.interp = 'nn'
            transform_pwmh_to_mni_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_in_mni_entities, 'PWMHmask', '.nii.gz'))
            transform_pwmh_to_mni_node.inputs.ref_file = mni_template

            pwmh_shape_analysis_node = Node(WMHShape(), name="pwmh_shape_analysis")
            pwmh_shape_analysis_node.inputs.threshold = 10
            pwmh_shape_analysis_node.inputs.save_plots = False
            pwmh_shape_analysis_node.inputs.output_dir = shape_features_subdir
            wmh_workflow.connect([
                (transform_pwmh_to_mni_node, pwmh_shape_analysis_node, [("out_file", "wmh_mask")]),
            ])

            pwmh_shape_analysis_node.inputs.wmh_labeled_filename = rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_labeled_filename_entities, 'PWMHlabel', '.nii.gz')
            pwmh_shape_analysis_node.inputs.shape_csv_filename = f"sub-{self.subject.subject_id}{session_entity}_PWMHshape.csv"
            pwmh_shape_analysis_node.inputs.shape_csv_avg_filename = f"sub-{self.subject.subject_id}{session_entity}_avgPWMHshape.csv"

            transform_dwmh_to_t1w_node = Node(fsl.FLIRT(), name="transform_dwmh_to_t1w")
            wmh_workflow.connect([
                (fazekas_classification_node, transform_dwmh_to_t1w_node, [("dwmh_mask", "in_file")]),
                (flair_to_t1w_xfm_node, transform_dwmh_to_t1w_node, [("source_to_target_mat", "in_matrix_file")]),
                (inputnode, transform_dwmh_to_t1w_node, [("t1w_file", "reference")]),
            ])
            transform_dwmh_to_t1w_node.inputs.interp = 'nearestneighbour'
            transform_dwmh_to_t1w_node.inputs.apply_xfm = True
            transform_dwmh_to_t1w_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_in_t1w_entities, 'DWMHmask', '.nii.gz'))

            transform_dwmh_to_mni_node = Node(fsl.ApplyWarp(), name="transform_dwmh_to_mni")
            wmh_workflow.connect([
                (transform_dwmh_to_t1w_node, transform_dwmh_to_mni_node, [("out_file", "in_file")]),
                (fsl_anat_node, transform_dwmh_to_mni_node, [("t1w_to_mni_nonlin_field", "field_file")]),
            ])
            transform_dwmh_to_mni_node.inputs.interp = 'nn'
            transform_dwmh_to_mni_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_in_mni_entities, 'DWMHmask', '.nii.gz'))
            transform_dwmh_to_mni_node.inputs.ref_file = mni_template

            dwmh_shape_analysis_node = Node(WMHShape(), name="dwmh_shape_analysis")
            dwmh_shape_analysis_node.inputs.threshold = 10
            dwmh_shape_analysis_node.inputs.save_plots = False
            dwmh_shape_analysis_node.inputs.output_dir = shape_features_subdir
            wmh_workflow.connect([
                (transform_dwmh_to_mni_node, dwmh_shape_analysis_node, [("out_file", "wmh_mask")]),
            ])
            dwmh_shape_analysis_node.inputs.wmh_labeled_filename = rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_labeled_filename_entities, 'DWMHlabel', '.nii.gz')
            dwmh_shape_analysis_node.inputs.shape_csv_filename = f"sub-{self.subject.subject_id}{session_entity}_DWMHshape.csv"
            dwmh_shape_analysis_node.inputs.shape_csv_avg_filename = f"sub-{self.subject.subject_id}{session_entity}_avgDWMHshape.csv"

        return wmh_workflow

    # def run(self):
    #     flair_files = self.session.get_flair_files()
    #     if self.use_which_flair:
    #         flair_files = [f for f in flair_files if self.use_which_flair in f]
    #         # 确保最终只有1个合适的文件
    #         if len(flair_files) != 1:
    #             raise FileNotFoundError(f"No specific FLAIR file found for {self.use_which_flair} or more than one found.")
    #         flair_file = flair_files[0]
    #     else:
    #         flair_files = [flair_files[0]]
    #         flair_file = flair_files[0]
    #         print(f"No specific FLAIR file selected. Using the first one: {flair_file}.")
        
    #     # self.session.get_t1w_files()不是空列表时
    #     if self.session.get_t1w_files():
    #         t1w_files = self.session.get_t1w_files()
    #         if self.use_which_t1w:
    #             t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
    #             # 确保最终只有1个合适的文件
    #             if len(t1w_files) != 1:
    #                 raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
    #             t1w_file = t1w_files[0]
    #         else:
    #             t1w_files = [t1w_files[0]]
    #             t1w_file = t1w_files[0]
    #             print(f"No specific T1w file selected. Using the first one: {t1w_file}.")
            
    #         t1w_filename = os.path.basename(t1w_file).split(".")[0]

    #     # 确保输出目录存在
    #     os.makedirs(self.output_path, exist_ok=True)
    #     os.makedirs(self.output_path_xfm, exist_ok=True)
    #     os.makedirs(self.output_path_fslanat, exist_ok=True)

    #     flair_filename = os.path.basename(flair_file).split(".")[0]
    #     session_entity = f"_ses-{self.session.session_id}" if self.session.session_id else ""

    #     """
    #     1. WMH 分割
    #     """
    #     thresholds = f"[{', '.join(map(str, self.thresholds))}]"
    #     seg_threshold = self.thresholds[-1]
    #     thr_string = f'{seg_threshold:.2f}'.replace('.', 'p')

    #     if self.seg_method == 'LST':
    #         analysis_space = 'T1w'
    #         matlab_command = f"addpath('{os.path.dirname(self.script_path1)}'); wmh_seg_lst('{flair_file}', {thresholds}, '{self.output_path}'); exit;"

    #         cmd = [
    #             self.matlab_path,
    #             "-nodesktop",  # 不启动桌面
    #             "-nosplash",  # 禁用 splash 屏幕
    #             "-r",  # 执行指定的命令
    #             f"{matlab_command}"  # 执行后退出 MATLAB
    #         ]

    #         print("Running WMH segmentation...")
    #         try:
    #             subprocess.run(cmd, check=True)
    #             print(f"WMH segmentation completed. Results saved to {self.output_path}")
    #         except subprocess.CalledProcessError as e:
    #             print(f"Error occurred while running MATLAB script: {e}")
            
    #         binarized_wmh_file_old = os.path.join(self.output_path, f'bles_{self.thresholds[-1]}_lpa_mFLAIR.nii.gz')
    #         probmap_file_old = os.path.join(self.output_path, 'ples_lpa_mFLAIR.nii.gz')

    #         # delete intermediate files
    #         lpa_mat = os.path.join(self.output_path, 'LST_lpa_mFLAIR.mat')
    #         mflair = os.path.join(self.output_path, 'mFLAIR.nii.gz')
    #         os.remove(lpa_mat)
    #         os.remove(mflair)

    #         binarized_wmh_file = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-FLAIR_WMHmask.nii.gz')
    #         binarized_wmh_filename = f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-FLAIR_WMHmask'

    #         probmap_file = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}_space-FLAIR_WMHprobmap.nii.gz')
    #         probmap_filename = f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}_space-FLAIR_WMHprobmap'

    #         # rename the output files
    #         os.rename(binarized_wmh_file_old, binarized_wmh_file)
    #         os.rename(probmap_file_old, probmap_file)
    #     elif self.seg_method == 'WMHSynthSeg':
    #         analysis_space = 'WMHSynthSeg'
    #         # copy the flair file to the output directory, in a folder named 'input4WMHSynthSeg'
    #         input4WMHSynthSeg_dir = os.path.join(self.output_path, 'input4WMHSynthSeg')
    #         os.makedirs(input4WMHSynthSeg_dir, exist_ok=True)
    #         shutil.copy(flair_file, os.path.join(input4WMHSynthSeg_dir, f'{flair_filename}.nii.gz'))

    #         cmd = [
    #             'mri_WMHsynthseg',
    #             '--i', input4WMHSynthSeg_dir,
    #             '--o', self.output_path,
    #             '--csv_vols', os.path.join(self.output_path, 'WMHSynthSegVols.csv'),
    #             '--device', 'cuda',
    #             '--crop',
    #             '--save_lesion_probabilities'
    #         ]

    #         try:
    #             subprocess.run(cmd, check=True)
    #             print(f"WMH segmentation completed. Results saved to {self.output_path}")
    #         except subprocess.CalledProcessError as e:
    #             print(f"Error occurred while running WMHSynthSeg: {e}")
            
    #         volume_csv = os.path.join(self.output_path, 'WMHSynthSegVols.csv')
    #         probmap_file_old = os.path.join(self.output_path, f'{flair_filename}_seg.lesion_probs.nii.gz')
    #         seg_file_old = os.path.join(self.output_path, f'{flair_filename}_seg.nii.gz')

    #         entities_WMHSynthSeg = {
    #             'space': 'WMHSynthSeg',
    #             'desc': self.seg_method
    #         }
    #         seg_file = os.path.join(self.output_path, rename_bids_file(flair_filename, entities_WMHSynthSeg, 'WMHSynthSeg', '.nii.gz'))
    #         probmap_file = os.path.join(self.output_path, rename_bids_file(flair_filename, entities_WMHSynthSeg, 'WMHprobmap', '.nii.gz'))
    #         os.rename(probmap_file_old, probmap_file)
    #         os.rename(seg_file_old, seg_file)
    #         #binarized_wmh_file = threshold_binarize_nifti(probmap_file, seg_threshold, os.path.join(self.output_path, rename_bids_file(flair_filename, entities_WMHSynthSeg, 'WMHmask', '.nii.gz')))
    #         binarized_wmh_file = extract_roi_from_nii(seg_file, [77], binarize=True, output_nii_path=os.path.join(self.output_path, rename_bids_file(flair_filename, entities_WMHSynthSeg, 'WMHmask', '.nii.gz')))
    #     else:
    #         print("No segmentation method specified. Skipping WMH segmentation.")
        
    #     """
    #     2. WMH 部位分类
    #     """
    #     # 2.1 Preprocess T1w and FLAIR images
    #     if self.location_method is not None:
    #         if self.seg_method == 'WMHSynthSeg':
    #             # SynthSeg the flair image
    #             subprocess.run(['mri_synthseg',
    #                             '--i', input4WMHSynthSeg_dir,
    #                             '--o', self.output_path,
    #                             '--parc', '--robust',
    #                             '--vol', os.path.join(self.output_path, 'SynthSegVols.csv'),
    #                             '--qc', os.path.join(self.output_path, 'SynthSegQC.csv'),
    #                             ])
                
    #             entities_SynthSeg = {
    #                 'space': 'SynthSeg',
    #             }
    #             synthseg_file = os.path.join(self.output_path, rename_bids_file(flair_filename, entities_SynthSeg, 'SynthSeg', '.nii.gz'))
    #             os.rename(os.path.join(self.output_path, f'{flair_filename}_SynthSeg.nii.gz'), synthseg_file)

    #             vent_mask = extract_roi_from_nii(seg_file, [4, 43], binarize=True, output_nii_path=os.path.join(self.output_path, rename_bids_file(flair_filename, entities_WMHSynthSeg, 'VentMask', '.nii.gz')))

    #             # generate the dist_to_vent_periventricular mask
    #             entities_3mm = {
    #                 'space': 'WMHSynthSeg',
    #                 'desc': '3mm'
    #             }
    #             entities_10mm = {
    #                 'space': 'WMHSynthSeg',
    #                 'desc': '10mm'
    #             }
    #             dist_to_vent = os.path.join(self.output_path, rename_bids_file(flair_filename, entities_WMHSynthSeg, 'Dist2Vent', '.nii.gz'))
    #             bianca_mask_path = None
    #             mask_3mm_nii_path = os.path.join(self.output_path, rename_bids_file(flair_filename, entities_3mm, 'PeriventricularMask', '.nii.gz'))
    #             mask_10mm_nii_path = os.path.join(self.output_path, rename_bids_file(flair_filename, entities_10mm, 'PeriventricularMask', '.nii.gz'))
    #             subprocess.run(["distancemap", "-i", vent_mask, "-o", dist_to_vent])
    #             subprocess.run(["fslmaths", dist_to_vent, '-uthr', '3', '-bin', mask_3mm_nii_path])
    #             subprocess.run(["fslmaths", dist_to_vent, '-uthr', '10', '-bin', mask_10mm_nii_path])

    #             wmh_for_analysis = binarized_wmh_file

    #         else:
    #             # 2.1.1
    #             print("Registering FLAIR to T1w image...")
    #             bash_command = ["bash", self.script_path_register, t1w_file, '0', flair_file, '0', '0', self.output_path_xfm]

    #             try:
    #                 subprocess.run(bash_command, check=True)
    #                 print("Registration completed.")
    #                 os.rename(os.path.join(self.output_path_xfm, f'{flair_filename}_brain.nii.gz'), os.path.join(self.output_path_xfm, f'{flair_filename.split("_FLAIR")[0]}_desc-stripped_FLAIR.nii.gz'))
    #                 os.rename(os.path.join(self.output_path_xfm, f'{flair_filename}_mask.nii.gz'), os.path.join(self.output_path_xfm, f'{flair_filename.split("_FLAIR")[0]}_space-FLAIR_brainmask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path_xfm, f'{t1w_filename}_brain.nii.gz'), os.path.join(self.output_path_xfm, f'{t1w_filename.split("_T1w")[0]}_desc-stripped_T1w.nii.gz'))
    #                 os.rename(os.path.join(self.output_path_xfm, f'{t1w_filename}_mask.nii.gz'), os.path.join(self.output_path_xfm, f'{t1w_filename.split("_T1w")[0]}_space-T1w_brainmask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path_xfm, f'{flair_filename}_in_{t1w_filename}.nii.gz'), os.path.join(self.output_path_xfm, f'{flair_filename.split("_FLAIR")[0]}_desc-stripped_space-T1w_FLAIR.nii.gz'))
    #                 os.rename(os.path.join(self.output_path_xfm, f'{flair_filename}2{t1w_filename}.mat'), os.path.join(self.output_path_xfm, f'sub-{self.subject.subject_id}{session_entity}_from-FLAIR_to-T1w_xfm.mat'))
    #                 os.rename(os.path.join(self.output_path_xfm, f'{t1w_filename}2{flair_filename}.mat'), os.path.join(self.output_path_xfm, f'sub-{self.subject.subject_id}{session_entity}_from-T1w_to-FLAIR_xfm.mat'))

    #                 stripped_flair_in_t1w = os.path.join(self.output_path_xfm, f'{flair_filename.split("_FLAIR")[0]}_desc-stripped_space-T1w_FLAIR.nii.gz')
    #                 flair2t1w_xfm = os.path.join(self.output_path_xfm, f'sub-{self.subject.subject_id}{session_entity}_from-FLAIR_to-T1w_xfm.mat')

    #                 # transform WMH results to T1w space
    #                 subprocess.run(["flirt", "-in", binarized_wmh_file, "-ref", t1w_file, "-applyxfm", "-init", flair2t1w_xfm, "-out", os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-T1w_WMHmask.nii.gz'), "-interp", "nearestneighbour"], check=True)
    #                 subprocess.run(["flirt", "-in", probmap_file, "-ref", t1w_file, "-applyxfm", "-init", flair2t1w_xfm, "-out", os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}_space-T1w_WMHprobmap.nii.gz')], check=True)
    #                 binarized_wmh_file_in_t1w = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-T1w_WMHmask.nii.gz')
    #                 probmap_file_in_t1w = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}_space-T1w_WMHprobmap.nii.gz')
    #             except subprocess.CalledProcessError as e:
    #                 print(f"Error occurred while running registration script: {e}")
                
    #             # 2.1.2
    #             print("Running fsl_anat...")
    #             # make a COPY of the original T1w file in the fsl_anat directory, keep the original file
    #             shutil.copy(t1w_file, os.path.join(self.output_path_fslanat, f'{t1w_filename}.nii.gz'))
    #             bash_command = ["fsl_anat", "-i", os.path.join(self.output_path_fslanat, f'{t1w_filename}.nii.gz')]

    #             fslanat_output_dir = os.path.join(self.output_path_fslanat, f'{t1w_filename}.anat')

    #             try:
    #                 # 如果已经存在 fsl_anat 输出目录，则可以跳过
    #                 if not os.path.exists(fslanat_output_dir):
    #                     subprocess.run(bash_command, check=True)
    #                     print("fsl_anat completed.")
    #                     os.remove(os.path.join(self.output_path_fslanat, f'{t1w_filename}.nii.gz'))
    #                 else:
    #                     print("fsl_anat output directory already exists. Skipping fsl_anat.")
                    
    #             except subprocess.CalledProcessError as e:
    #                 print(f"Error occurred while running fsl_anat: {e}")
                
    #             # 2.1.3
    #             print("Normalization and BIANCA mask generation...")
    #             bash_command_fslpreproc2 = ["bash", self.script_path_fslpreproc, stripped_flair_in_t1w, binarized_wmh_file_in_t1w, probmap_file_in_t1w, fslanat_output_dir, self.output_path]

    #             try:
    #                 subprocess.run(bash_command_fslpreproc2, check=True)
    #                 print("Normalization and BIANCA mask generation completed.")

    #                 os.rename(os.path.join(self.output_path, f'dist_to_vent_periventricular_1mm_orig.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-1mm_space-T1w_PeriventricularMask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'dist_to_vent_periventricular_3mm_orig.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-3mm_space-T1w_PeriventricularMask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'dist_to_vent_periventricular_5mm_orig.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-5mm_space-T1w_PeriventricularMask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'dist_to_vent_periventricular_10mm_orig.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-10mm_space-T1w_PeriventricularMask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'T1_biascorr_bianca_mask_orig.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_space-T1w_BiancaMask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'T1_biascorr_ventmask_orig.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_space-T1w_VentMask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'T1_biascorr_ventmask_2_MNI.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_space-MNI_VentMask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'WMHprobmap_in_MNI.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}_space-MNI_WMHprobmap.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'WMHmask_in_MNI.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-MNI_WMHmask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, f'FLAIR_MNI.nii.gz'), os.path.join(self.output_path, f'{flair_filename.split("_FLAIR")[0]}_desc-stripped_space-MNI_FLAIR.nii.gz'))

    #                 bianca_mask_path = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_space-T1w_BiancaMask.nii.gz')
    #                 mask_3mm_nii_path = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-3mm_space-T1w_PeriventricularMask.nii.gz')
    #                 mask_10mm_nii_path = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-10mm_space-T1w_PeriventricularMask.nii.gz')
    #                 wmh_for_analysis = binarized_wmh_file_in_t1w
    #             except subprocess.CalledProcessError as e:
    #                 print(f"Error occurred while running fsl_preproc2: {e}")

    #     if 'Fazekas' in self.location_method:
    #         """
    #         WMH 部位分类预处理
    #         """
 
    #         # analyze WMH in T1w space
    #         """
    #         WMH 部位分类（PWMH and DWMH）
    #         """
    #         print("Classifying PWMH and DWMH...")
    #         wmh_for_analysis = wmh_for_analysis
    #         wmh_for_analysis_filename = f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-T1w_WMHmask'

    #         wmh_nii = nib.load(wmh_for_analysis)
    #         wmh_data = wmh_nii.get_fdata()
    #         wmh_data[np.isnan(wmh_data)] = 0

    #         if bianca_mask_path is not None:
    #             bianca_mask_nii = nib.load(bianca_mask_path)
    #             bianca_mask_data = bianca_mask_nii.get_fdata()

    #             # 首先排除bianca_mask文件中的非WM区域
    #             wmh_data = np.multiply(wmh_data, bianca_mask_data)

    #         mask_3mm_nii = nib.load(mask_3mm_nii_path)
    #         mask_3mm_data = mask_3mm_nii.get_fdata()
    #         mask_3mm_data[np.isnan(mask_3mm_data)] = 0

    #         mask_10mm_nii = nib.load(mask_10mm_nii_path)
    #         mask_10mm_data = mask_10mm_nii.get_fdata()
    #         mask_10mm_data[np.isnan(mask_10mm_data)] = 0

    #         # 识别WMH图像中的所有连续区域
    #         labeled_wmh, num_features = label(wmh_data)

    #         # 初始化结果数组
    #         result_confluent_WMH = np.zeros_like(wmh_data)
    #         result_periventricular_WMH = np.zeros_like(wmh_data)
    #         result_deep_WMH = np.zeros_like(wmh_data)
    #         result_periventricular_or_confluent_WMH = np.zeros_like(wmh_data)

    #         for region_num in range(1, num_features + 1):
    #             # 提取当前连续区域
    #             region = (labeled_wmh == region_num).astype(np.int32)

    #             # 检查区域是否部分落在3mm mask内
    #             in_3mm = np.any(np.logical_and(region, mask_3mm_data))

    #             # 检查区域是否部分扩展到10mm mask之外
    #             out_10mm = np.any(np.logical_and(region, np.logical_not(mask_10mm_data)))

    #             # confluent WMH：区域部分落在3mm mask内，且部分扩展到10mm mask之外
    #             if in_3mm and out_10mm:
    #                 result_confluent_WMH = np.logical_or(result_confluent_WMH, region)

    #             # periventricular WMH：区域部分落在3mm mask内，且没有扩展到10mm mask之外
    #             if in_3mm and not out_10mm:
    #                 result_periventricular_WMH = np.logical_or(result_periventricular_WMH, region)

    #             # deep WMH：区域没有落在3mm mask内
    #             if not in_3mm:
    #                 result_deep_WMH = np.logical_or(result_deep_WMH, region)

    #             result_periventricular_or_confluent_WMH = np.logical_or(result_confluent_WMH, result_periventricular_WMH)
            
    #         # 保存结果
    #         fazekas_wmh_entities = {
    #             'space': analysis_space
    #         }
    #         masked_wmh_path = os.path.join(self.output_path, rename_bids_file(flair_filename, fazekas_wmh_entities, 'TWMHmask', '.nii.gz'))
    #         result_masked_WMH_nii = nib.Nifti1Image(wmh_data, wmh_nii.affine, wmh_nii.header)
    #         nib.save(result_masked_WMH_nii, masked_wmh_path)

    #         result_deep_WMH_nii_path = os.path.join(self.output_path, rename_bids_file(flair_filename, fazekas_wmh_entities, 'DWMHmask', '.nii.gz'))
    #         result_deep_WMH_nii = nib.Nifti1Image(result_deep_WMH.astype(np.int32), wmh_nii.affine, wmh_nii.header)
    #         nib.save(result_deep_WMH_nii, result_deep_WMH_nii_path)

    #         result_periventricular_or_confluent_WMH_nii_path = os.path.join(self.output_path, rename_bids_file(flair_filename, fazekas_wmh_entities, 'PWMHmask', '.nii.gz'))
    #         result_periventricular_or_confluent_WMH_nii = nib.Nifti1Image(result_periventricular_or_confluent_WMH.astype(np.int32), wmh_nii.affine, wmh_nii.header)
    #         nib.save(result_periventricular_or_confluent_WMH_nii, result_periventricular_or_confluent_WMH_nii_path)

    #         """
    #         WMH 定量及转换到MNI空间
    #         """
    #         subprocess.run(['bash', self.script_quantification,
    #                         masked_wmh_path,
    #                         result_periventricular_or_confluent_WMH_nii_path,
    #                         result_deep_WMH_nii_path,
    #                         self.output_path])
            

    #         """
    #         WMH Shape Analysis
    #         """
    #         if self.shape_analysis:
    #             vent_mask_path = os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_space-MNI_VentMask.nii.gz')
    #             bash_command = ["bash", self.script_normalization, masked_wmh_path, result_periventricular_or_confluent_WMH_nii_path, result_deep_WMH_nii_path, vent_mask_path, fslanat_output_dir, self.output_path]

    #             try:
    #                 subprocess.run(bash_command, check=True)
    #                 print("WMH volume quantification completed.")

    #                 os.rename(os.path.join(self.output_path, f'WMH_PWMH&DWMH_volume_thr5voxels.txt'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}-thr5voxels_WMHVolume.txt'))
    #                 os.rename(os.path.join(self.output_path, 'TWMH_MNI.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-MNI_TWMHmask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, 'PWMH_MNI.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-MNI_PWMHmask.nii.gz'))
    #                 os.rename(os.path.join(self.output_path, 'DWMH_MNI.nii.gz'), os.path.join(self.output_path, f'sub-{self.subject.subject_id}{session_entity}_desc-{self.seg_method}-thr{thr_string}_space-MNI_DWMHmask.nii.gz'))
    #             except subprocess.CalledProcessError as e:
    #                 print(f"Error occurred while running WMH volume quantification script: {e}")

    #             min_voxels = self.shape_analysis_voxel_thr
    #             plot = False
    #             na_placeholder = None # because we initially use 'NA', but it will lead to warnings when running 'pd.concat' for extracting results

    #             # 储存单个被试的结果
    #             columns = ['Region', 'Surface Area', 'Volume', 'Convex Hull Area', 'Convex Hull Volume', 'Convexity', 'Solidity',
    #                 'Concavity Index', 'Inverse Sphericity Index', 'Eccentricity', 'Fractal Dimension']
    #             PWMH_shape_features_df = pd.DataFrame(columns=columns)
    #             DWMH_shape_features_df = pd.DataFrame(columns=columns)

    #             shape_features_dir = os.path.join(self.output_path, 'shape_features') # directory to save shape features
    #             os.makedirs(shape_features_dir, exist_ok=True)

    #             if plot:
    #                 shape_features_plot_dir = os.path.join(shape_features_dir, 'plot')
    #                 os.makedirs(shape_features_plot_dir, exist_ok=True)

    #             vent_mask_path = os.path.join(self.output_path, 'T1_biascorr_ventmask_2_MNI.nii.gz')
    #             vent_img = nib.load(vent_mask_path)
    #             vent_data = vent_img.get_fdata()

    #             DWMH_path = os.path.join(self.output_path, 'DWMH_MNI.nii.gz')
    #             DWMH_img = nib.load(DWMH_path)
    #             DWMH_data = DWMH_img.get_fdata()
    #             labeled_DWMH, num_features_DWMH = label(DWMH_data)
    #             volume = sum(DWMH_data, labeled_DWMH, range(num_features_DWMH + 1))
    #             remove = volume < min_voxels
    #             remove_indices = np.where(remove)[0]
    #             for idx in remove_indices:
    #                 DWMH_data[labeled_DWMH == idx] = 0
    #             labeled_DWMH, num_features_DWMH = label(DWMH_data)
    #             print('{0} DWMH regions detected'.format(num_features_DWMH))

    #             PWMH_path = os.path.join(self.output_path, 'PWMH_MNI.nii.gz')
    #             PWMH_img = nib.load(PWMH_path)
    #             PWMH_data = PWMH_img.get_fdata()
    #             labeled_PWMH, num_features_PWMH = label(PWMH_data)
    #             volume = sum(PWMH_data, labeled_PWMH, range(num_features_PWMH + 1))
    #             remove = volume < min_voxels
    #             remove_indices = np.where(remove)[0]
    #             for idx in remove_indices:
    #                 PWMH_data[labeled_PWMH == idx] = 0
    #             labeled_PWMH, num_features_PWMH = label(PWMH_data)
    #             print('{0} PWMH regions detected'.format(num_features_PWMH))

    #             # labeled WMH
    #             PWMH_labeled = np.zeros_like(PWMH_data, dtype=np.int32)
    #             DWMH_labeled = np.zeros_like(DWMH_data, dtype=np.int32)

    #             # Total lesion plot
    #             if not DWMH_data.any() or not PWMH_data.any() or not vent_data.any():
    #                 print(f"At least one of DWMH or PWMH data is empty. Skipping total lesion plot")
    #             else:
    #                 unique_part_PWMH = np.logical_and(PWMH_data, np.logical_not(vent_data)).astype(np.uint8)
    #                 unique_part_DWMH = np.logical_and(DWMH_data, np.logical_not(vent_data)).astype(np.uint8)
    #                 unique_part_vent = np.logical_and(vent_data, np.logical_not(PWMH_data), np.logical_not(DWMH_data)).astype(np.uint8)

    #                 vertices1, faces1, _, _ = marching_cubes(unique_part_PWMH, level=0)
    #                 vertices2, faces2, _, _ = marching_cubes(unique_part_DWMH, level=0)
    #                 vertices3, faces3, _, _ = marching_cubes(unique_part_vent, level=0)

    #                 angles_three_views = [(30, 30), (210, 60), (-90, 90)]

    #                 plot_three_views_in_row(vertices1, faces1, vertices2, faces2, vertices3, faces3, angles_three_views)
    #                 plt.savefig(os.path.join(shape_features_dir, 'Total_lesion_3D_plot.png'))
    #                 plt.close('all')
                
    #             # PWMH shape features calculation
    #             if PWMH_data.any():
    #                 for region_num in range(1, num_features_PWMH + 1):
    #                     region = (labeled_PWMH == region_num).astype(np.int32)
    #                     region[region != 0] = 1

    #                     # Calculate the shape features of the region/cluster
    #                     # mc_surface_area, mc_volume, convex_hull_area, convex_hull_volume, convexity, solidity, concavity_index, \
    #                     #     inverse_sphericity_index, eccentricity, fractal_dimension = calculate_shape_features(region)
    #                     #print(mc_surface_area, mc_volume, convex_hull_area, convex_hull_volume, convexity, solidity, concavity_index, inverse_shape_index, eccentricity, fractal_dimension)

    #                     PWMH_labeled[labeled_PWMH == region_num] = region_num

    #                     if plot:
    #                         # Save the shape features plots
    #                         name = 'PWMH_region_{}'.format(region_num)
    #                         shape_features_plot(region, name, shape_features_plot_dir)

    #                     # 将形态学特征添加到DataFrame
    #                     shape_features = calculate_shape_features(region)
    #                     PWMH_shape_features_df.loc[len(PWMH_shape_features_df)] = [f'PWMH_region_{region_num}'] + list(shape_features)

    #                 PWMH_labeled_name = 'PWMH_MNI_labeled.nii.gz'
    #                 PWMH_labeled_nii_path = os.path.join(shape_features_dir, PWMH_labeled_name)
    #                 PWMH_labeled_nii = nib.Nifti1Image(PWMH_labeled, PWMH_img.affine, PWMH_img.header)
    #                 nib.save(PWMH_labeled_nii, PWMH_labeled_nii_path)

    #                 # 1个被试的所有病灶
    #                 excel_path = os.path.join(shape_features_dir, f'PWMH_shape_features_{min_voxels}voxels.xlsx')
    #                 PWMH_shape_features_df.to_excel(excel_path, index=False)

    #                 subject_averages = PWMH_shape_features_df.iloc[:, 5:11].mean()  # 选择从 'Convexity' 到 'Fractal Dimension' 的列
    #                 subject_avg_data = {'Subject': f'sub-{self.subject.subject_id}_ses-{self.session.session_id}'}
    #                 subject_avg_data.update(subject_averages.to_dict())

    #             else:
    #                 print('no PWMH lesion detected')
    #                 excel_path = os.path.join(shape_features_dir, f'PWMH_shape_features_{min_voxels}voxels.xlsx')
    #                 DWMH_shape_features_df.loc[len(DWMH_shape_features_df)] = [na_placeholder] * len(columns)
    #                 DWMH_shape_features_df.to_excel(excel_path, index=False)

    #                 subject_avg_data = {
    #                     'Subject': f'sub-{self.subject.subject_id}_ses-{self.session.session_id}',
    #                     'Convexity': na_placeholder,
    #                     'Solidity': na_placeholder,
    #                     'Concavity Index': na_placeholder,
    #                     'Inverse Sphericity Index': na_placeholder,
    #                     'Eccentricity': na_placeholder,
    #                     'Fractal Dimension': na_placeholder
    #                 }

    #             # 1个被试的所有病灶的平均值
    #             subject_avg_df = pd.DataFrame([subject_avg_data])
    #             subject_avg_path = os.path.join(shape_features_dir, f'average_PWMH_shape_features_{min_voxels}voxels.xlsx')
    #             subject_avg_df.to_excel(subject_avg_path, index=False)

    #             # DWMH shape features calculation
    #             if DWMH_data.any():
    #                 for region_num in range(1, num_features_DWMH + 1):
    #                     region = (labeled_DWMH == region_num).astype(np.int32)
    #                     region[region != 0] = 1

    #                     # Calculate the shape features of the region/cluster
    #                     # mc_surface_area, mc_volume, convex_hull_area, convex_hull_volume, convexity, solidity, concavity_index, \
    #                     #     inverse_sphericity_index, eccentricity, fractal_dimension = calculate_shape_features(region)
    #                     #print(mc_surface_area, mc_volume, convex_hull_area, convex_hull_volume, convexity, solidity, concavity_index, inverse_shape_index, eccentricity, fractal_dimension)

    #                     DWMH_labeled[labeled_DWMH == region_num] = region_num

    #                     if plot:
    #                         # Save the shape features plots
    #                         name = 'DWMH_region_{}'.format(region_num)
    #                         shape_features_plot(region, name)

    #                     # 将形态学特征添加到DataFrame
    #                     shape_features = calculate_shape_features(region)
    #                     DWMH_shape_features_df.loc[len(DWMH_shape_features_df)] = [f'DWMH_region_{region_num}'] + list(shape_features)

    #                 DWMH_labeled_name = 'DWMH_MNI_labeled.nii.gz'
    #                 DWMH_labeled_nii_path = os.path.join(shape_features_dir, DWMH_labeled_name)
    #                 DWMH_labeled_nii = nib.Nifti1Image(DWMH_labeled, DWMH_img.affine, DWMH_img.header)
    #                 nib.save(DWMH_labeled_nii, DWMH_labeled_nii_path)

    #                 excel_path = os.path.join(shape_features_dir, f'DWMH_shape_features_{min_voxels}voxels.xlsx')
    #                 DWMH_shape_features_df.to_excel(excel_path, index=False)

    #                 subject_averages = DWMH_shape_features_df.iloc[:, 5:11].mean()  # 选择从 'Convexity' 到 'Fractal Dimension' 的列
    #                 subject_avg_data = {'Subject': f'sub-{self.subject.subject_id}_ses-{self.session.session_id}'}
    #                 subject_avg_data.update(subject_averages.to_dict())

    #             else:
    #                 print('no DWMH lesion detected')
    #                 excel_path = os.path.join(shape_features_dir, f'DWMH_shape_features_{min_voxels}voxels.xlsx')
    #                 DWMH_shape_features_df.loc[len(DWMH_shape_features_df)] = [na_placeholder] * len(columns)
    #                 DWMH_shape_features_df.to_excel(excel_path, index=False)

    #                 subject_avg_data = {
    #                     'Subject': f'sub-{self.subject.subject_id}_ses-{self.session.session_id}',
    #                     'Convexity': na_placeholder,
    #                     'Solidity': na_placeholder,
    #                     'Concavity Index': na_placeholder,
    #                     'Inverse Sphericity Index': na_placeholder,
    #                     'Eccentricity': na_placeholder,
    #                     'Fractal Dimension': na_placeholder
    #                 }

    #             subject_avg_df = pd.DataFrame([subject_avg_data])
    #             subject_avg_path = os.path.join(shape_features_dir, f'average_DWMH_shape_features_{min_voxels}voxels.xlsx')
    #             subject_avg_df.to_excel(subject_avg_path, index=False)
                
    #     if 'bullseyes' in self.location_method:
    #         # check if already have freesurfer output
    #         assume_fs_output_dir = os.path.join(self.freesurfer_output_root_dir, f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
    #         if not os.path.exists(assume_fs_output_dir):
    #             if self.skip_if_no_freesurfer:
    #                 print(f"No FreeSurfer output found for sub-{self.subject.subject_id} ses-{self.session.session_id}. Skipping Bullseyes analysis.")
    #                 return
    #             else:
    #                 # TODO：不跳过则运行FreeSurfer
    #                 raise FileNotFoundError(f"No FreeSurfer output found for sub-{self.subject.subject_id} ses-{self.session.session_id}.")
    #         else:
    #             print(f"Found FreeSurfer output for sub-{self.subject.subject_id} ses-{self.session.session_id}.")

    #             bullseys_output_dir = os.path.join(self.output_path, 'bullseye_output')
    #             os.makedirs(bullseys_output_dir, exist_ok=True)

    #             if self.session.session_id is None:
    #                 command = [
    #                     "python", self.script_path4,
    #                     "-s", os.path.join(self.freesurfer_output_root_dir),
    #                     "--subjects", f'sub-{self.subject.subject_id}',
    #                     "-w", bullseys_output_dir,
    #                     "-o", bullseys_output_dir,
    #                     "-p", "8"
    #                 ]
    #             else:
    #                 command = [
    #                     "python", self.script_path4,
    #                     "-s", os.path.join(self.freesurfer_output_root_dir, f"sub-{self.subject.subject_id}"),
    #                     "--subjects", f'ses-{self.session.session_id}',
    #                     "-w", bullseys_output_dir,
    #                     "-o", bullseys_output_dir,
    #                     "-p", "8"
    #                 ]

    #                 # TODO：确保T1_2_FLAIR.mat存在
    #                 bash_command = ["bash", 
    #                                 self.script_path5, os.path.join(bullseys_output_dir, f'ses-{self.session.session_id}'), 
    #                                 os.path.join(self.freesurfer_output_root_dir, f"sub-{self.subject.subject_id}"),
    #                                 f'ses-{self.session.session_id}',
    #                                 os.path.join(self.output_path, 'T1_2_FLAIR.mat'),
    #                                 os.path.join(self.output_path, 'FLAIR_brain.nii.gz'),
    #                                 bullseys_output_dir]

    #             result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #             if result.returncode == 0:
    #                 print("Bullseyes preprocessing successfully!")
    #                 print(result.stdout)
    #             else:
    #                 print(f"Error: {result.stderr}")

    #             try:
    #                 subprocess.run(bash_command, check=True)

    #                 WMH_img = nib.load(wmh_for_analysis)
    #                 WMH_data = WMH_img.get_fdata()
    #                 WMH_affine = WMH_img.affine
    #                 # calculate single voxel size
    #                 WMH_voxel_size = np.abs(np.linalg.det(WMH_affine[:3, :3]))

    #                 bullseyes_in_flair = os.path.join(bullseys_output_dir, 'bullseye_wmparc_2_flair.nii.gz')
    #                 bullseyes_in_flair_img = nib.load(bullseyes_in_flair)
    #                 bullseyes_in_flair_data = bullseyes_in_flair_img.get_fdata()
    #                 unique_labels = np.unique(bullseyes_in_flair_data)

    #                 results = []

    #                 for region_label in unique_labels:
    #                     if region_label == 0:
    #                         continue
    #                     region = (bullseyes_in_flair_data == region_label).astype(np.int32)
    #                     region[region != 0] = 1

    #                     # Calculate the number of WMH voxels in the region
    #                     intersection = np.logical_and(WMH_data, region)
    #                     num_voxels = np.sum(intersection)

    #                     # Calculate the volume of the region
    #                     volume = num_voxels * WMH_voxel_size

    #                     results.append([region_label, num_voxels, volume])
                    
    #                 results_df = pd.DataFrame(results, columns=['Bullseyes_region', 'Voxels', 'Volume'])
    #                 results_df_transposed = results_df.set_index('Bullseyes_region').transpose()
    #                 results_df_transposed.to_csv(os.path.join(bullseys_output_dir, 'WMH_bullseyes_quantification.csv'))

    #                 print("Bullseyes quantification completed.")
    #             except subprocess.CalledProcessError as e:
    #                 print(f"Error occurred while running Bullseyes quantification script: {e}")
    
    """
    提取结果
    """
    def _process_wmh_data(self, subject_id, session_id, base_path):
        """
        提取 WMH 和形态特征数据，并返回一个 DataFrame
        """
        # 文件路径
        txt_path = os.path.join(base_path, 'WMH_PWMH&DWMH_volume_thr5voxels.txt')
        PWMH_shape_features_path = os.path.join(base_path, 'shape_features', 'average_PWMH_shape_features_10voxels.xlsx')
        DWMH_shape_features_path = os.path.join(base_path, 'shape_features', 'average_DWMH_shape_features_10voxels.xlsx')

        # 初始化变量为 None
        total_wmh_volume = None
        pwmh_volume = None
        dwmh_volume = None

        pwmh_convexity = None
        pwmh_solidity = None
        pwmh_concavity_index = None
        pwmh_inverse_sphericity_index = None
        pwmh_eccentricity = None
        pwmh_fractal_dimension = None

        dwmh_convexity = None
        dwmh_solidity = None
        dwmh_concavity_index = None
        dwmh_inverse_sphericity_index = None
        dwmh_eccentricity = None
        dwmh_fractal_dimension = None

        # 提取 WMH 体积数据
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                content = f.read()

            total_wmh_match = re.search(r'TWMH[\s\S]+?total volume for .+? is (\d+\.\d+)', content)
            pwmh_match = re.search(r'PWMH[\s\S]+?total volume for .+? is (\d+\.\d+)', content)
            dwmh_match = re.search(r'DWMH[\s\S]+?total volume for .+? is (\d+\.\d+)', content)

            if total_wmh_match:
                total_wmh_volume = total_wmh_match.group(1)
            if pwmh_match:
                pwmh_volume = pwmh_match.group(1)
            if dwmh_match:
                dwmh_volume = dwmh_match.group(1)

        # 提取 PWMH 形态特征
        if os.path.exists(PWMH_shape_features_path):
            pwmh_shape_features_df = pd.read_excel(PWMH_shape_features_path)
            pwmh_convexity = pwmh_shape_features_df.get('Convexity', [None])[0]
            pwmh_solidity = pwmh_shape_features_df.get('Solidity', [None])[0]
            pwmh_concavity_index = pwmh_shape_features_df.get('Concavity Index', [None])[0]
            pwmh_inverse_sphericity_index = pwmh_shape_features_df.get('Inverse Sphericity Index', [None])[0]
            pwmh_eccentricity = pwmh_shape_features_df.get('Eccentricity', [None])[0]
            pwmh_fractal_dimension = pwmh_shape_features_df.get('Fractal Dimension', [None])[0]

        # 提取 DWMH 形态特征
        if os.path.exists(DWMH_shape_features_path):
            dwmh_shape_features_df = pd.read_excel(DWMH_shape_features_path)
            dwmh_convexity = dwmh_shape_features_df.get('Convexity', [None])[0]
            dwmh_solidity = dwmh_shape_features_df.get('Solidity', [None])[0]
            dwmh_concavity_index = dwmh_shape_features_df.get('Concavity Index', [None])[0]
            dwmh_inverse_sphericity_index = dwmh_shape_features_df.get('Inverse Sphericity Index', [None])[0]
            dwmh_eccentricity = dwmh_shape_features_df.get('Eccentricity', [None])[0]
            dwmh_fractal_dimension = dwmh_shape_features_df.get('Fractal Dimension', [None])[0]

        # 返回结果 DataFrame
        return pd.DataFrame([{
            'Subject': subject_id,
            'Session': session_id,
            'Total_WMH_volume': total_wmh_volume,
            'PWMH_volume': pwmh_volume,
            'DWMH_volume': dwmh_volume,
            'PWMH_Convexity': pwmh_convexity,
            'PWMH_Solidity': pwmh_solidity,
            'PWMH_Concavity_Index': pwmh_concavity_index,
            'PWMH_Inverse_Sphericity_Index': pwmh_inverse_sphericity_index,
            'PWMH_Eccentricity': pwmh_eccentricity,
            'PWMH_Fractal_Dimension': pwmh_fractal_dimension,
            'DWMH_Convexity': dwmh_convexity,
            'DWMH_Solidity': dwmh_solidity,
            'DWMH_Concavity_Index': dwmh_concavity_index,
            'DWMH_Inverse_Sphericity_Index': dwmh_inverse_sphericity_index,
            'DWMH_Eccentricity': dwmh_eccentricity,
            'DWMH_Fractal_Dimension': dwmh_fractal_dimension
        }])

    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        wmh_output_path = self.extract_from

        # 创建一个空的DataFrame来保存结果
        columns = ['Subject', 'Session', 'Total_WMH_volume', 'PWMH_volume', 'DWMH_volume', 
                'PWMH_Convexity', 'PWMH_Solidity', 'PWMH_Concavity_Index', 'PWMH_Inverse_Sphericity_Index', 
                'PWMH_Eccentricity', 'PWMH_Fractal_Dimension', 'DWMH_Convexity', 'DWMH_Solidity', 
                'DWMH_Concavity_Index', 'DWMH_Inverse_Sphericity_Index', 'DWMH_Eccentricity', 
                'DWMH_Fractal_Dimension']
        results_df = pd.DataFrame(columns=columns)

        # 遍历所有 sub-* 文件夹
        for subject_folder in os.listdir(wmh_output_path):
            subject_id = subject_folder.split('-')[1]
            subject_folder_path = os.path.join(wmh_output_path, subject_folder)

            if os.path.isdir(subject_folder_path):
                # 检查是否有 ses-* 文件夹
                session_folders = [f for f in os.listdir(subject_folder_path) if 'ses-' in f]

                if session_folders:  # 如果有 ses-* 文件夹
                    for session_folder in session_folders:
                        session_path = os.path.join(subject_folder_path, session_folder)
                        new_data = self._process_wmh_data(subject_id, session_folder.split('-')[1], session_path)
                        results_df = pd.concat([results_df, new_data], ignore_index=True)
                else:  # 如果没有 ses-* 文件夹
                    new_data = self._process_wmh_data(subject_id, 'N/A', subject_folder_path)
                    results_df = pd.concat([results_df, new_data], ignore_index=True)

        # 保存结果到 Excel 文件
        output_excel_path = os.path.join(self.output_path, 'wmh_quantification_results.xlsx')
        results_df.to_excel(output_excel_path, header=True, index=False)
        print(f"Quantification results saved to {output_excel_path}")
