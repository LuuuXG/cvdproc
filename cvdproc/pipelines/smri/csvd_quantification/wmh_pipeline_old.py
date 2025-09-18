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

from .wmh.wmh_seg_nipype import LSTSegmentation, LSTAI, WMHSynthSegSingle, PrepareTrueNetData, TrueNetEvaluate, TrueNetPostProcess
from .wmh.wmh_location_nipype import Fazekas, Bullseyes
from .wmh.wmh_shape_nipype import WMHShape

from nipype.interfaces import fsl
from ...smri.fsl.fsl_anat_nipype import FSLANAT
from ...smri.freesurfer.synthSR import SynthSR
from ...common.register import ModalityRegistration

class WMHSegmentationPipeline:
    """
    WMH Segmentation and Quantification Pipeline
    """
    def __init__(self, 
                 subject, 
                 session, 
                 output_path, 
                 matlab_path=None, 
                 use_which_t1w: str = None,
                 use_which_flair: str = None,
                 **kwargs):
        """
        WMH Segmentation and Quantification Pipeline
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

        self.use_which_t1w = use_which_t1w
        self.use_which_flair = use_which_flair
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
        self.skip_if_no_freesurfer = kwargs.get('skip_if_no_freesurfer', True)
        self.extract_from = kwargs.get('extract_from', None)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.script_path_lst = os.path.join(base_dir, 'matlab', 'wmh_seg_lst.m')

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
            print(f"No specific FLAIR file selected. Using the first one.")
        
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
        wmh_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')

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

            wmh_synthseg_node = Node(WMHSynthSegSingle(), name="wmh_synthseg")
            wmh_synthseg_node.inputs.wmh_filepath = os.path.join(self.output_path, binarized_wmh_filename)
            wmh_synthseg_node.inputs.prob_filepath = os.path.join(self.output_path, probmap_filename)
            wmh_synthseg_node.inputs.output = os.path.join(self.output_path, wmh_synthseg_filename)
            wmh_synthseg_node.inputs.device = 'cpu'
            wmh_synthseg_node.inputs.threads = 8

            wmh_workflow.connect([
                (inputnode, wmh_synthseg_node, [("flair_file", "input")]),
                (wmh_synthseg_node, wmh_mask_node, [("wmh_filepath", "wmh_mask"),
                                                     ("prob_filepath", "wmh_prob_map")]),
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
            print('Only FLAIR to T1w registeration will perform if the two exist')

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

            flair_to_t1w_xfm = Node(IdentityInterface(fields=["flair_to_t1w_xfm"]), name="flair_to_t1w_xfm_mat")

            if self.seg_method == 'WMHSynthSeg':
                flair_synthsr_node = Node(SynthSR(), name="flair_synthsr")
                wmh_workflow.connect([
                    (inputnode, flair_synthsr_node, [("flair_file", "input")]),
                ])
                flair_synthsr_node.inputs.output = os.path.join(self.output_path, rename_bids_file(placeholder, {'desc': 'SynthSRraw', 'space': 'FLAIR'}, 'T1w', '.nii.gz'))

                reslice_synthsr_node = Node(fsl.FLIRT(), name="reslice_synthsr")
                wmh_workflow.connect([
                    (flair_synthsr_node, reslice_synthsr_node, [("output", "in_file")]),
                    (wmh_mask_node, reslice_synthsr_node, [("wmh_mask", "reference")]),
                ])
                reslice_synthsr_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(placeholder, {'desc': 'SynthSRresliced', 'space': 'FLAIR'}, 'T1w', '.nii.gz'))
                reslice_synthsr_node.inputs.args = '-applyxfm -usesqform'

                synthsr_flair_to_t1w_xfm_node = Node(ModalityRegistration(), name="synthsr_flair_to_t1w_xfm")
                wmh_workflow.connect([
                    (reslice_synthsr_node, synthsr_flair_to_t1w_xfm_node, [("out_file", "image_source")]),
                    (inputnode, synthsr_flair_to_t1w_xfm_node, [("t1w_file", "image_target")])
                ])
                synthsr_flair_to_t1w_xfm_node.inputs.image_target_strip = 0
                synthsr_flair_to_t1w_xfm_node.inputs.image_source_strip = 0
                synthsr_flair_to_t1w_xfm_node.inputs.flirt_direction = 1
                synthsr_flair_to_t1w_xfm_node.inputs.output_dir = xfm_output_path
                synthsr_flair_to_t1w_xfm_node.inputs.registered_image_filename = rename_bids_file(placeholder, {'desc': 'SynthSRflair', 'space': 'T1w'}, 'T1w', '.nii.gz')
                synthsr_flair_to_t1w_xfm_node.inputs.source_to_target_mat_filename = rename_bids_file(placeholder, {'from': 'SynthSRflair', 'to': 'T1w'}, 'xfm', '.mat')
                synthsr_flair_to_t1w_xfm_node.inputs.target_to_source_mat_filename = rename_bids_file(placeholder, {'from': 'T1w', 'to': 'SynthSRflair'}, 'xfm', '.mat')
                synthsr_flair_to_t1w_xfm_node.inputs.dof = 6

                wmh_workflow.connect([
                    (synthsr_flair_to_t1w_xfm_node, flair_to_t1w_xfm, [("source_to_target_mat", "flair_to_t1w_xfm")]),
                ])
            else:
                wmh_workflow.connect([
                    (flair_to_t1w_xfm_node, flair_to_t1w_xfm, [("source_to_target_mat", "flair_to_t1w_xfm")]),
                ])

            os.makedirs(self.output_path_fslanat, exist_ok=True)

            fsl_anat_node = Node(FSLANAT(), name="fsl_anat")
            wmh_workflow.connect(inputnode, "t1w_file", fsl_anat_node, "input_image")
            fsl_anat_node.inputs.output_directory = os.path.join(self.subject.bids_dir, 'derivatives', 'fsl_anat', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'fsl')

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
                    fazekas_classification_node.inputs.vent_mask_filename = rename_bids_file(placeholder, wmh_synthseg_flair_entities, 'VentMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_3mm_filename = rename_bids_file(placeholder, wmh_synthseg_flair_entities, '3mmPeriventricularMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_10mm_filename = rename_bids_file(placeholder, wmh_synthseg_flair_entities, '10mmPeriventricularMask', '.nii.gz')
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

                    fazekas_classification_node.inputs.vent_mask_filename = rename_bids_file(placeholder, synthseg_flair_entities, 'VentMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_3mm_filename = rename_bids_file(placeholder, synthseg_flair_entities, '3mmPeriventricularMask', '.nii.gz')
                    fazekas_classification_node.inputs.perivent_mask_10mm_filename = rename_bids_file(placeholder, synthseg_flair_entities, '10mmPeriventricularMask', '.nii.gz')
                
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

            transform_probmap_to_t1w_node = Node(fsl.FLIRT(), name="transform_probmap_to_t1w")
            wmh_workflow.connect([
                (wmh_mask_node, transform_probmap_to_t1w_node, [("wmh_prob_map", "in_file")]),
                (flair_to_t1w_xfm, transform_probmap_to_t1w_node, [("flair_to_t1w_xfm", "in_matrix_file")]),
                (inputnode, transform_probmap_to_t1w_node, [("t1w_file", "reference")]),
            ])

            transform_probmap_to_t1w_node.inputs.interp = 'nearestneighbour'
            transform_probmap_to_t1w_node.inputs.apply_xfm = True
            transform_probmap_to_t1w_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_in_t1w_entities, 'WMHprobmap', '.nii.gz'))

            transform_probmap_to_mni_node = Node(fsl.ApplyWarp(), name="transform_probmap_to_mni")
            wmh_workflow.connect([
                (transform_probmap_to_t1w_node, transform_probmap_to_mni_node, [("out_file", "in_file")]),
                (fsl_anat_node, transform_probmap_to_mni_node, [("t1w_to_mni_nonlin_field", "field_file")]),
            ])
            transform_probmap_to_mni_node.inputs.interp = 'nn'
            transform_probmap_to_mni_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(os.path.join(self.output_path, binarized_wmh_filename), wmh_in_mni_entities, 'WMHprobmap', '.nii.gz'))
            transform_probmap_to_mni_node.inputs.ref_file = mni_template

            transform_pwmh_to_t1w_node = Node(fsl.FLIRT(), name="transform_pwmh_to_t1w")
            wmh_workflow.connect([
                (fazekas_classification_node, transform_pwmh_to_t1w_node, [("pwmh_mask", "in_file")]),
                (flair_to_t1w_xfm, transform_pwmh_to_t1w_node, [("flair_to_t1w_xfm", "in_matrix_file")]),
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
                (flair_to_t1w_xfm, transform_dwmh_to_t1w_node, [("flair_to_t1w_xfm", "in_matrix_file")]),
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

    """
    Extract results from WMH segmentation and shape analysis
    """
    def _process_wmh_data(self, subject_id, session_id, base_path):
        """
        Extract WMH and shape features data, and return a DataFrame
        """
        # File names
        synthseg_volume_path = os.path.join(base_path, "SynthSegVols.csv")
        total_volume_path = os.path.join(base_path, f"sub-{subject_id}_ses-{session_id}_TotalWMHVolume.csv")
        pwmh_volume_path = os.path.join(base_path, f"sub-{subject_id}_ses-{session_id}_PWMHVolume.csv")
        dwmh_volume_path = os.path.join(base_path, f"sub-{subject_id}_ses-{session_id}_DWMHVolume.csv")
        PWMH_shape_features_path = os.path.join(base_path, 'shape_features', 'average_PWMH_shape_features_10voxels.xlsx')
        DWMH_shape_features_path = os.path.join(base_path, 'shape_features', 'average_DWMH_shape_features_10voxels.xlsx')

        total_wmh_volume = None
        pwmh_volume = None
        dwmh_volume = None
        icv = None
        total_wmh_percenticv = None
        pwmh_percenticv = None
        dwmh_percenticv = None

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

        # Extract volume from CSV files
        def extract_volume_from_csv(path, lookup_row=2, lookup_col=3):
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, header=None)
                    return df.iloc[lookup_row, lookup_col]  # Row 3, Column 4
                except Exception:
                    return None
            else:
                return None

        # Convert to cm^3
        total_wmh_volume = float(extract_volume_from_csv(total_volume_path, lookup_row=2, lookup_col=3)) / 1000 
        pwmh_volume = float(extract_volume_from_csv(pwmh_volume_path, lookup_row=2, lookup_col=3)) / 1000
        dwmh_volume = float(extract_volume_from_csv(dwmh_volume_path, lookup_row=2, lookup_col=3)) / 1000
        icv = float(extract_volume_from_csv(synthseg_volume_path, lookup_row=1, lookup_col=1)) / 1000

        if icv is not None and total_wmh_volume is not None:
            total_wmh_percenticv = (total_wmh_volume / icv) * 100
        if icv is not None and pwmh_volume is not None:
            pwmh_percenticv = (pwmh_volume / icv) * 100
        if icv is not None and dwmh_volume is not None:
            dwmh_percenticv = (dwmh_volume / icv) * 100

        # Extract shape features from Excel files
        if os.path.exists(PWMH_shape_features_path):
            pwmh_shape_features_df = pd.read_excel(PWMH_shape_features_path)
            pwmh_convexity = pwmh_shape_features_df.get('Convexity', [None])[0]
            pwmh_solidity = pwmh_shape_features_df.get('Solidity', [None])[0]
            pwmh_concavity_index = pwmh_shape_features_df.get('Concavity Index', [None])[0]
            pwmh_inverse_sphericity_index = pwmh_shape_features_df.get('Inverse Sphericity Index', [None])[0]
            pwmh_eccentricity = pwmh_shape_features_df.get('Eccentricity', [None])[0]
            pwmh_fractal_dimension = pwmh_shape_features_df.get('Fractal Dimension', [None])[0]
        
        if os.path.exists(DWMH_shape_features_path):
            dwmh_shape_features_df = pd.read_excel(DWMH_shape_features_path)
            dwmh_convexity = dwmh_shape_features_df.get('Convexity', [None])[0]
            dwmh_solidity = dwmh_shape_features_df.get('Solidity', [None])[0]
            dwmh_concavity_index = dwmh_shape_features_df.get('Concavity Index', [None])[0]
            dwmh_inverse_sphericity_index = dwmh_shape_features_df.get('Inverse Sphericity Index', [None])[0]
            dwmh_eccentricity = dwmh_shape_features_df.get('Eccentricity', [None])[0]
            dwmh_fractal_dimension = dwmh_shape_features_df.get('Fractal Dimension', [None])[0]

        # Return results DataFrame
        return pd.DataFrame([{
            'Subject': subject_id,
            'Session': session_id,
            'Total_WMH_volume(ml)': total_wmh_volume,
            'PWMH_volume(ml)': pwmh_volume,
            'DWMH_volume(ml)': dwmh_volume,
            'ICV(ml)': icv,
            'Total_WMH_percentICV(%)': total_wmh_percenticv,
            'PWMH_percentICV(%)': pwmh_percenticv,
            'DWMH_percentICV(%)': dwmh_percenticv,
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

        # Create an empty DataFrame to store results
        columns = ['Subject', 'Session', 'Total_WMH_volume(ml)', 'PWMH_volume(ml)', 'DWMH_volume(ml)', 'ICV(ml)',
                   'Total_WMH_percentICV(%)', 'PWMH_percentICV(%)', 'DWMH_percentICV(%)',
                'PWMH_Convexity', 'PWMH_Solidity', 'PWMH_Concavity_Index', 'PWMH_Inverse_Sphericity_Index', 
                'PWMH_Eccentricity', 'PWMH_Fractal_Dimension', 'DWMH_Convexity', 'DWMH_Solidity', 
                'DWMH_Concavity_Index', 'DWMH_Inverse_Sphericity_Index', 'DWMH_Eccentricity', 
                'DWMH_Fractal_Dimension']
        results_df = pd.DataFrame(columns=columns)

        # Iterate through all sub-* folders
        for subject_folder in os.listdir(wmh_output_path):
            subject_id = subject_folder.split('-')[1]
            subject_folder_path = os.path.join(wmh_output_path, subject_folder)

            if os.path.isdir(subject_folder_path):
                # Check for ses-* folders
                session_folders = [f for f in os.listdir(subject_folder_path) if 'ses-' in f]

                if session_folders:  # If there are ses-* folders
                    for session_folder in session_folders:
                        session_path = os.path.join(subject_folder_path, session_folder)
                        new_data = self._process_wmh_data(subject_id, session_folder.split('-')[1], session_path)
                        results_df = pd.concat([results_df, new_data], ignore_index=True)
                else:  # If there are no ses-* folders
                    new_data = self._process_wmh_data(subject_id, 'N/A', subject_folder_path)
                    results_df = pd.concat([results_df, new_data], ignore_index=True)

        # Save results to Excel file
        output_excel_path = os.path.join(self.output_path, 'wmh_quantification_results.xlsx')
        results_df.to_excel(output_excel_path, header=True, index=False)
        print(f"Quantification results saved to {output_excel_path}")
