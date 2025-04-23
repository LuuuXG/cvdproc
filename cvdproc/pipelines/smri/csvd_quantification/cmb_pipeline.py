# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

# modified by Youjie Wang, 2025-01-09

import os
import subprocess
import nibabel as nib
import numpy as np
import gc
import time
from pathlib import Path
#import tensorflow as tf
from skimage import measure
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge
from ....bids_data.rename_bids_file import rename_bids_file
from ...common.register import create_register_workflow
from .shiva_segmentation.shiva_segmentation import SHIVAPredictImage
from .shiva_segmentation.shiva_nipype import PrepareShivaInput, ShivaSegmentation

class CMBSegmentationPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        session_part = f"ses-{self.session.session_id}" if self.session.session_id else ""
        self.output_path_xfm = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', session_part)
        
        self.use_which_swi = kwargs.get('use_which_swi', None)
        self.use_which_t1w = kwargs.get('use_which_t1w', None)
        self.method = kwargs.get('method', 'SHIVA')
        self.modality = kwargs.get('modality', 'swi')
        self.swi_stripped = kwargs.get('swi_stripped', False)
        self.shiva_config = kwargs.get('shiva_config', os.path.join(self.subject.bids_dir, 'code', 'shiva_config.yml'))
        # Path()
        self.predictor_files = [Path(f) for f in kwargs.get('predictor_files', [])]

        self.crop_or_pad_percentage = kwargs.get('crop_or_pad_percentage', (0.5, 0.5, 0.5))
        self.save_intermediate_image = kwargs.get('save_intermediate_image', False)
        self.threshold = kwargs.get('threshold', 0.5)

    def check_data_requirements(self):
        """
        检查数据需求
        :return: bool
        """
        if self.modality == 'swi':
            return self.session.get_swi_files() is not None
    
    def create_workflow(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_path_xfm, exist_ok=True)
        session_entity = f"_ses-{self.session.session_id}" if self.session.session_id else ""

        ## 1. Get the SWI and T1w files
        swi_files = self.session.get_swi_files()
        if self.use_which_swi:
            swi_files = [f for f in swi_files if self.use_which_swi in f]
            # 确保最终只有1个合适的文件
            if len(swi_files) != 1:
                raise FileNotFoundError(f"No specific SWI file found for {self.use_which_swi} or more than one found.")
            swi_file = swi_files[0]
        else:
            print("No specific SWI file selected. Using the first one.")
            swi_files = [swi_files[0]]
            swi_file = swi_files[0]
        
        t1w_files = self.session.get_t1w_files()
        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            # 确保最终只有1个合适的文件
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]
        
        cmb_quantification_pipeline = Workflow(name='cmb_quantification_pipeline')
        cmb_quantification_pipeline.base_dir = self.output_path

        if self.method == 'SHIVA':
            prefix = 'SHIVA_CMB'
            predict_node = Node(SHIVAPredictImage(), name="predict_image")

            shiva_input_dir = os.path.join(self.output_path, 'shiva_input')
            os.makedirs(shiva_input_dir, exist_ok=True)

            if self.modality == 'swi':
                print("Registering SWI to T1w image")

                entities_t1w_mask = {
                    'space': 'T1w',
                }

                entities_flair_mask = {
                    'space': 'FLAIR',
                }

                entities_stripped = {
                    'desc': 'stripped'
                }

                entities_swi2t1wxfm = {
                    'from': 'swi',
                    'to': 'T1w'
                }

                entities_t1w2swixfm = {
                    'from': 'T1w',
                    'to': 'swi'
                }

                entities_swiint1w = {
                    'space': 'T1w',
                    'desc': 'stripped'
                }

                t1w_mask_file = os.path.join(self.output_path_xfm,
                                             rename_bids_file(t1w_file, entities_t1w_mask, "brainmask", '.nii.gz'))
                t1w_stripped_file = os.path.join(self.output_path_xfm,
                                                 rename_bids_file(t1w_file, entities_stripped, "T1w", '.nii.gz'))
                swi_mask_file = os.path.join(self.output_path_xfm,
                                               rename_bids_file(swi_file, entities_flair_mask, "brainmask",
                                                                '.nii.gz'))
                swi_stripped_file = os.path.join(self.output_path_xfm,
                                                   rename_bids_file(swi_file, entities_stripped, "swi", '.nii.gz'))
                swi2t1w_xfm_file = os.path.join(self.output_path_xfm,
                                                  rename_bids_file(t1w_file, entities_swi2t1wxfm, "xfm", '.mat'))
                t1w2swi_xfm_file = os.path.join(self.output_path_xfm,
                                                  rename_bids_file(t1w_file, entities_t1w2swixfm, "xfm", '.mat'))
                swi_in_t1w_file = os.path.join(self.output_path_xfm,
                                                 rename_bids_file(swi_file, entities_swiint1w, "swi", '.nii.gz'))
                
                register_wf = create_register_workflow(t1w_file, t1w_stripped_file, t1w_mask_file,
                                                       swi_file, swi_stripped_file, swi_mask_file,
                                                       swi2t1w_xfm_file, t1w2swi_xfm_file, swi_in_t1w_file,
                                                       self.output_path_xfm,
                                                       False, self.swi_stripped)
                
                prepare_shiva_input_node = Node(PrepareShivaInput(), name="prepare_shiva_input")
                shiva_segmentation_node = Node(ShivaSegmentation(), name='shiva_segmentation')

                cmb_quantification_pipeline.connect([
                    (register_wf, prepare_shiva_input_node, [("outputnode.flirt_out_file", "swi_path")]),
                    (prepare_shiva_input_node, shiva_segmentation_node, [("shiva_input_dir", "shiva_input_dir")]),
                ])
                prepare_shiva_input_node.inputs.subject_id = f'sub-{self.subject.subject_id}{session_entity}'
                prepare_shiva_input_node.inputs.t1_path = t1w_file
                prepare_shiva_input_node.inputs.flair_path = ''
                prepare_shiva_input_node.inputs.output_dir = shiva_input_dir

                shiva_segmentation_node.inputs.shiva_output_dir = self.output_path
                shiva_segmentation_node.inputs.input_type = 'standard'
                shiva_segmentation_node.inputs.prediction = ['CMB']
                shiva_segmentation_node.inputs.shiva_config = self.shiva_config
                shiva_segmentation_node.inputs.brain_seg = 'synthseg'

        return cmb_quantification_pipeline

    # def run(self):
    #     os.makedirs(self.output_path, exist_ok=True)
    #     os.makedirs(self.output_path_xfm, exist_ok=True)
    #     session_entity = f"_ses-{self.session.session_id}" if self.session.session_id else ""

    #     ## 1. Get the SWI and T1w files
    #     swi_files = self.session.get_swi_files()
    #     if self.use_which_swi:
    #         swi_files = [f for f in swi_files if self.use_which_swi in f]
    #         # 确保最终只有1个合适的文件
    #         if len(swi_files) != 1:
    #             raise FileNotFoundError(f"No specific SWI file found for {self.use_which_swi} or more than one found.")
    #         swi_file = swi_files[0]
    #     else:
    #         print("No specific SWI file selected. Using the first one.")
    #         swi_files = [swi_files[0]]
    #         swi_file = swi_files[0]
        
    #     t1w_files = self.session.get_t1w_files()
    #     if self.use_which_t1w:
    #         t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
    #         # 确保最终只有1个合适的文件
    #         if len(t1w_files) != 1:
    #             raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
    #         t1w_file = t1w_files[0]
    #     else:
    #         print("No specific T1w file selected. Using the first one.")
    #         t1w_files = [t1w_files[0]]
    #         t1w_file = t1w_files[0]
        
    #     '''
    #     Main part of the pipeline
    #     '''
    #     if self.method == 'SHIVA':
    #         prefix = 'SHIVA_CMB'
    #         predict_node = Node(SHIVAPredictImage(), name="predict_image")
    #         shiva_cmb_pipeline = Workflow(name='shiva_cmb_pipeline')
    #         shiva_cmb_pipeline.base_dir = self.output_path

    #         if self.modality == 'swi':
    #             print("Registering SWI to T1w image")

    #             entities_t1w_mask = {
    #                 'space': 'T1w',
    #             }

    #             entities_flair_mask = {
    #                 'space': 'FLAIR',
    #             }

    #             entities_stripped = {
    #                 'desc': 'stripped'
    #             }

    #             entities_swi2t1wxfm = {
    #                 'from': 'swi',
    #                 'to': 'T1w'
    #             }

    #             entities_t1w2swixfm = {
    #                 'from': 'T1w',
    #                 'to': 'swi'
    #             }

    #             entities_swiint1w = {
    #                 'space': 'T1w',
    #                 'desc': 'stripped'
    #             }

    #             t1w_mask_file = os.path.join(self.output_path_xfm,
    #                                          rename_bids_file(t1w_file, entities_t1w_mask, "brainmask", '.nii.gz'))
    #             t1w_stripped_file = os.path.join(self.output_path_xfm,
    #                                              rename_bids_file(t1w_file, entities_stripped, "T1w", '.nii.gz'))
    #             swi_mask_file = os.path.join(self.output_path_xfm,
    #                                            rename_bids_file(swi_file, entities_flair_mask, "brainmask",
    #                                                             '.nii.gz'))
    #             swi_stripped_file = os.path.join(self.output_path_xfm,
    #                                                rename_bids_file(swi_file, entities_stripped, "swi", '.nii.gz'))
    #             swi2t1w_xfm_file = os.path.join(self.output_path_xfm,
    #                                               rename_bids_file(t1w_file, entities_swi2t1wxfm, "xfm", '.mat'))
    #             t1w2swi_xfm_file = os.path.join(self.output_path_xfm,
    #                                               rename_bids_file(t1w_file, entities_t1w2swixfm, "xfm", '.mat'))
    #             swi_in_t1w_file = os.path.join(self.output_path_xfm,
    #                                              rename_bids_file(swi_file, entities_swiint1w, "swi", '.nii.gz'))

    #             register_wf = create_register_workflow(t1w_file, t1w_stripped_file, t1w_mask_file,
    #                                                    swi_file, swi_stripped_file, swi_mask_file,
    #                                                    swi2t1w_xfm_file, t1w2swi_xfm_file, swi_in_t1w_file,
    #                                                    self.output_path_xfm,
    #                                                    False, self.swi_stripped)

    #             merge_node = Node(Merge(numinputs=1), name="merge_node")

    #             shiva_cmb_pipeline.connect([
    #                 (register_wf, merge_node, [("outputnode.flirt_out_file", "in1")])
    #             ])

    #             shiva_cmb_pipeline.connect([
    #                 (merge_node, predict_node, [('out', 'image_to_predict')]),
    #             ])

    #         thr_string = f'{self.threshold:.2f}'.replace('.', 'p')
    #         entities_CMBprobmap = {
    #             'space': 'T1w',
    #             'desc': 'SHIVA',
    #         }
    #         entities_CMBmask = {
    #             'space': 'T1w',
    #             'desc': f'SHIVA-thr{thr_string}',
    #         }
    #         entities_CMBlabel = {
    #             'space': 'T1w',
    #             'desc': f'SHIVA-thr{thr_string}',
    #         }
    #         suffix_CMBprobmap = "CMBprobmap"
    #         suffix_CMBmask = "CMBmask"
    #         suffix_CMBlabel = "CMBlabel"
    #         extension = ".nii.gz"
    #         CMBprobmap_file = os.path.join(self.output_path,
    #                                        rename_bids_file(t1w_file, entities_CMBprobmap, suffix_CMBprobmap,
    #                                                         extension))
    #         CMBmask_file = os.path.join(self.output_path,
    #                                     rename_bids_file(t1w_file, entities_CMBmask, suffix_CMBmask, extension))
    #         CMBlabel_file = os.path.join(self.output_path,
    #                                      rename_bids_file(t1w_file, entities_CMBlabel, suffix_CMBlabel, extension))

    #         predict_node.inputs.predictor_files = self.predictor_files
    #         predict_node.inputs.crop_or_pad_percentage = tuple(self.crop_or_pad_percentage)
    #         predict_node.inputs.threshold = self.threshold
    #         predict_node.inputs.prefix = prefix
    #         predict_node.inputs.output_path = self.output_path
    #         predict_node.inputs.prediction_file = CMBprobmap_file
    #         predict_node.inputs.binary_file = CMBmask_file
    #         predict_node.inputs.thresholded_file = CMBlabel_file
    #         predict_node.inputs.target_shape = (160, 214, 176)
    #         predict_node.inputs.save_intermediate_image = self.save_intermediate_image

    #         shiva_cmb_pipeline.run()

    #         entities_swicropped = {
    #             'space': 'SHIVA',
    #             'desc': 'cropped',
    #         }

    #         suffix_swicropped = "swi"

    #         swi_in_t1w_file_filename = os.path.basename(swi_in_t1w_file).split('.nii')[0]
    #         if self.save_intermediate_image:
    #             os.rename(os.path.join(self.output_path, f'crop_{swi_in_t1w_file_filename}.nii.gz'),
    #                       os.path.join(self.output_path, rename_bids_file(t1w_file, entities_swicropped, suffix_swicropped, extension)))