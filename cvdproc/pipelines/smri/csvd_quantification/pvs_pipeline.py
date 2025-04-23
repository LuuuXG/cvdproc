# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

# modified by Youjie Wang, 2025-01-20

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
#from .shiva_segmentation.shiva_segmentation import SHIVAPredictImage
from .shiva_segmentation.shiva_nipype import PrepareShivaInput, ShivaSegmentation

class PVSSegmentationPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.output_path_xfm = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f"ses-{self.session.session_id}")
        
        self.use_which_t1w = kwargs.get('use_which_t1w', None)
        self.use_which_flair = kwargs.get('use_which_flair', None)
        self.method = kwargs.get('method', 'SHIVA')
        self.modality = kwargs.get('modality', 'T1w')
        self.shiva_config = kwargs.get('shiva_config', os.path.join(self.subject.bids_dir, 'code', 'shiva_config.yml'))
        # Path()
        # self.predictor_files = kwargs.get('predictor_files', [])

        # self.crop_or_pad_percentage = kwargs.get('crop_or_pad_percentage', (0.5, 0.5, 0.5))
        # self.save_intermediate_image = kwargs.get('save_intermediate_image', False)
        # self.threshold = kwargs.get('threshold', 0.5)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.script_path_register = os.path.join(base_dir, 'bash', 'register.sh')

    def check_data_requirements(self):
        """
        检查数据需求
        :return: bool
        """
        if self.modality == 'T1w':
            return self.session.get_t1w_files() is not None
        elif self.modality == 'T1w+FLAIR':
            return self.session.get_flair_files() is not None and self.session.get_t1w_files() is not None
        else:
            return False
    
    def create_workflow(self):
        os.makedirs(self.output_path, exist_ok=True)
        session_entity = f"_ses-{self.session.session_id}" if self.session.session_id else ""

        ## 1. Get the T1w and FLAIR files
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
        
        t1w_filename = os.path.basename(t1w_file).split('.nii')[0]

        flair_files = self.session.get_flair_files()
        if self.modality == 'T1w+FLAIR':
            if self.use_which_flair:
                flair_files = [f for f in flair_files if self.use_which_flair in f]
                # 确保最终只有1个合适的文件
                if len(flair_files) != 1:
                    raise FileNotFoundError(f"No specific FLAIR file found for {self.use_which_flair} or more than one found.")
                flair_file = flair_files[0]
            else:
                print("No specific FLAIR file selected. Using the first one.")
                flair_files = [flair_files[0]]
                flair_file = flair_files[0]

            flair_filename = os.path.basename(flair_file).split('.nii')[0]

        pvs_quantification_workflow = Workflow(name='pvs_quantification_workflow')
        pvs_quantification_workflow.base_dir = self.output_path

        shiva_input_dir = os.path.join(self.output_path, 'shiva_input')
        os.makedirs(shiva_input_dir, exist_ok=True)
        # shiva_output_dir = os.path.join(self.output_path, 'shiva_output')
        # os.makedirs(shiva_output_dir, exist_ok=True)

        if self.method == 'SHIVA':
            inputnode = Node(IdentityInterface(fields=['subject_id', 'flair_path', 't1_path', 'swi_path', 'output_dir',
                                                    'shiva_input_dir', 'shiva_output_dir', 'input_type', 'prediction', 'shiva_config', 'brain_seg']), name='inputnode')
            inputnode.inputs.subject_id = f'sub-{self.subject.subject_id}{session_entity}'
            inputnode.inputs.t1_path = t1w_file
            inputnode.inputs.output_dir = shiva_input_dir
            if self.modality == 'T1w+FLAIR':
                inputnode.inputs.flair_path = flair_file
            else:
                inputnode.inputs.flair_path = ''
            inputnode.inputs.swi_path = ''
            inputnode.inputs.shiva_output_dir = self.output_path
            inputnode.inputs.input_type = 'standard'
            inputnode.inputs.prediction = ['PVS']
            inputnode.inputs.shiva_config = self.shiva_config
            inputnode.inputs.brain_seg = 'synthseg'
            
            prepare_shiva_input_node = Node(PrepareShivaInput(), name='prepare_shiva_input')
            pvs_quantification_workflow.connect([
                (inputnode, prepare_shiva_input_node, [('subject_id', 'subject_id'),
                                                    ('flair_path', 'flair_path'),
                                                    ('t1_path', 't1_path'),
                                                        ('swi_path', 'swi_path'),
                                                    ('output_dir', 'output_dir')]),
            ])

            shiva_segmentation_node = Node(ShivaSegmentation(), name='shiva_segmentation')
            pvs_quantification_workflow.connect([
                (prepare_shiva_input_node, shiva_segmentation_node, [('shiva_input_dir', 'shiva_input_dir')]),
                (inputnode, shiva_segmentation_node, [('shiva_output_dir', 'shiva_output_dir'),
                                                      ('input_type', 'input_type'),
                                                    ('prediction', 'prediction'),
                                                    ('shiva_config', 'shiva_config'),
                                                    ('brain_seg', 'brain_seg')]),
            ])

        return pvs_quantification_workflow

    # def run(self):
    #     os.makedirs(self.output_path, exist_ok=True)
    #     session_entity = f"_ses-{self.session.session_id}" if self.session.session_id else ""

    #     ## 1. Get the T1w and FLAIR files
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
        
    #     t1w_filename = os.path.basename(t1w_file).split('.nii')[0]

    #     flair_files = self.session.get_flair_files()
    #     if self.modality == 'T1w+FLAIR':
    #         if self.use_which_flair:
    #             flair_files = [f for f in flair_files if self.use_which_flair in f]
    #             # 确保最终只有1个合适的文件
    #             if len(flair_files) != 1:
    #                 raise FileNotFoundError(f"No specific FLAIR file found for {self.use_which_flair} or more than one found.")
    #             flair_file = flair_files[0]
    #         else:
    #             print("No specific FLAIR file selected. Using the first one.")
    #             flair_files = [flair_files[0]]
    #             flair_file = flair_files[0]

    #         flair_filename = os.path.basename(flair_file).split('.nii')[0]
        
    #     '''
    #     Main part of the pipeline
    #     '''
    #     if self.method == 'SHIVA':
    #         prefix = 'SHIVA_PVS'

    #         thr_string = f'{self.threshold:.2f}'.replace('.', 'p')
    #         entities_PVSprobmap = {
    #             'space': 'T1w',
    #             'desc': 'SHIVA',
    #         }
    #         entities_PVSmask = {
    #             'space': 'T1w',
    #             'desc': f'SHIVA-thr{thr_string}',
    #         }
    #         entities_PVSlabel = {
    #             'space': 'T1w',
    #             'desc': f'SHIVA-thr{thr_string}',
    #         }
    #         suffix_PVSprobmap = "PVSprobmap"
    #         suffix_PVSmask = "PVSmask"
    #         suffix_PVSlabel = "PVSlabel"
    #         extension = ".nii.gz"
    #         PVSprobmap_file = os.path.join(self.output_path, rename_bids_file(t1w_file, entities_PVSprobmap, suffix_PVSprobmap, extension))
    #         PVSmask_file = os.path.join(self.output_path, rename_bids_file(t1w_file, entities_PVSmask, suffix_PVSmask, extension))
    #         PVSlabel_file = os.path.join(self.output_path, rename_bids_file(t1w_file, entities_PVSlabel, suffix_PVSlabel, extension))

    #         predict_node = Node(SHIVAPredictImage(), name="predict_image")
    #         shiva_pvs_pipeline = Workflow(name='shiva_pvs_pipeline')
    #         shiva_pvs_pipeline.base_dir = self.output_path

    #         if self.modality == 'T1w':
    #             image_to_predict = [t1w_file]

    #             IdentityNode = Node(IdentityInterface(fields=['image_to_predict']), name='IdentityNode')
    #             IdentityNode.inputs.image_to_predict = image_to_predict
    #             shiva_pvs_pipeline.connect([
    #                 (IdentityNode, predict_node, [('image_to_predict', 'image_to_predict')]),
    #             ])
    #         elif self.modality == 'T1w+FLAIR':
    #             entities_t1w_mask = {
    #                 'space': 'T1w',
    #             }

    #             entities_flair_mask = {
    #                 'space': 'FLAIR',
    #             }

    #             entities_stripped = {
    #                 'desc': 'stripped'
    #             }

    #             entities_flair2t1wxfm = {
    #                 'from': 'FLAIR',
    #                 'to': 'T1w'
    #             }

    #             entities_t1w2flairxfm = {
    #                 'from': 'T1w',
    #                 'to': 'FLAIR'
    #             }

    #             entities_flairint1w = {
    #                 'space': 'T1w',
    #                 'desc': 'stripped'
    #             }

    #             t1w_mask_file = os.path.join(self.output_path_xfm,
    #                                          rename_bids_file(t1w_file, entities_t1w_mask, "brainmask", '.nii.gz'))
    #             t1w_stripped_file = os.path.join(self.output_path_xfm,
    #                                              rename_bids_file(t1w_file, entities_stripped, "T1w", '.nii.gz'))
    #             flair_mask_file = os.path.join(self.output_path_xfm,
    #                                            rename_bids_file(flair_file, entities_flair_mask, "brainmask",
    #                                                             '.nii.gz'))
    #             flair_stripped_file = os.path.join(self.output_path_xfm,
    #                                                rename_bids_file(flair_file, entities_stripped, "FLAIR", '.nii.gz'))
    #             flair2t1w_xfm_file = os.path.join(self.output_path_xfm,
    #                                               rename_bids_file(t1w_file, entities_flair2t1wxfm, "xfm", '.mat'))
    #             t1w2flair_xfm_file = os.path.join(self.output_path_xfm,
    #                                               rename_bids_file(t1w_file, entities_t1w2flairxfm, "xfm", '.mat'))
    #             flair_in_t1w_file = os.path.join(self.output_path_xfm,
    #                                              rename_bids_file(flair_file, entities_flairint1w, "FLAIR", '.nii.gz'))

    #             register_wf = create_register_workflow(t1w_file, t1w_stripped_file, t1w_mask_file,
    #                                                    flair_file, flair_stripped_file, flair_mask_file,
    #                                                    flair2t1w_xfm_file, t1w2flair_xfm_file, flair_in_t1w_file,
    #                                                    self.output_path_xfm,
    #                                                    False, False)

    #             merge_node = Node(Merge(numinputs=2), name="merge_node")

    #             shiva_pvs_pipeline.connect([
    #                 (register_wf, merge_node, [("outputnode.highres_out_file", "in1")]),
    #                 (register_wf, merge_node, [("outputnode.flirt_out_file", "in2")]),
    #                 (merge_node, predict_node, [("out", "image_to_predict")]),
    #             ])

    #         predict_node.inputs.predictor_files = self.predictor_files
    #         predict_node.inputs.crop_or_pad_percentage = tuple(self.crop_or_pad_percentage)
    #         predict_node.inputs.threshold = self.threshold
    #         predict_node.inputs.prefix = prefix
    #         predict_node.inputs.output_path = self.output_path
    #         predict_node.inputs.prediction_file = PVSprobmap_file
    #         predict_node.inputs.binary_file = PVSmask_file
    #         predict_node.inputs.thresholded_file = PVSlabel_file
    #         predict_node.inputs.target_shape = (160, 214, 176)
    #         predict_node.inputs.save_intermediate_image = self.save_intermediate_image

    #         shiva_pvs_pipeline.run()

    #         entities_T1wcropped = {
    #             'space': 'T1w',
    #             'desc': 'SHIVA',
    #         }
    #         entities_FLAIRcropped = {
    #             'space': 'FLAIR',
    #             'desc': 'SHIVA',
    #         }
    #         suffix_T1wcropped = "T1w"
    #         suffix_FLAIRcropped = "FLAIR"

    #         if self.save_intermediate_image:
    #             os.rename(os.path.join(self.output_path, f'crop_{t1w_filename}.nii.gz'), os.path.join(self.output_path, rename_bids_file(t1w_file, entities_T1wcropped, suffix_T1wcropped, extension)))
    #             if self.modality == 'T1w+FLAIR':
    #                 os.rename(os.path.join(self.output_path, f'crop_{flair_filename}.nii.gz'), os.path.join(self.output_path, rename_bids_file(flair_file, entities_FLAIRcropped, suffix_FLAIRcropped, extension)))