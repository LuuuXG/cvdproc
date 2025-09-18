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
#from .shiva_segmentation.shiva_segmentation import SHIVAPredictImage
from cvdproc.pipelines.smri.csvd_quantification.shiva_segmentation.shiva_parc import ShivaGeneralParcellation, Brain_Seg_for_biomarker
from cvdproc.pipelines.smri.freesurfer.synthseg import SynthSeg
from cvdproc.pipelines.smri.freesurfer.synthstrip import SynthStrip
from cvdproc.pipelines.smri.ants.n4biascorr_nipype import SimpleN4BiasFieldCorrection
from cvdproc.pipelines.smri.csvd_quantification.segcsvd.segment_pvs import SegCSVDPVS

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
            print("[PVS Pipeline] No specific T1w file selected. Using the first one.")
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
                print("[PVS Pipeline] No specific FLAIR file selected. Using the first one.")
                flair_files = [flair_files[0]]
                flair_file = flair_files[0]

            flair_filename = os.path.basename(flair_file).split('.nii')[0]

        pvs_quantification_workflow = Workflow(name='pvs_quantification_workflow')
        pvs_quantification_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')

        if self.method == 'SHIVA':
            from .shiva_segmentation.shiva_nipype import PrepareShivaInput, ShivaSegmentation

            shiva_input_dir = os.path.join(self.output_path, 'shiva_input')
            os.makedirs(shiva_input_dir, exist_ok=True)

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
        elif self.method == 'segcsvd':
            inputnode = Node(IdentityInterface(fields=['t1_path']), name='inputnode')
            inputnode.inputs.t1_path = t1w_file
            # 1. synthseg
            synthseg_out_ndoe = Node(IdentityInterface(fields=['synthseg_out']), name='synthseg_output')
            anat_seg_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'anat_seg', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
            synthseg_out = os.path.join(anat_seg_dir, 'synthseg', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_synthseg.nii.gz")
            if os.path.exists(synthseg_out):
                print(f"[PVS Pipeline] Found existing SynthSeg output: {synthseg_out}")
                synthseg_out_ndoe.inputs.synthseg_out = synthseg_out
            else:
                synthseg = Node(SynthSeg(), name='synthseg')
                pvs_quantification_workflow.connect(inputnode, 't1_path', synthseg, 'image')
                synthseg.inputs.out = os.path.join(anat_seg_dir, 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_synthseg.nii.gz')
                synthseg.inputs.vol = os.path.join(anat_seg_dir, 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-synthseg_volume.csv')
                synthseg.inputs.robust = True
                synthseg.inputs.parc = True
                synthseg.inputs.keepgeom = True

                pvs_quantification_workflow.connect(synthseg, 'out', synthseg_out_ndoe, 'synthseg_out')
            
            # 2. shiva derived brain seg for biomarker
            shiva_general_parc_node = Node(ShivaGeneralParcellation(), name='shiva_general_parcellation')
            pvs_quantification_workflow.connect(synthseg_out_ndoe, 'synthseg_out', shiva_general_parc_node, 'in_seg')
            shiva_general_parc_node.inputs.out_seg = os.path.join(self.output_path, f"sub-{self.subject.subject_id}{session_entity}_space-T1w_desc-shivaParc_synthseg.nii.gz")

            # 3. shiva pvs seg
            shiva_pvs_seg_node = Node(Brain_Seg_for_biomarker(), name='shiva_pvs_segmentation')
            pvs_quantification_workflow.connect(shiva_general_parc_node, 'out_seg', shiva_pvs_seg_node, 'brain_seg')
            shiva_pvs_seg_node.inputs.custom_parc = 'pvs'
            shiva_pvs_seg_node.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}{session_entity}_space-T1w_desc-shivaParcPVS_synthseg.nii.gz")
            shiva_pvs_seg_node.inputs.parc_json = os.path.join(self.output_path, f"shivaParcPVS.json")

            # 4. skull strip T1w and N4 bias correction
            stripped_t1w_node = Node(IdentityInterface(fields=['stripped_t1w']), name='stripped_t1w_output')
            xfm_output_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f"ses-{self.session.session_id}")
            os.makedirs(xfm_output_dir, exist_ok=True)
            t1w_stripped = os.path.join(xfm_output_dir, rename_bids_file(t1w_file, {'desc': 'brain'}, 'T1w', '.nii.gz'))
            if os.path.exists(t1w_stripped):
                print(f"[PVS Pipeline] Found existing skull-stripped T1w: {t1w_stripped}")
                stripped_t1w_node.inputs.stripped_t1w = t1w_stripped
            else:
                synthstrip_t1w_node = Node(SynthStrip(), name='synthstrip_t1w')
                pvs_quantification_workflow.connect(inputnode, 't1_path', synthstrip_t1w_node, 'image')
                synthstrip_t1w_node.inputs.out_file = t1w_stripped
                synthstrip_t1w_node.inputs.mask_file = os.path.join(self.output_path, rename_bids_file(t1w_file, {'label': 'brain', 'space': 'T1w'}, 'mask', '.nii.gz'))

                pvs_quantification_workflow.connect(synthstrip_t1w_node, 'out_file', stripped_t1w_node, 'stripped_t1w')
            
            n4_biascorr_node = Node(SimpleN4BiasFieldCorrection(), name='n4_bias_correction')
            pvs_quantification_workflow.connect(stripped_t1w_node, 'stripped_t1w', n4_biascorr_node, 'input_image')
            n4_biascorr_node.inputs.output_image = os.path.join(self.output_path, rename_bids_file(t1w_file, {'desc': 'n4biascorr'}, 'T1w', '.nii.gz'))
            n4_biascorr_node.inputs.output_bias = os.path.join(self.output_path, rename_bids_file(t1w_file, {'desc': 'n4biascorr'}, 'biasfield', '.nii.gz'))

            # 5. PVS segmentation
            # found whether there is an existing WMH file
            wmh_output_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'wmh_quantification', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
            # look for: lebel-WMH and space-T1w and mask.nii.gz, if multiple, use the first one
            try:
                wmh_files = [f for f in os.listdir(wmh_output_dir) if f.endswith('.nii.gz') and 'label-WMH' in f and 'space-T1w' in f and 'mask' in f]
                if wmh_files:
                    wmh_file = os.path.join(wmh_output_dir, wmh_files[0])
                    print(f"[PVS Pipeline] Found existing WMH file: {wmh_file}")
                else:
                    wmh_file = 'none'
                    print(f"[PVS Pipeline] No existing WMH file found. A pseudo WMH (all=0) will be created.")
            except:
                wmh_file = 'none'
                print(f"[PVS Pipeline] No existing WMH file found. A pseudo WMH (all=0) will be created.")

            segcsvd_pvs_node = Node(SegCSVDPVS(), name='segcsvd_pvs_segmentation')
            pvs_quantification_workflow.connect(n4_biascorr_node, 'output_image', segcsvd_pvs_node, 't1w')
            pvs_quantification_workflow.connect(synthseg_out_ndoe, 'synthseg_out', segcsvd_pvs_node, 'synthseg_out')
            segcsvd_pvs_node.inputs.wmh_file = wmh_file
            segcsvd_pvs_node.inputs.output_dir = self.output_path
            segcsvd_pvs_node.inputs.pvs_probmap_filename = f"sub-{self.subject.subject_id}{session_entity}_space-T1w_label-PVS_desc-segcsvd_probmap.nii.gz"
            segcsvd_pvs_node.inputs.pvs_binary_filename = f"sub-{self.subject.subject_id}{session_entity}_space-T1w_label-PVS_desc-segcsvdThr0p35_mask.nii.gz"
            segcsvd_pvs_node.inputs.threshold = 0.35

        return pvs_quantification_workflow