# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

# modified by Youjie Wang, 2025-01-20

import os
import pandas as pd
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

from cvdproc.pipelines.common.calculate_roi_volume import CalculateROIVolume

class PVSSegmentationPipeline:
    def __init__(self, 
                 subject: object, 
                 session: object, 
                 output_path: str,
                 use_which_t1w: str = None,
                 use_which_flair: str = None,
                 method: str = 'segcsvd',
                 modality: str = 'T1w',
                 shiva_config: str = None,
                 use_wmh: bool = False,
                 extract_from: str = None,
                 **kwargs):
        """
        PVS Segmentation Pipeline

        Args:
            subject: BIDSSubject object
            session: BIDSSession object
            output_path: output directory for the pipeline
            use_which_t1w: specific string to select T1w image, e.g. 'acq-highres'. If None, use the first T1w image found.
            use_which_flair: specific string to select FLAIR image, e.g. 'acq-highres'. If None, use the first FLAIR image found.
            method: 'SHIVA' or 'segcsvd'
            modality: 'T1w' or 'T1w+FLAIR'. Applicable when method is 'SHIVA'.
            shiva_config: path to SHIVA configuration file. Applicable when method is 'SHIVA'.
            use_wmh: whether to use existing WMH segmentation for PVS segmentation. Applicable when method is 'segcsvd'.
            extract_from: path to the output directory from which to extract results. (Currently only for 'segcsvd' outputs)
        """
        
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.use_wmh = use_wmh
        self.use_which_t1w = use_which_t1w
        self.use_which_flair = use_which_flair
        self.method = method
        self.modality = modality
        self.shiva_config = shiva_config if shiva_config is not None else (os.path.join(self.subject.bids_dir, 'code', 'shiva_config.yml') if self.method == 'SHIVA' and self.subject is not None else None)
        self.extract_from = extract_from

    def check_data_requirements(self):
        """
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
            synthseg_out = os.path.join(anat_seg_dir, 'synthseg', rename_bids_file(t1w_file, {'space': 'T1w'}, 'synthseg', '.nii.gz'))
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
            t1w_stripped = os.path.join(xfm_output_dir, rename_bids_file(t1w_file, {'desc': 'brain', 'space': 'T1w'}, 'T1w', '.nii.gz'))
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
                if wmh_files and self.use_wmh:
                    wmh_file = os.path.join(wmh_output_dir, wmh_files[0])
                    print(f"[PVS Pipeline] Found existing WMH file: {wmh_file}")
                elif wmh_files and not self.use_wmh:
                    wmh_file = 'none'
                    print(f"[PVS Pipeline] Existing WMH file found but use_wmh is set to False. A pseudo WMH (all=0) will be created.")
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

            # 6. ROI volume
            generate_volume_csv_node = Node(CalculateROIVolume(), name='calculate_roi_volume')
            pvs_quantification_workflow.connect(segcsvd_pvs_node, 'pvs_binary', generate_volume_csv_node, 'in_nii')
            pvs_quantification_workflow.connect(shiva_pvs_seg_node, 'brain_seg', generate_volume_csv_node, 'roi_nii')
            generate_volume_csv_node.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}{session_entity}_space-T1w_desc-shivaParcPVS_volume.csv")


        return pvs_quantification_workflow
    
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        pvs_seg_output_path = self.extract_from

        # SegCSVD columns
        segcsvd_columns = [
            'Subject', 'Session',
            'Left Deep WM', 'Left Basal Ganglia', 'Left Hippocampus', 'Left Cerebellar', 'Left Ventral DC',
            'Right Deep WM', 'Right Basal Ganglia', 'Right Hippocampus', 'Right Cerebellar', 'Right Ventral DC',
            'Brainstem',
            'Total PVS Volume (mm3)'
        ]
        segcsvd_df = pd.DataFrame(columns=segcsvd_columns)

        region_order = [
            0,  # Left Deep WM
            1,  # Left Basal Ganglia
            2,  # Left Hippocampus
            3,  # Left Cerebellar
            4,  # Left Ventral DC
            5,  # Right Deep WM
            6,  # Right Basal Ganglia
            7,  # Right Hippocampus
            8,  # Right Cerebellar
            9,  # Right Ventral DC
            10  # Brainstem
        ]

        for subject_folder in os.listdir(pvs_seg_output_path):
            subject_path = os.path.join(pvs_seg_output_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            # session level
            for session_folder in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session_folder)
                if not os.path.isdir(session_path):
                    continue

                # Find the correct CSV
                segcsvd_shivaparc_csv = [
                    f for f in os.listdir(session_path)
                    if f.endswith('desc-shivaParcPVS_volume.csv')
                ]

                if len(segcsvd_shivaparc_csv) != 1:
                    print(f"Warning: expected 1 CSV but found {len(segcsvd_shivaparc_csv)} in {session_path}")
                    continue

                csv_path = os.path.join(session_path, segcsvd_shivaparc_csv[0])

                # Load CSV
                df = pd.read_csv(csv_path)

                # Extract 11 region volumes by row index
                volumes = df.loc[region_order, "Volume (mm^3)"].tolist()

                # Total PVS = sum of the 11 regions
                total_volume = sum(volumes)

                # Build one row
                row = [subject_folder, session_folder] + volumes + [total_volume]

                # Append to dataframe
                segcsvd_df.loc[len(segcsvd_df)] = row

        # =============================
        # Save final CSV
        # =============================
        output_csv = os.path.join(self.output_path, "segcsvd_shivaparc_volume_summary.csv")
        segcsvd_df.to_csv(output_csv, index=False)

        print(f"Saved PVS quantification results to: {output_csv}")


                
