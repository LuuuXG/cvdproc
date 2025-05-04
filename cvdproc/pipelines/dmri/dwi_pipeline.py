import os
import pandas as pd
import json
import logging
import nibabel as nib
import numpy as np
logger = logging.getLogger(__name__)

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from .fdt.fdt_nipype import B0AllAndAcqparam, IndexTxt, Topup, EddyCuda, DTIFit, DTIpreprocessing, PrepareBedpostx, Bedpostx, Tractography, ExtractSurfaceParameters, DTIALPSsimple
from .synb0.synb0_nipype import Synb0
from .freewater.single_shell_freewater import SingleShellFW
from ..smri.mirror.mirror_nipype import MirrorMask
from nipype.interfaces.fsl import FLIRT, TOPUP, Eddy, ExtractROI, ConvertXFM
from nipype.interfaces.dipy import Denoise
from ..smri.fsl.fsl_anat_nipype import FSLANAT
from nipype.interfaces.mrtrix3 import Tractography, MRConvert, DWIDenoise, MRDeGibbs, DWIPreproc
from cvdproc.pipelines.dmri.mrtrix3.tcksample_nipype import TckSampleCommand, CalculateMeanTckSample
from cvdproc.pipelines.dmri.stats.dti_scalar_maps import GenerateWMMaskCommandLine, CalculateScalarMaps
from ..common.unzip import UnzipCommandLine
from ..common.move_file import MoveFileCommandLine
from ..common.copy_file import CopyFileCommandLine
from ..common.delete_file import DeleteFileCommandLine
from ..common.mri_synthstrip import MRISynthstripCommandLine
from cvdproc.pipelines.common.filter_existing import FilterExisting
from cvdproc.pipelines.common.merge_filename import MergeFilename
from .psmd.psmd_nipype import PSMDCommandLine, SavePSMDOutputCommandLine

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class DWIPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        DWI pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_dwi = kwargs.get('use_which_dwi', None) # which dwi file to use
        self.use_which_t1w = kwargs.get('use_which_t1w', None) # which t1w file to use
        self.use_which_flair = kwargs.get('use_which_flair', None) # which flair file to use

        # Preprocessing
        self.preprocess = kwargs.get('preprocess', False) # whether to preprocess
        self.preprocess_method = kwargs.get('preprocess_method', 'fdt') # preprocessing method
        self.synb0 = kwargs.get('synb0', False) # whether to use synthetic b0 image for TOPUP
        self.use_which_reverse_b0 = kwargs.get('use_which_reverse_b0', None) # whether to use reverse b0 image for TOPUP. if synb0 is True, this will be ignored

        # DTI fit
        self.dti_fit = kwargs.get('dti_fit', True) # whether to fit DTI model

        # ROI-based tractography
        self.tractography = kwargs.get('tractography', None)
        self.seed_mask = kwargs.get('seed_mask', 'lesion_mask') # seed mask for probabilistic tractography
        self.use_which_mask = kwargs.get('use_which_mask', None) # which mask to use (in the processed DTI space)
        self.tckgen_method = kwargs.get('tckgen_method', 'iFOD2')
        self.use_freesurfer_clinical = kwargs.get('use_freesurfer_clinical', False) # whether to use freesurfer clinical directory. False will use freesurfer

        # DTI-ALPS
        self.dtialps = kwargs.get('dtialps', False)

        # Freewater
        self.single_shell_freewater = kwargs.get('single_shell_freewater', False)

        # PSMD
        self.psmd = kwargs.get('psmd', False)
        self.psmd_lesion_mask = kwargs.get('psmd_lesion_mask', None) # Mask to exclude in PSMD calculation
        self.use_which_psmd_lesion_mask = kwargs.get('use_which_psmd_lesion_mask', None)

        # Calculate scalar maps
        self.calculate_dwi_metrics = kwargs.get('calculate_dwi_metrics', False) # whether to calculate DWI metrics
        self.exclude_seed_mask = kwargs.get('exclude_seed_mask', True) # whether to exclude seed mask in DWI metrics calculation
        self.exclude_wmh_mask = kwargs.get('exclude_wmh_mask', False) # whether to exclude WMH mask in DWI metrics calculation

        # extract results
        self.extract_from = kwargs.get('extract_from', None)

        #### Not user configurable ####
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.script_path2 = os.path.join(base_dir, 'bash', 'fdt_fs_processing.sh') # FreeSurfer processing
        self.alps_script_path = os.path.join(base_dir, 'external', 'alps', 'alps.sh') # DTI-ALPS
        self.psmd_skeleton_mask = os.path.join(base_dir, 'external', 'psmd', 'skeleton_mask_2019.nii.gz')

    def check_data_requirements(self):
        """
        检查数据需求
        :return: bool
        """
        return self.session.get_dwi_files() is not None and self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # Find Image Files
        os.makedirs(self.output_path, exist_ok=True)

        dwi_files = self.session.get_dwi_files()
        phase_encoding_direction_dict = {
            "i": "-1 0 0",
            "i-": "1 0 0",
            "j": "0 -1 0",
            "j-": "0 1 0"
        }  
        
        if self.use_which_dwi:
            dwi_files = [file for file in dwi_files if self.use_which_dwi in file['path']]
            
            if len(dwi_files) != 4:
                raise FileNotFoundError(f"No specific DWI file found for {self.use_which_dwi} or more than one found or not in standard BIDS format.")
        else:
            print("No specific DWI file selected. Using the first complete set.")

            grouped_files = {}
            
            for file in dwi_files:
                base_name = file['path'].split('_acq-')[0]
                if base_name not in grouped_files:
                    grouped_files[base_name] = []
                grouped_files[base_name].append(file)

            first_complete_group = None
            for group in grouped_files.values():
                file_types = {file['type'] for file in group}
                
                if {'dwi', 'bval', 'bvec', 'json'}.issubset(file_types):
                    first_complete_group = group
                    break

            if not first_complete_group:
                raise FileNotFoundError("No complete set of DWI files found.")

            dwi_files = first_complete_group

        dwi_files_dict = {file['type']: file['path'] for file in dwi_files}
        print(f"Using DWI files: {dwi_files_dict}")

        dwi_image = dwi_files_dict['dwi']
        dwi_json = dwi_files_dict['json']
        dwi_bval = dwi_files_dict['bval']
        dwi_bvec = dwi_files_dict['bvec']

        # Get the total readout time and phase encoding direction from the json file
        with open(dwi_json, 'r') as f:
            dwi_json_data = json.load(f)
        try:
            total_readout_time = dwi_json_data['TotalReadoutTime']
            # as a str
            total_readout_time = str(total_readout_time)
            phase_encoding_direction = dwi_json_data['PhaseEncodingDirection']     
        except KeyError:
            logger.warning("Seems TotalReadoutTime or PhaseEncodingDirection not found in the json file.")
            if not total_readout_time:
                total_readout_time = '0.05'
                logger.warning("Set TotalReadoutTime to 0.05. But make sure the value is the same in all acquisitions.")

        # Find Reverse b0 Image
        if not self.synb0 and self.use_which_reverse_b0 is not None:
            dwi_files = self.session.get_dwi_files()
            reverse_b0_files = [file for file in dwi_files if self.use_which_reverse_b0 in file['path']]
            
            if len(reverse_b0_files) != 4:
                #logger.error(f"Found reverse b0 files: {reverse_b0_files}")
                raise FileNotFoundError(f"No specific reverse b0 file found for {self.use_which_reverse_b0} or more than one found.")
            
            reverse_b0_files_dict = {file['type']: file['path'] for file in reverse_b0_files}
            logger.info(f"Using reverse b0 files: {reverse_b0_files_dict}")

            reverse_b0_image = reverse_b0_files_dict['dwi']
            reverse_b0_json = reverse_b0_files_dict['json']
            reverse_b0_bval = reverse_b0_files_dict['bval']
            reverse_b0_bvec = reverse_b0_files_dict['bvec']

            with open(reverse_b0_json, 'r') as f:
                reverse_b0_json_data = json.load(f)
            try:
                reverse_b0_total_readout_time = str(reverse_b0_json_data['TotalReadoutTime'])
                reverse_b0_phase_encoding_direction = reverse_b0_json_data['PhaseEncodingDirection']

            except KeyError:
                logger.warning("Seems TotalReadoutTime or PhaseEncodingDirection not found in the json file.")
                if not reverse_b0_total_readout_time:
                    reverse_b0_total_readout_time = '0.05'
                    logger.warning("Set TotalReadoutTime to 0.05. But make sure the value is the same in all acquisitions.")
        else:
            reverse_b0_image = None

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
        
        # get FLAIR image
        flair_files = self.session.get_flair_files()
        if self.use_which_flair:
            flair_files = [f for f in flair_files if self.use_which_flair in f]
            if len(flair_files) != 1:
                raise FileNotFoundError(f"No specific FLAIR file found for {self.use_which_flair} or more than one found.")
            flair_file = flair_files[0]
        else:
            print("No specific FLAIR file selected. Using the first one.")
            flair_files = [flair_files[0]]
            flair_file = flair_files[0]

        if self.use_freesurfer_clinical:
            fs_output = self.session.freesurfer_clinical_dir
        else:
            fs_output = self.session.freesurfer_dir
        
        ################
        # dwi_workflow #
        ################
        dwi_workflow = Workflow(name='dwi_workflow')
        dwi_workflow.base_dir = self.output_path

        inputnode = Node(IdentityInterface(fields=['bids_dir', 't1w_file',
                                                   'dwi_file', 'bval_file',
                                                   'bvec_file', 'json_file',
                                                   'use_synb0', 'output_path_synb0',
                                                   'output_path', 
                                                   'phase_encoding_number', 'total_readout_time', 
                                                   'fs_output', 'seed_mask_dtispace',
                                                   'fs_processing_dir']), name="inputnode")
        
        inputnode.inputs.bids_dir = self.subject.bids_dir
        inputnode.inputs.t1w_file = t1w_file
        inputnode.inputs.dwi_file = dwi_image
        inputnode.inputs.bval_file = dwi_bval
        inputnode.inputs.bvec_file = dwi_bvec
        inputnode.inputs.json_file = dwi_json
        inputnode.inputs.use_synb0 = self.synb0
        inputnode.inputs.output_path_synb0 = os.path.join(self.subject.bids_dir, 'derivatives', 'synb0', f'sub-{self.subject.subject_id}', f"ses-{self.session.session_id}")
        inputnode.inputs.output_path = self.output_path
        inputnode.inputs.phase_encoding_number = phase_encoding_direction_dict[phase_encoding_direction]
        inputnode.inputs.total_readout_time = total_readout_time

        ############################################
        #               Prerocessing               #
        ############################################

        # Node to store the b0_all and acqparam for TOPUP
        b0_all_node = Node(IdentityInterface(fields=['b0_all', 'acqparam']), name='b0_all')

        # Node to store preprocessed DWI data
        preproc_dwi_node = Node(IdentityInterface(fields=['preproc_dwi', 'bvec', 'bval', 'dwi_mask']), name='preproc_dwi')

        if self.preprocess_method is None:
            raise ValueError("Preprocess method is not specified. Please specify a preprocess method.")
        else:
            if self.synb0:
                synb0_node = Node(Synb0(), name='synb0')
                dwi_workflow.connect([
                    (inputnode, synb0_node, [('t1w_file', 't1w_img'),
                                             ('dwi_file', 'dwi_img'),
                                             ('output_path_synb0', 'output_path_synb0'),
                                             ('phase_encoding_number', 'phase_encoding_number'),
                                             ('total_readout_time', 'total_readout_time')])
                ])

                dwi_workflow.connect(synb0_node, 'acqparam', b0_all_node, 'acqparam')
                dwi_workflow.connect(synb0_node, 'b0_all', b0_all_node, 'b0_all')
            else:
                if reverse_b0_image is not None:
                    create_b0_acqparam_node = Node(B0AllAndAcqparam(), name='create_b0_acqparam')
                    dwi_workflow.connect([
                        (inputnode, create_b0_acqparam_node, [('dwi_file', 'dwi_img'),
                                                            ('bval_file', 'dwi_bval'),
                                                            ('phase_encoding_number', 'phase_encoding_number'),
                                                            ('total_readout_time', 'total_readout_time'),
                                                            ('output_path', 'output_path')])
                    ])
                    create_b0_acqparam_node.inputs.reverse_dwi_img = reverse_b0_image
                    create_b0_acqparam_node.inputs.reverse_dwi_bval = reverse_b0_bval
                    create_b0_acqparam_node.inputs.reverse_phase_encoding_number = phase_encoding_direction_dict[reverse_b0_phase_encoding_direction]
                    create_b0_acqparam_node.inputs.reverse_total_readout_time = reverse_b0_total_readout_time

                    dwi_workflow.connect(create_b0_acqparam_node, 'b0_all', b0_all_node, 'b0_all')
                    dwi_workflow.connect(create_b0_acqparam_node, 'acqparam', b0_all_node, 'acqparam')
            
            if self.preprocess_method == 'fdt':
                if self.preprocess:
                    # denoise_node = Node(Denoise(), name='denoise')
                    # dwi_workflow.connect(inputnode, 'dwi_file', denoise_node, 'in_file')
                    # denoise_node.inputs.noise_model = 'rician'

                    topup_node = Node(Topup(), name='topup')
                    dwi_workflow.connect([
                        (b0_all_node, topup_node, [('b0_all', 'b0_all_file'),
                                                    ('acqparam', 'acqparam_file')])
                    ])
                    topup_node.inputs.output_basename = os.path.join(self.output_path, 'topup_results')
                    topup_node.inputs.output_b0_basename = os.path.join(self.output_path, 'hifi_b0')
                    topup_node.inputs.config_file = 'b02b0.cnf'

                    extract_b0_node = Node(ExtractROI(), name='extract_b0')
                    dwi_workflow.connect(topup_node, 'b0_image', extract_b0_node, 'in_file')
                    extract_b0_node.inputs.roi_file = os.path.join(self.output_path, 'dwi_b0.nii.gz')
                    extract_b0_node.inputs.t_min = 0
                    extract_b0_node.inputs.t_size = 1

                    create_dwi_mask_node = Node(MRISynthstripCommandLine(), name='create_dwi_mask')
                    dwi_workflow.connect([
                        (extract_b0_node, create_dwi_mask_node, [('roi_file', 'input_file')])
                    ])
                    create_dwi_mask_node.inputs.mask_file = os.path.join(self.output_path, 'dwi_b0_brain_mask.nii.gz')

                    create_index_node = Node(IndexTxt(), name='create_index')
                    dwi_workflow.connect(inputnode, 'bval_file', create_index_node, 'bval_file')
                    dwi_workflow.connect(inputnode, 'output_path', create_index_node, 'output_dir')

                    eddy_node = Node(EddyCuda(), name='eddy')
                    dwi_workflow.connect([
                        (inputnode, eddy_node, [('dwi_file', 'dwi_file'), ('bvec_file', 'bvec_file'), ('bval_file', 'bval_file')]),
                        (b0_all_node, eddy_node, [('acqparam', 'acqparam_file')]),
                        (create_index_node, eddy_node, [('index_file', 'index_file')]),
                        (create_dwi_mask_node, eddy_node, [('mask_file', 'mask_file')]),
                        (topup_node, eddy_node, [('topup_basename', 'topup_basename')]),
                    ])
                    eddy_node.inputs.output_basename = os.path.join(self.output_path, 'eddy_corrected_data')

                    dwi_workflow.connect(eddy_node, 'eddy_corrected_data', preproc_dwi_node, 'preproc_dwi')
                    dwi_workflow.connect(eddy_node, 'eddy_corrected_bvecs', preproc_dwi_node, 'bvec')
                    dwi_workflow.connect(inputnode, 'bval_file', preproc_dwi_node, 'bval')
                    dwi_workflow.connect(create_dwi_mask_node, 'mask_file', preproc_dwi_node, 'dwi_mask')
                else:
                    preproc_dwi_node.inputs.preproc_dwi = os.path.join(self.output_path, 'eddy_corrected_data.nii.gz')
                    preproc_dwi_node.inputs.bvec = os.path.join(self.output_path, 'eddy_corrected_data.eddy_rotated_bvecs')
                    preproc_dwi_node.inputs.bval = dwi_bval
                    preproc_dwi_node.inputs.dwi_mask = os.path.join(self.output_path, 'dwi_b0_brain_mask.nii.gz')
            elif self.preprocess_method == 'mrtrix3':
                if self.preprocess:
                    # Get dwi_raw.mif
                    convert_raw_dwi_mif_node = Node(MRConvert(), name='convert_raw_dwi_mif')
                    dwi_workflow.connect(inputnode, 'dwi_file', convert_raw_dwi_mif_node, 'in_file')
                    dwi_workflow.connect(inputnode, 'bvec_file', convert_raw_dwi_mif_node, 'in_bvec')
                    dwi_workflow.connect(inputnode, 'bval_file', convert_raw_dwi_mif_node, 'in_bval')
                    convert_raw_dwi_mif_node.inputs.out_file = os.path.join(self.output_path, 'dwi_raw.mif')
                    convert_raw_dwi_mif_node.inputs.args = '-force'

                    # Get b0_all.mif
                    convert_b0_all_mif_node = Node(MRConvert(), name='convert_b0_all_mif')
                    dwi_workflow.connect(b0_all_node, 'b0_all', convert_b0_all_mif_node, 'in_file')
                    convert_b0_all_mif_node.inputs.out_file = os.path.join(self.output_path, 'b0_all.mif')
                    convert_b0_all_mif_node.inputs.args = '-force'
                    #convert_raw_dwi_mif_node.inputs.args = f'-import_pe_table {os.path.join(self.output_path, "acqparam.txt")}'

                    # denoise
                    denoise_node = Node(DWIDenoise(), name='denoise')
                    dwi_workflow.connect(convert_raw_dwi_mif_node, 'out_file', denoise_node, 'in_file')
                    denoise_node.inputs.out_file = os.path.join(self.output_path, 'dwi_denoised.mif')
                    denoise_node.inputs.noise = os.path.join(self.output_path, 'noise.mif')

                    # degibbs
                    degibbs_node = Node(MRDeGibbs(), name='degibbs')
                    dwi_workflow.connect(denoise_node, 'out_file', degibbs_node, 'in_file')
                    degibbs_node.inputs.out_file = os.path.join(self.output_path, 'dwi_denoised_degibbs.mif')

                    # TOPUP and Eddy (I don't know how to integrate TOPUP and Eddy in mrtrix3 if use synb0 :( )
                    # preprocess
                    dwi_preprocess_node = Node(DWIPreproc(), name='dwi_preprocess')
                    dwi_workflow.connect(degibbs_node, 'out_file', dwi_preprocess_node, 'in_file')
                    dwi_workflow.connect(convert_b0_all_mif_node, 'out_file', dwi_preprocess_node, 'in_epi')
                    dwi_preprocess_node.inputs.out_file = os.path.join(self.output_path, 'dwi_preproc.mif')
                    dwi_preprocess_node.inputs.rpe_options = 'pair'
                    dwi_preprocess_node.inputs.pe_dir = phase_encoding_direction

                    # alternative of preprocess
                    # https://community.mrtrix.org/t/synb0-for-dwifslpreproc-how/6386/2
                    #TODO

                    # convert to nifti
                    convert_preproc_dwi_node = Node(MRConvert(), name='convert_preproc_dwi')
                    dwi_workflow.connect(dwi_preprocess_node, 'out_file', convert_preproc_dwi_node, 'in_file')
                    convert_preproc_dwi_node.inputs.out_file = os.path.join(self.output_path, 'dwi_preproc.nii.gz')
                    convert_preproc_dwi_node.inputs.out_bvec = os.path.join(self.output_path, 'dwi_preproc.bvec')
                    convert_preproc_dwi_node.inputs.out_bval = os.path.join(self.output_path, 'dwi_preproc.bval')
                    convert_preproc_dwi_node.inputs.args = '-force'

                    # get the mask
                    extract_b0_node = Node(ExtractROI(), name='extract_b0')
                    dwi_workflow.connect(convert_preproc_dwi_node, 'out_file', extract_b0_node, 'in_file')
                    extract_b0_node.inputs.roi_file = os.path.join(self.output_path, 'dwi_b0.nii.gz')
                    extract_b0_node.inputs.t_min = 0
                    extract_b0_node.inputs.t_size = 1

                    create_dwi_mask_node = Node(MRISynthstripCommandLine(), name='create_dwi_mask')
                    dwi_workflow.connect(extract_b0_node, 'roi_file', create_dwi_mask_node, 'input_file')
                    create_dwi_mask_node.inputs.mask_file = os.path.join(self.output_path, 'dwi_b0_brain_mask.nii.gz')

                    dwi_workflow.connect(convert_preproc_dwi_node, 'out_file', preproc_dwi_node, 'preproc_dwi')
                    dwi_workflow.connect(convert_preproc_dwi_node, 'out_bvec', preproc_dwi_node, 'bvec')
                    dwi_workflow.connect(convert_preproc_dwi_node, 'out_bval', preproc_dwi_node, 'bval')
                    dwi_workflow.connect(create_dwi_mask_node, 'mask_file', preproc_dwi_node, 'dwi_mask')
                else:
                    preproc_dwi_node.inputs.preproc_dwi = os.path.join(self.output_path, 'dwi_preproc.nii.gz')
                    preproc_dwi_node.inputs.bvec = os.path.join(self.output_path, 'dwi_preproc.bvec')
                    preproc_dwi_node.inputs.bval = os.path.join(self.output_path, 'dwi_preproc.bval')
                    preproc_dwi_node.inputs.dwi_mask = os.path.join(self.output_path, 'dwi_b0_brain_mask.nii.gz')
                    
        
        ##########################
        #         DTI fit        #
        ##########################
        dti_fit_output_node = Node(IdentityInterface(fields=['output_dir', 'fa_img', 'md_img', 'tensor_img']), name='dti_fit_output')

        if self.dti_fit:
            dti_fit_node = Node(DTIFit(), name='dti_fit')
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', dti_fit_node, 'dwi_file')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', dti_fit_node, 'bvec_file')
            dwi_workflow.connect(preproc_dwi_node, 'bval', dti_fit_node, 'bval_file')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', dti_fit_node, 'mask_file')
            dti_fit_node.inputs.output_basename = os.path.join(self.output_path, 'dti')

            dwi_workflow.connect([
                (dti_fit_node, dti_fit_output_node, [('dti_fa', 'fa_img'),
                                                     ('dti_md', 'md_img'),
                                                     ('dti_tensor', 'tensor_img'),
                                                     ('output_dir', 'output_dir')]),
            ])
        else:
            dti_fit_output_node.inputs.fa_img = os.path.join(self.output_path, 'dti_FA.nii.gz')
            dti_fit_output_node.inputs.md_img = os.path.join(self.output_path, 'dti_MD.nii.gz')
            dti_fit_output_node.inputs.tensor_img = os.path.join(self.output_path, 'dti_tensor.nii.gz')
            dti_fit_output_node.inputs.output_dir = self.output_path
        
        #########################################################################
        # 2. Tractography: Designed for lesion-based probabilistic tractography #
        #########################################################################

        # Decide one seed mask to use
        lesion_mask_dir = self.session._find_output(self.seed_mask)
        seed_mask = [file for file in os.listdir(lesion_mask_dir) if f'{self.use_which_mask}' in file]
        if seed_mask is not None and len(seed_mask) == 1:
            print(f"Using seed mask: {seed_mask[0]}.")
            seed_mask = os.path.join(lesion_mask_dir, seed_mask[0])
        elif seed_mask is not None and len(seed_mask) > 1:
            print(f"Using the first mask found: {seed_mask[0]}.")
            seed_mask = os.path.join(lesion_mask_dir, seed_mask[0])
            print(f"Using seed mask: {seed_mask}.")
        else:
            seed_mask = ''
            print("No seed mask found.")
            #raise FileNotFoundError(f"No seed mask found in {lesion_mask_dir}. Please check the directory.")

        tractography_output_node = Node(IdentityInterface(fields=['fs_to_dwi_xfm', 't1w_to_dwi_xfm']), name='tractography_output')
        fs_processing_dir = os.path.join(self.output_path, 'fs_processing')
    
        if 'fdt' in self.tractography:
            os.makedirs(fs_processing_dir, exist_ok=True)
            probtrackx_output_dir1 = os.path.join(self.output_path, 'probtrackx_output')
            os.makedirs(probtrackx_output_dir1, exist_ok=True)
            probtrackx_output_dir2 = os.path.join(self.output_path, 'probtrackx_output_PathLengthCorrected')
            os.makedirs(probtrackx_output_dir2, exist_ok=True)

            prepare_bedpostx_node = Node(PrepareBedpostx(), name='prepare_bedpostx')
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', prepare_bedpostx_node, 'dwi_img')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', prepare_bedpostx_node, 'bvec')
            dwi_workflow.connect(preproc_dwi_node, 'bval', prepare_bedpostx_node, 'bval')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', prepare_bedpostx_node, 'mask')
            prepare_bedpostx_node.inputs.output_dir = os.path.join(self.output_path, 'bedpostX_input')

            bedpostx_node = Node(Bedpostx(), name='bedpostx')
            dwi_workflow.connect(prepare_bedpostx_node, 'output_dir', bedpostx_node, 'input_dir')

            tractography_node = Node(Tractography(), name='tractography')
            tractography_node.inputs.fs_output = fs_output
            tractography_node.inputs.seed_mask_dtispace = seed_mask
            tractography_node.inputs.fs_processing_dir = fs_processing_dir
            dwi_workflow.connect(inputnode, 't1w_file', tractography_node, 't1w_file')
            #dwi_workflow.connect(dtipreprocessing_node, 'fa_file', tractography_node, 'fa_file')
            tractography_node.inputs.fa_file = os.path.join(self.output_path, 'dti_FA.nii.gz')
            tractography_node.inputs.script_path_fspreprocess = self.script_path2
            if self.preprocess:
                dwi_workflow.connect(bedpostx_node, 'output_dir', tractography_node, 'bedpostx_output_dir')
            else:
                tractography_node.inputs.bedpostx_output_dir = os.path.join(self.output_path, 'bedpostX_input.bedpostX')
            tractography_node.inputs.probtrackx_output_dir1 = probtrackx_output_dir1
            tractography_node.inputs.probtrackx_output_dir2 = probtrackx_output_dir2

            extract_parameters_node = Node(ExtractSurfaceParameters(), name='extract_parameters')
            dwi_workflow.connect(tractography_node, 'lh_unconn_mask', extract_parameters_node, 'lh_unconn_mask')
            dwi_workflow.connect(tractography_node, 'rh_unconn_mask', extract_parameters_node, 'rh_unconn_mask')
            dwi_workflow.connect(tractography_node, 'lh_unconn_corrected_mask', extract_parameters_node, 'lh_unconn_corrected_mask')
            dwi_workflow.connect(tractography_node, 'rh_unconn_corrected_mask', extract_parameters_node, 'rh_unconn_corrected_mask')
            dwi_workflow.connect(tractography_node, 'lh_low_conn_mask', extract_parameters_node, 'lh_low_conn_mask')
            dwi_workflow.connect(tractography_node, 'rh_low_conn_mask', extract_parameters_node, 'rh_low_conn_mask')
            dwi_workflow.connect(tractography_node, 'lh_low_conn_corrected_mask', extract_parameters_node, 'lh_low_conn_corrected_mask')
            dwi_workflow.connect(tractography_node, 'rh_low_conn_corrected_mask', extract_parameters_node, 'rh_low_conn_corrected_mask')
            dwi_workflow.connect(tractography_node, 'lh_medium_conn_mask', extract_parameters_node, 'lh_medium_conn_mask')
            dwi_workflow.connect(tractography_node, 'rh_medium_conn_mask', extract_parameters_node, 'rh_medium_conn_mask')
            dwi_workflow.connect(tractography_node, 'lh_medium_conn_corrected_mask', extract_parameters_node, 'lh_medium_conn_corrected_mask')
            dwi_workflow.connect(tractography_node, 'rh_medium_conn_corrected_mask', extract_parameters_node, 'rh_medium_conn_corrected_mask')
            dwi_workflow.connect(tractography_node, 'lh_high_conn_mask', extract_parameters_node, 'lh_high_conn_mask')
            dwi_workflow.connect(tractography_node, 'rh_high_conn_mask', extract_parameters_node, 'rh_high_conn_mask')
            dwi_workflow.connect(tractography_node, 'lh_high_conn_corrected_mask', extract_parameters_node, 'lh_high_conn_corrected_mask')
            dwi_workflow.connect(tractography_node, 'rh_high_conn_corrected_mask', extract_parameters_node, 'rh_high_conn_corrected_mask')
            extract_parameters_node.inputs.output_dir = self.output_path
            extract_parameters_node.inputs.fs_subjects_dir = os.path.dirname(fs_output)
            extract_parameters_node.inputs.sessions = self.subject.sessions_id
            extract_parameters_node.inputs.csv_file_name = 'surface_parameters.csv'

            # experimental: mirror mask
            fsl_anat_node = Node(FSLANAT(), name='fsl_anat')
            dwi_workflow.connect(inputnode, 't1w_file', fsl_anat_node, 'input_image')
            fsl_anat_node.inputs.output_directory = os.path.join(self.subject.bids_dir, 'derivatives', 'fsl_anat', f'sub-{self.session.subject_id}', f"ses-{self.session.session_id}", 'fsl')
            os.makedirs(os.path.dirname(fsl_anat_node.inputs.output_directory), exist_ok=True)

            mirror_mask_node = Node(MirrorMask(), name='mirror_mask')
            mirror_mask_node.inputs.in_file = os.path.join(fs_processing_dir, 'seed_mask_in_t1w.nii.gz')
            dwi_workflow.connect(inputnode, 't1w_file', mirror_mask_node, 't1w_file')
            mirror_mask_node.inputs.out_dir = fs_processing_dir
            # mirror_mask_node.inputs.t1w_to_mni_xfm = os.path.join(self.session.fsl_anat_dir, 'fsl.anat', 'T1_to_MNI_nonlin_field.nii.gz')
            # mirror_mask_node.inputs.mni_to_t1w_xfm = os.path.join(self.session.fsl_anat_dir, 'fsl.anat', 'MNI_to_T1_nonlin_field.nii.gz')
            dwi_workflow.connect(fsl_anat_node, 'output_directory', mirror_mask_node, 'fsl_anat_output_dir')

            mirror_mask_node.inputs.mask_in_mni_filename = 'seed_mask_in_mni.nii.gz'
            mirror_mask_node.inputs.flipped_mask_mni_filename = 'seed_mask_in_mni_flipped.nii.gz'
            mirror_mask_node.inputs.flipped_mask_t1w_filename = 'seed_mask_in_t1w_flipped.nii.gz'

            mirror_mask_t1w_to_fs_node = Node(FLIRT(), name='mirror_mask_t1w_to_fs')
            dwi_workflow.connect(mirror_mask_node, 'flipped_mask_t1w', mirror_mask_t1w_to_fs_node, 'in_file')
            t1w_to_fs_xfm = os.path.join(fs_processing_dir, 'struct2freesurfer.mat')
            mirror_mask_t1w_to_fs_node.inputs.in_matrix_file = t1w_to_fs_xfm
            dwi_workflow.connect(tractography_node, 'fs_orig', mirror_mask_t1w_to_fs_node, 'reference')
            mirror_mask_t1w_to_fs_node.inputs.out_file = os.path.join(fs_processing_dir, 'seed_mask_in_fs_flipped.nii.gz')
            mirror_mask_t1w_to_fs_node.inputs.apply_xfm = True
            mirror_mask_t1w_to_fs_node.inputs.interp = 'nearestneighbour'

            mirror_tractography_node = Node(Tractography(), name='mirror_tractography')
            mirror_tractography_node.inputs.fs_output = fs_output
            dwi_workflow.connect(tractography_node, 'fs_processing_dir', mirror_tractography_node, 'fs_processing_dir')
            dwi_workflow.connect(inputnode, 't1w_file', mirror_tractography_node, 't1w_file')
            mirror_tractography_node.inputs.skip_fs_preprocess = True
            dwi_workflow.connect(mirror_mask_t1w_to_fs_node, 'out_file', mirror_tractography_node, 'seed_mask_fsspace')
            mirror_tractography_node.inputs.bedpostx_output_dir = os.path.join(self.output_path, 'bedpostX_input.bedpostX')
            mirror_tractography_node.inputs.probtrackx_output_dir1 = os.path.join(self.output_path, 'mirror_probtrackx_output')
            mirror_tractography_node.inputs.probtrackx_output_dir2 = os.path.join(self.output_path, 'mirror_probtrackx_output_PathLengthCorrected')

            mirror_extract_parameters_node = Node(ExtractSurfaceParameters(), name='mirror_extract_parameters')
            dwi_workflow.connect([
                (mirror_tractography_node, mirror_extract_parameters_node, [('lh_unconn_mask', 'lh_unconn_mask'),
                                                                                  ('rh_unconn_mask', 'rh_unconn_mask'),
                ('lh_unconn_corrected_mask', 'lh_unconn_corrected_mask'),
                ('rh_unconn_corrected_mask', 'rh_unconn_corrected_mask'),
                ('lh_low_conn_mask', 'lh_low_conn_mask'),
                ('rh_low_conn_mask', 'rh_low_conn_mask'),
                ('lh_low_conn_corrected_mask', 'lh_low_conn_corrected_mask'),
                ('rh_low_conn_corrected_mask', 'rh_low_conn_corrected_mask'),
                ('lh_medium_conn_mask', 'lh_medium_conn_mask'),
                ('rh_medium_conn_mask', 'rh_medium_conn_mask'),
                ('lh_medium_conn_corrected_mask', 'lh_medium_conn_corrected_mask'),
                ('rh_medium_conn_corrected_mask', 'rh_medium_conn_corrected_mask'),
                ('lh_high_conn_mask', 'lh_high_conn_mask'),
                ('rh_high_conn_mask', 'rh_high_conn_mask'),
                ('lh_high_conn_corrected_mask', 'lh_high_conn_corrected_mask'),
                ('rh_high_conn_corrected_mask', 'rh_high_conn_corrected_mask')])
            ])
            mirror_extract_parameters_node.inputs.output_dir = self.output_path
            mirror_extract_parameters_node.inputs.fs_subjects_dir = os.path.dirname(fs_output)
            mirror_extract_parameters_node.inputs.sessions = self.subject.sessions_id
            mirror_extract_parameters_node.inputs.csv_file_name = 'mirror_surface_parameters.csv'

            dwi_workflow.connect(tractography_node, 'fs_to_fa_xfm', tractography_output_node, 'fs_to_dwi_xfm')
            dwi_workflow.connect(tractography_node, 't1w_to_fa_xfm', tractography_output_node, 't1w_to_dwi_xfm')
        else:
            tractography_output_node.inputs.fs_to_dwi_xfm = os.path.join(fs_processing_dir, 'freesurfer2fa.mat')
            tractography_output_node.inputs.t1w_to_dwi_xfm = os.path.join(fs_processing_dir, 'struct2fa.mat')
        
        seed_based_track_node = Node(IdentityInterface(fields=['seed_based_track']), name='seed_based_track')
        tckgen_outout_dir = os.path.join(self.output_path, 'tckgen_output')

        if 'mrtrix3' in self.tractography:
            # self.tckgen_method must be in one of: 'FACT', 'iFOD2', 'iFOD1', 'NullDist1', 'NullDist2', 'SD_STREAM',
            # 'SeedTest', 'Tensor_Det', 'Tensor_Prob'
            if self.tckgen_method not in ['FACT', 'iFOD2', 'iFOD1', 'NullDist1', 'NullDist2', 'SD_STREAM',
                                        'SeedTest', 'Tensor_Det', 'Tensor_Prob']:
                raise ValueError(f"Invalid tckgen method: {self.tckgen_method}. Must be one of: "
                                 "'FACT', 'iFOD2', 'iFOD1', 'NullDist1', 'NullDist2', 'SD_STREAM', "
                                 "'SeedTest', 'Tensor_Det', 'Tensor_Prob'")
            
            logger.warning(f"Using tckgen method: {self.tckgen_method}.")
            logger.warning(f"Only these methods are tested: 'Tensor_Det', 'Tensor_Prob'.")
            
            mri_convert_node = Node(MRConvert(), name='mri_convert')
            mri_convert_node.inputs.out_file = os.path.join(self.output_path, 'preproc_dwi.mif')
            mri_convert_node.inputs.args = '-force'
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', mri_convert_node, 'in_file')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', mri_convert_node, 'in_bvec')
            dwi_workflow.connect(preproc_dwi_node, 'bval', mri_convert_node, 'in_bval')

            tckgen_node = Node(Tractography(), name='tckgen')
            os.makedirs(tckgen_outout_dir, exist_ok=True)
            dwi_workflow.connect(mri_convert_node, 'out_file', tckgen_node, 'in_file')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', tckgen_node, 'roi_mask')
            tckgen_node.inputs.seed_image = seed_mask
            tckgen_node.inputs.out_file = os.path.join(tckgen_outout_dir, 'tracked.tck')
            tckgen_node.inputs.algorithm = self.tckgen_method
            tckgen_node.inputs.select = 10000
            tckgen_node.inputs.args = '-force'

            dwi_workflow.connect(tckgen_node, 'out_file', seed_based_track_node, 'seed_based_track')
        else:
            seed_based_track_node.inputs.seed_based_track = os.path.join(tckgen_outout_dir, 'tracked.tck')

        ############
        # DTI-ALPS #
        ############
        if self.dtialps:
            # script: alps.sh
            dtialps_output_dir = os.path.join(self.output_path, 'DTI-ALPS')
            
            dtialps_node = Node(DTIALPSsimple(), name='dtialps')
            dtialps_node.inputs.perform_roi_analysis = '1'
            dtialps_node.inputs.use_templete = '1'
            dwi_workflow.connect(dti_fit_output_node, 'output_dir', dtialps_node, 'dtifit_output_dir')
            dtialps_node.inputs.alps_input_dir = dtialps_output_dir
            dtialps_node.inputs.skip_preprocessing = '1'
            dtialps_node.inputs.alps_script_path = self.alps_script_path
        
        ##########################
        # Single Shell Freewater #
        ##########################
        freewater_node = Node(IdentityInterface(fields=['fw_img']), name='freewater_node')

        if self.single_shell_freewater:
            single_shell_freewater_node = Node(SingleShellFW(), name='single_shell_freewater')

            dwi_workflow.connect(inputnode, 'bval_file', single_shell_freewater_node, 'fbval')

            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', single_shell_freewater_node, 'fdwi')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', single_shell_freewater_node, 'fbvec')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', single_shell_freewater_node, 'mask_file')

            fw_output_path = os.path.join(self.output_path, 'single_shell_freewater')
            single_shell_freewater_node.inputs.working_directory = fw_output_path
            single_shell_freewater_node.inputs.output_directory = fw_output_path
            single_shell_freewater_node.inputs.crop_shells = False

            dwi_workflow.connect(single_shell_freewater_node, 'output_fw', freewater_node, 'fw_img')
        else:
            freewater_node.inputs.fw_img = os.path.join(self.output_path, 'single_shell_freewater', 'freewater.nii.gz')
        
        ########
        # PSMD #
        ########
        if self.psmd:
            unzip_node = Node(UnzipCommandLine(), name="unzip_node")
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', unzip_node, 'file')
            unzip_node.inputs.decompress = True
            unzip_node.inputs.keep = True

            move_file_node = Node(MoveFileCommandLine(), name="move_file_node")
            dwi_workflow.connect(unzip_node, 'unzipped_file', move_file_node, 'source_file')
            move_file_node.inputs.destination_file = os.path.join(self.output_path, 'data.nii')

            psmd_node = Node(PSMDCommandLine(), name="psmd_node")
            dwi_workflow.connect(move_file_node, 'moved_file', psmd_node, 'dwi_data')
            dwi_workflow.connect(inputnode, 'bval_file', psmd_node, 'bval_file')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', psmd_node, 'bvec_file')
            psmd_node.inputs.mask_file = self.psmd_skeleton_mask
            if self.psmd_lesion_mask is not None and self.psmd_lesion_mask != '':
                psmd_lesion_mask_dir = self.session._find_output(self.psmd_lesion_mask)

                if self.use_which_psmd_lesion_mask is None:
                    self.use_which_psmd_lesion_mask = ''

                psmd_lesion_mask = [file for file in os.listdir(psmd_lesion_mask_dir) if f'{self.use_which_psmd_lesion_mask}' in file]
                if psmd_lesion_mask is not None and len(psmd_lesion_mask) == 1:
                    print(f"Using lesion mask: {psmd_lesion_mask[0]}.")
                    psmd_lesion_mask = os.path.join(psmd_lesion_mask_dir, psmd_lesion_mask[0])
                elif psmd_lesion_mask is not None and len(psmd_lesion_mask) > 1:
                    print(f"Using the first lesion mask found: {psmd_lesion_mask[0]}.")
                    psmd_lesion_mask = os.path.join(psmd_lesion_mask_dir, psmd_lesion_mask[0])
                else:
                    print("No lesion mask found.")

                psmd_node.inputs.lesion_mask = psmd_lesion_mask

            save_psmd_node = Node(SavePSMDOutputCommandLine(), name="save_psmd_node")
            save_psmd_node.inputs.output_file = os.path.join(self.output_path, 'psmd_output.csv')
            save_psmd_node.inputs.subject_id = self.subject.subject_id
            save_psmd_node.inputs.session_id = self.session.session_id
            dwi_workflow.connect(psmd_node, "psmd", save_psmd_node, "psmd")
            dwi_workflow.connect(psmd_node, "psmd_left", save_psmd_node, "psmd_left")
            dwi_workflow.connect(psmd_node, "psmd_right", save_psmd_node, "psmd_right")

            delete_dwi_file_node = Node(DeleteFileCommandLine(), name="delete_dwi_file_node")
            dwi_workflow.connect(psmd_node, "dwi", delete_dwi_file_node, "file")
        
        #########################
        # Calculate DWI metrics #
        #########################
        if self.calculate_dwi_metrics:
            dwi_metrics_output_dir = os.path.join(self.output_path, 'dwi_metrics_stats')
            os.makedirs(dwi_metrics_output_dir, exist_ok=True)

            dwi_metrics_node = Node(Merge(3), name='dwi_metrics')
            dwi_workflow.connect(dti_fit_output_node, 'fa_img', dwi_metrics_node, 'in1')
            dwi_workflow.connect(dti_fit_output_node, 'md_img', dwi_metrics_node, 'in2')
            dwi_workflow.connect(freewater_node, 'fw_img', dwi_metrics_node, 'in3')

            exist_dwi_metrics_node = Node(FilterExisting(), name='exist_dwi_metrics')
            dwi_workflow.connect(dwi_metrics_node, 'out', exist_dwi_metrics_node, 'input_file_list')

            # -------------------------------------- #
            # Calculate DWI metrics using .tck files #
            # -------------------------------------- #
            merge_csv_filename_node = Node(MergeFilename(), name='merge_csv_filename')
            dwi_workflow.connect(exist_dwi_metrics_node, 'filtered_filename_list', merge_csv_filename_node, 'filename_list')
            merge_csv_filename_node.inputs.dirname = dwi_metrics_output_dir
            merge_csv_filename_node.inputs.prefix = 'track_in_'
            merge_csv_filename_node.inputs.suffix = ''
            merge_csv_filename_node.inputs.extension = '.csv'

            tcksample_node = MapNode(TckSampleCommand(), name='tcksample', iterfield=['image', 'values'])
            tcksample_node.synchronize = True
            dwi_workflow.connect(seed_based_track_node, 'seed_based_track', tcksample_node, 'tracks')
            dwi_workflow.connect(merge_csv_filename_node, 'merge_file_list', tcksample_node, 'values')
            dwi_workflow.connect(exist_dwi_metrics_node, 'filtered_file_list', tcksample_node, 'image')
            tcksample_node.inputs.precise = True
            tcksample_node.inputs.stat_tck = 'mean'
            tcksample_node.inputs.args = '-force'

            get_csv_filename_node = Node(FilterExisting(), name='get_csv_filename')
            dwi_workflow.connect(tcksample_node, 'values', get_csv_filename_node, 'input_file_list')

            merge_mean_csv_filename_node = Node(MergeFilename(), name='merge_mean_csv_filename')
            dwi_workflow.connect(get_csv_filename_node, 'filtered_filename_list', merge_mean_csv_filename_node, 'filename_list')
            merge_mean_csv_filename_node.inputs.dirname = dwi_metrics_output_dir
            merge_mean_csv_filename_node.inputs.prefix = ''
            merge_mean_csv_filename_node.inputs.suffix = '_mean'
            merge_mean_csv_filename_node.inputs.extension = '.csv'

            calculate_mean_node = MapNode(CalculateMeanTckSample(), name='calculate_mean', iterfield=['csv_file', 'output_file'])
            calculate_mean_node.synchronize = True
            dwi_workflow.connect(tcksample_node, 'values', calculate_mean_node, 'csv_file')
            dwi_workflow.connect(merge_mean_csv_filename_node, 'merge_file_list', calculate_mean_node, 'output_file')

            # ------------------------------------------------ #
            # Calculate DWI metrics using binary .nii.gz masks #
            # ------------------------------------------------ #
            merge_exclude_masks_node = Node(Merge(2), name='merge_exclude_masks')

            if self.exclude_wmh_mask:
                wmh_output_dir = self.session._find_output('wmh_quantification')
                # search for the first file contain '_WMHmask' in dir
                wmh_mask = [file for file in os.listdir(wmh_output_dir) if '_WMHmask' in file]
                if wmh_mask is not None and len(wmh_mask) == 1:
                    logger.warning(f"Using WMH mask: {wmh_mask[0]}.")
                elif wmh_mask is not None and len(wmh_mask) > 1:
                    logger.warning(f"Using the first mask found: {wmh_mask[0]}.")
                else:
                    #logger.warning("No WMH mask found.")
                    raise FileNotFoundError(f"No WMH mask found in {wmh_output_dir}. Please check the directory.")
                wmh_mask_file = os.path.join(wmh_output_dir, wmh_mask[0])

                xfm_output_dir = self.session._find_output('xfm')
                # search for the first file contain '_from-FLAIR_to-T1w_xfm' in dir
                xfm_file = [file for file in os.listdir(xfm_output_dir) if '_from-FLAIR_to-T1w_xfm' in file]
                if xfm_file is not None and len(xfm_file) == 1:
                    logger.warning(f"Using xfm file: {xfm_file[0]}.")
                elif xfm_file is not None and len(xfm_file) > 1:
                    logger.warning(f"Using the first xfm file found: {xfm_file[0]}.")
                else:
                    #logger.warning("No xfm file found.")
                    raise FileNotFoundError(f"No xfm file found in {xfm_output_dir}. Please check the directory.")
                flair_to_t1w_xfm_file = os.path.join(xfm_output_dir, xfm_file[0])

                flair_to_dwi_xfm_entity = {'from': 'FLAIR', 'to': 'PreprocDWI'}
                flair_to_dwi_xfm_filename = os.path.join(xfm_output_dir, rename_bids_file(flair_to_t1w_xfm_file, flair_to_dwi_xfm_entity, 'xfm', '.mat'))

                concat_flair_to_dwi_xfm_node = Node(ConvertXFM(), name='concat_flair_to_dwi_xfm')
                concat_flair_to_dwi_xfm_node.inputs.in_file = flair_to_t1w_xfm_file
                dwi_workflow.connect(tractography_output_node, 't1w_to_dwi_xfm', concat_flair_to_dwi_xfm_node, 'in_file2')
                concat_flair_to_dwi_xfm_node.inputs.concat_xfm = True
                concat_flair_to_dwi_xfm_node.inputs.out_file = os.path.join(self.output_path, flair_to_dwi_xfm_filename)

                transform_wmh_to_dwi_node = Node(FLIRT(), name='transform_wmh_to_dwi')

                # if the WMH mask has different resolution than the FLAIR image, we need to resample it
                flair_img = nib.load(flair_file)
                wmh_mask_img = nib.load(wmh_mask_file)

                if flair_img.shape != wmh_mask_img.shape:
                    reslice_wmh_mask_node = Node(FLIRT(), name='reslice_wmh_mask')
                    reslice_wmh_mask_node.inputs.in_file = wmh_mask_file
                    reslice_wmh_mask_node.inputs.reference = flair_file
                    reslice_wmh_mask_node.inputs.args = '-applyxfm -usesqform'
                    reslice_wmh_mask_node.inputs.out_file = os.path.join(self.output_path, 'resliced_wmh_mask.nii.gz')
                    reslice_wmh_mask_node.inputs.interp = 'nearestneighbour'

                    dwi_workflow.connect(reslice_wmh_mask_node, 'out_file', transform_wmh_to_dwi_node, 'in_file')
                else:
                    transform_wmh_to_dwi_node.inputs.in_file = wmh_mask_file
                
                dwi_workflow.connect(concat_flair_to_dwi_xfm_node, 'out_file', transform_wmh_to_dwi_node, 'in_matrix_file')
                transform_wmh_to_dwi_node.inputs.in_file = wmh_mask_file
                dwi_workflow.connect(dti_fit_output_node, 'fa_img', transform_wmh_to_dwi_node, 'reference')
                transform_wmh_to_dwi_node.inputs.out_file = os.path.join(dwi_metrics_output_dir, 'wmh_mask_in_dwi.nii.gz')
                transform_wmh_to_dwi_node.inputs.interp = 'nearestneighbour'
                transform_wmh_to_dwi_node.inputs.apply_xfm = True

                dwi_workflow.connect(transform_wmh_to_dwi_node, 'out_file', merge_exclude_masks_node, 'in1')
            else:
                merge_exclude_masks_node.inputs.in1 = ''
            
            if self.exclude_seed_mask:               
                merge_exclude_masks_node.inputs.in2 = seed_mask
            else:
                merge_exclude_masks_node.inputs.in2 = ''
            
            make_wm_mask_node = Node(GenerateWMMaskCommandLine(), name='make_wm_mask')
            dwi_workflow.connect(dti_fit_output_node, 'fa_img', make_wm_mask_node, 'dwi_data')
            dwi_workflow.connect(tractography_output_node, 'fs_to_dwi_xfm', make_wm_mask_node, 'fs_to_dwi_xfm')
            make_wm_mask_node.inputs.output_dir = dwi_metrics_output_dir
            make_wm_mask_node.inputs.fs_output = fs_output
            dwi_workflow.connect(merge_exclude_masks_node, 'out', make_wm_mask_node, 'exclude_masks')

            merge_masks = Node(Merge(4), name='merge_masks')
            dwi_workflow.connect(make_wm_mask_node, 'wm_mask', merge_masks, 'in1')
            dwi_workflow.connect(transform_wmh_to_dwi_node, 'out_file', merge_masks, 'in2')
            merge_masks.inputs.in3 = seed_mask
            dwi_workflow.connect(make_wm_mask_node, 'wm_mask_raw', merge_masks, 'in4')

            exist_masks_node = Node(FilterExisting(), name='exist_masks')
            dwi_workflow.connect(merge_masks, 'out', exist_masks_node, 'input_file_list')

            calculate_dwi_metrics_node = MapNode(CalculateScalarMaps(), name='calculate_dwi_metrics', iterfield=['wm_mask'])
            #calculate_dwi_metrics_node.synchronize = False
            calculate_dwi_metrics_node.inputs.output_dir = dwi_metrics_output_dir
            dwi_workflow.connect(exist_dwi_metrics_node, 'filtered_file_list', calculate_dwi_metrics_node, 'dwi_data')
            dwi_workflow.connect(exist_masks_node, 'filtered_file_list', calculate_dwi_metrics_node, 'wm_mask')
            
        return dwi_workflow
    
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        fdt_output_path = self.extract_from

        # DTI-ALPS
        columns = ['Subject', 'Session', 'ALPS_L', 'ALPS_R', 'ALPS_mean']
        results_df = pd.DataFrame(columns=columns)

        for subject_folder in os.listdir(fdt_output_path):
            subject_id = subject_folder.split('-')[1]
            subject_folder_path = os.path.join(fdt_output_path, subject_folder)

            if os.path.isdir(subject_folder_path):
                session_folders = [f for f in os.listdir(subject_folder_path) if 'ses-' in f]

                if session_folders:
                    for session_folder in session_folders:
                        session_id = session_folder.split('-')[1]
                        session_path = os.path.join(subject_folder_path, session_folder)

                        dti_alps_stat = os.path.join(session_path, 'DTI-ALPS', 'alps.stat', 'alps.csv')
                        if os.path.exists(dti_alps_stat):
                            df = pd.read_csv(dti_alps_stat)
                            if {'alps_L', 'alps_R', 'alps'}.issubset(df.columns):
                                new_data = pd.DataFrame([{
                                    'Subject': subject_id,
                                    'Session': session_id,
                                    'ALPS_L': df['alps_L'].values[0],
                                    'ALPS_R': df['alps_R'].values[0],
                                    'ALPS_mean': df['alps'].values[0]
                                }])
                                results_df = pd.concat([results_df, new_data], ignore_index=True)
                else:
                    dti_alps_stat = os.path.join(subject_folder_path, 'DTI-ALPS', 'alps.stat', 'alps.csv')

                    if os.path.exists(dti_alps_stat):
                        df = pd.read_csv(dti_alps_stat)
                        if {'alps_L', 'alps_R', 'alps'}.issubset(df.columns):
                            new_data = pd.DataFrame([{
                                'Subject': subject_id,
                                'Session': 'N/A',
                                'ALPS_L': df['alps_L'].values[0],
                                'ALPS_R': df['alps_R'].values[0],
                                'ALPS_mean': df['alps'].values[0]
                            }])
                            results_df = pd.concat([results_df, new_data], ignore_index=True)

        output_excel_path = os.path.join(self.output_path, 'alps_results.xlsx')
        results_df.to_excel(output_excel_path, header=True, index=False)
        print(f"Quantification results saved to {output_excel_path}")

        # Lesion-based probabilistic tractography
        # TODO: situation when there are no sessions
        surface_parameters_df = pd.DataFrame()
        mirror_surface_parameters_df = pd.DataFrame()

        for subject_folder in os.listdir(fdt_output_path):
            subject_id = subject_folder.split('-')[1]
            subject_folder_path = os.path.join(fdt_output_path, subject_folder)

            if os.path.isdir(subject_folder_path):
                session_folders = [f for f in os.listdir(subject_folder_path) if 'ses-' in f]

                if session_folders:
                    for session_folder in session_folders:
                        session_id = session_folder.split('-')[1]
                        session_path = os.path.join(subject_folder_path, session_folder)

                        # surface_parameters.csv
                        surface_csv = os.path.join(session_path, "surface_parameters.csv")
                        if os.path.exists(surface_csv):
                            df = pd.read_csv(surface_csv)
                            df.insert(0, 'fdt_id', f'sub-{subject_id}_ses-{session_id}')
                            df.insert(1, 'subject', f'sub-{subject_id}')
                            surface_parameters_df = pd.concat([surface_parameters_df, df], ignore_index=True)

                        # mirror_surface_parameters.csv
                        mirror_csv = os.path.join(session_path, "mirror_surface_parameters.csv")
                        if os.path.exists(mirror_csv):
                            df_mirror = pd.read_csv(mirror_csv)
                            df_mirror.insert(0, 'fdt_id', f'sub-{subject_id}_ses-{session_id}')
                            df_mirror.insert(1, 'subject', f'sub-{subject_id}')
                            mirror_surface_parameters_df = pd.concat([mirror_surface_parameters_df, df_mirror], ignore_index=True)

        # Save results
        surface_output_excel = os.path.join(self.output_path, 'surface_parameters_results.xlsx')
        mirror_output_excel = os.path.join(self.output_path, 'mirror_surface_parameters_results.xlsx')

        if not surface_parameters_df.empty:
            surface_parameters_df.to_excel(surface_output_excel, header=True, index=False)
            print(f"Surface parameters results saved to {surface_output_excel}")
        else:
            print("No surface parameters found.")

        if not mirror_surface_parameters_df.empty:
            mirror_surface_parameters_df.to_excel(mirror_output_excel, header=True, index=False)
            print(f"Mirror surface parameters results saved to {mirror_output_excel}")
        else:
            print("No mirror surface parameters found.")

