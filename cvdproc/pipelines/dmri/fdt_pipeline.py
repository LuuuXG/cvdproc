import os
import subprocess
import pandas as pd
import nibabel as nib
import numpy as np
import json

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from .fdt.fdt_nipype import DTIpreprocessing, Bedpostx, Tractography, ExtractSurfaceParameters, DTIALPSsimple
from .freewater.single_shell_freewater import SingleShellFW
from ..smri.mirror.mirror_nipype import MirrorMask
from nipype.interfaces.fsl import FLIRT
from ..smri.fsl.fsl_anat_nipype import FSLANAT

class FDTPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        FDT pipeline: Diffusion Tensor Imaging (DTI) analysis
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.seed_mask = kwargs.get('seed_mask', 'lesion_mask') # seed mask for probabilistic tractography
        self.use_which_mask = kwargs.get('use_which_mask', None) # which mask to use (in the processed DTI space)
        self.use_which_dwi = kwargs.get('use_which_dwi', None) # which dwi file to use
        self.use_which_t1w = kwargs.get('use_which_t1w', None) # which t1w file to use
        self.preprocess = kwargs.get('preprocess', False) # whether to preprocess DTI
        self.synb0 = kwargs.get('synb0', False) # whether to use synthetic b0 image
        self.tractography = kwargs.get('tractography', False)
        self.dtialps = kwargs.get('dtialps', False)
        self.single_shell_freewater = kwargs.get('single_shell_freewater', False)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.script_path1 = os.path.join(base_dir, 'bash', 'fdt_dti.sh') # DTI processing, fitting, and Bedpostx
        self.script_path2 = os.path.join(base_dir, 'bash', 'fdt_fs_processing.sh') # FreeSurfer processing
        self.alps_script_path = os.path.join(base_dir, 'external', 'alps', 'alps.sh') # DTI-ALPS

        # extract results
        self.extract_from = kwargs.get('extract_from', None)

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

            phase_encoding_direction_dict = {
                "i": "-1 0 0",
                "i-": "1 0 0",
                "j": "0 -1 0",
                "j-": "0 1 0"
            }       

        except KeyError:
            print("TotalReadoutTime or PhaseEncodingDirection not found in the json file.")
            if not total_readout_time:
                total_readout_time = '0.05'
                print("Set TotalReadoutTime to 0.05. But make sure the value is the same in all acquisitions.")

        # Find Reverse b0 Image
        fmap_files = self.session.get_fmap_files()
        reverse_b0 = next((file['path'] for file in fmap_files if file['type'] == 'reverse_b0'), None)
        reverse_b0_flag = reverse_b0 is not None
        if reverse_b0 is None:
            reverse_b0 = 'none'

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
        
        ################
        # fdt_workflow #
        ################
        fdt_workflow = Workflow(name='fdt_workflow')
        fdt_workflow.base_dir = self.output_path

        inputnode = Node(IdentityInterface(fields=['bids_dir', 't1w_file',
                                                   'dwi_file', 'bval_file',
                                                   'bvec_file', 'json_file',
                                                   'use_synb0', 'output_path_synb0',
                                                   'output_path', 'phase_encoding_number',
                                                   'total_readout_time', 'script_path',
                                                   'fs_output', 'seed_mask_dtispace',
                                                   'fs_processing_dir', 'output_path_dtialps']), name="inputnode")
        
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
        # 1. DTI processing, fitting, and bedpostX #
        ############################################
        if self.preprocess:
            dtipreprocessing_node = Node(DTIpreprocessing(), name='dtipreprocessing')
            fdt_workflow.connect(inputnode, 'bids_dir', dtipreprocessing_node, 'bids_dir')
            fdt_workflow.connect(inputnode, 't1w_file', dtipreprocessing_node, 't1w_file')
            fdt_workflow.connect(inputnode, 'dwi_file', dtipreprocessing_node, 'dwi_file')
            fdt_workflow.connect(inputnode, 'bval_file', dtipreprocessing_node, 'bval_file')
            fdt_workflow.connect(inputnode, 'bvec_file', dtipreprocessing_node, 'bvec_file')
            fdt_workflow.connect(inputnode, 'json_file', dtipreprocessing_node, 'json_file')
            fdt_workflow.connect(inputnode, 'use_synb0', dtipreprocessing_node, 'use_synb0')
            fdt_workflow.connect(inputnode, 'output_path_synb0', dtipreprocessing_node, 'output_path_synb0')
            fdt_workflow.connect(inputnode, 'output_path', dtipreprocessing_node, 'output_path')
            fdt_workflow.connect(inputnode, 'phase_encoding_number', dtipreprocessing_node, 'phase_encoding_number')
            fdt_workflow.connect(inputnode, 'total_readout_time', dtipreprocessing_node, 'total_readout_time')
            dtipreprocessing_node.inputs.script_path_dtipreprocess = self.script_path1

            bedpostx_node = Node(Bedpostx(), name='bedpostx')
            fdt_workflow.connect(dtipreprocessing_node, 'bedpostx_input_dir', bedpostx_node, 'input_dir')

        #########################################################################
        # 2. Tractography: Designed for lesion-based probabilistic tractography #
        #########################################################################
        if self.tractography:
            fs_output = self.session.freesurfer_dir
            # decide to use which mask as seed
            if self.seed_mask == 'lesion_mask':
                lesion_mask_dir = self.session.lesion_mask_dir
            else:
                # TODO
                print("No other mask available now.")
                
            seed_mask = [file for file in os.listdir(lesion_mask_dir) if f'{self.use_which_mask}' in file]
            if seed_mask is not None and len(seed_mask) == 1:
                print(f"Using seed mask: {seed_mask[0]}.")
                seed_mask = os.path.join(lesion_mask_dir, seed_mask[0])
            elif seed_mask is not None and len(seed_mask) > 1:
                print(f"Using the first mask found: {seed_mask[0]}.")
                seed_mask = os.path.join(lesion_mask_dir, seed_mask[0])
                print(f"Using seed mask: {seed_mask}.")
            else:
                print("No seed mask found.")
                return False

            fs_processing_dir = os.path.join(self.output_path, 'fs_processing')
            os.makedirs(fs_processing_dir, exist_ok=True)

            probtrackx_output_dir1 = os.path.join(self.output_path, 'probtrackx_output')
            os.makedirs(probtrackx_output_dir1, exist_ok=True)
            probtrackx_output_dir2 = os.path.join(self.output_path, 'probtrackx_output_PathLengthCorrected')
            os.makedirs(probtrackx_output_dir2, exist_ok=True)

            tractography_node = Node(Tractography(), name='tractography')
            tractography_node.inputs.fs_output = fs_output
            tractography_node.inputs.seed_mask_dtispace = seed_mask
            tractography_node.inputs.fs_processing_dir = fs_processing_dir
            fdt_workflow.connect(inputnode, 't1w_file', tractography_node, 't1w_file')
            #fdt_workflow.connect(dtipreprocessing_node, 'fa_file', tractography_node, 'fa_file')
            tractography_node.inputs.fa_file = os.path.join(self.output_path, 'dti_FA.nii.gz')
            tractography_node.inputs.script_path_fspreprocess = self.script_path2
            if self.preprocess:
                fdt_workflow.connect(bedpostx_node, 'output_dir', tractography_node, 'bedpostx_output_dir')
            else:
                tractography_node.inputs.bedpostx_output_dir = os.path.join(self.output_path, 'bedpostX_input.bedpostX')
            tractography_node.inputs.probtrackx_output_dir1 = probtrackx_output_dir1
            tractography_node.inputs.probtrackx_output_dir2 = probtrackx_output_dir2

            extract_parameters_node = Node(ExtractSurfaceParameters(), name='extract_parameters')
            fdt_workflow.connect(tractography_node, 'lh_unconn_mask', extract_parameters_node, 'lh_unconn_mask')
            fdt_workflow.connect(tractography_node, 'rh_unconn_mask', extract_parameters_node, 'rh_unconn_mask')
            fdt_workflow.connect(tractography_node, 'lh_unconn_corrected_mask', extract_parameters_node, 'lh_unconn_corrected_mask')
            fdt_workflow.connect(tractography_node, 'rh_unconn_corrected_mask', extract_parameters_node, 'rh_unconn_corrected_mask')
            fdt_workflow.connect(tractography_node, 'lh_low_conn_mask', extract_parameters_node, 'lh_low_conn_mask')
            fdt_workflow.connect(tractography_node, 'rh_low_conn_mask', extract_parameters_node, 'rh_low_conn_mask')
            fdt_workflow.connect(tractography_node, 'lh_low_conn_corrected_mask', extract_parameters_node, 'lh_low_conn_corrected_mask')
            fdt_workflow.connect(tractography_node, 'rh_low_conn_corrected_mask', extract_parameters_node, 'rh_low_conn_corrected_mask')
            fdt_workflow.connect(tractography_node, 'lh_medium_conn_mask', extract_parameters_node, 'lh_medium_conn_mask')
            fdt_workflow.connect(tractography_node, 'rh_medium_conn_mask', extract_parameters_node, 'rh_medium_conn_mask')
            fdt_workflow.connect(tractography_node, 'lh_medium_conn_corrected_mask', extract_parameters_node, 'lh_medium_conn_corrected_mask')
            fdt_workflow.connect(tractography_node, 'rh_medium_conn_corrected_mask', extract_parameters_node, 'rh_medium_conn_corrected_mask')
            fdt_workflow.connect(tractography_node, 'lh_high_conn_mask', extract_parameters_node, 'lh_high_conn_mask')
            fdt_workflow.connect(tractography_node, 'rh_high_conn_mask', extract_parameters_node, 'rh_high_conn_mask')
            fdt_workflow.connect(tractography_node, 'lh_high_conn_corrected_mask', extract_parameters_node, 'lh_high_conn_corrected_mask')
            fdt_workflow.connect(tractography_node, 'rh_high_conn_corrected_mask', extract_parameters_node, 'rh_high_conn_corrected_mask')
            extract_parameters_node.inputs.output_dir = self.output_path
            extract_parameters_node.inputs.fs_subjects_dir = os.path.dirname(fs_output)
            extract_parameters_node.inputs.sessions = self.subject.sessions_id
            extract_parameters_node.inputs.csv_file_name = 'surface_parameters.csv'

            # experimental: mirror mask
            fsl_anat_node = Node(FSLANAT(), name='fsl_anat')
            fdt_workflow.connect(inputnode, 't1w_file', fsl_anat_node, 'input_image')
            fsl_anat_node.inputs.output_directory = os.path.join(self.subject.bids_dir, 'derivatives', 'fsl_anat', f'sub-{self.session.subject_id}', f"ses-{self.session.session_id}", 'fsl')
            os.makedirs(os.path.dirname(fsl_anat_node.inputs.output_directory), exist_ok=True)

            mirror_mask_node = Node(MirrorMask(), name='mirror_mask')
            mirror_mask_node.inputs.in_file = os.path.join(fs_processing_dir, 'seed_mask_in_t1w.nii.gz')
            fdt_workflow.connect(inputnode, 't1w_file', mirror_mask_node, 't1w_file')
            mirror_mask_node.inputs.out_dir = fs_processing_dir
            # mirror_mask_node.inputs.t1w_to_mni_xfm = os.path.join(self.session.fsl_anat_dir, 'fsl.anat', 'T1_to_MNI_nonlin_field.nii.gz')
            # mirror_mask_node.inputs.mni_to_t1w_xfm = os.path.join(self.session.fsl_anat_dir, 'fsl.anat', 'MNI_to_T1_nonlin_field.nii.gz')
            fdt_workflow.connect(fsl_anat_node, 'output_directory', mirror_mask_node, 'fsl_anat_output_dir')

            mirror_mask_node.inputs.mask_in_mni_filename = 'seed_mask_in_mni.nii.gz'
            mirror_mask_node.inputs.flipped_mask_mni_filename = 'seed_mask_in_mni_flipped.nii.gz'
            mirror_mask_node.inputs.flipped_mask_t1w_filename = 'seed_mask_in_t1w_flipped.nii.gz'

            mirror_mask_t1w_to_fs_node = Node(FLIRT(), name='mirror_mask_t1w_to_fs')
            fdt_workflow.connect(mirror_mask_node, 'flipped_mask_t1w', mirror_mask_t1w_to_fs_node, 'in_file')
            t1w_to_fs_xfm = os.path.join(fs_processing_dir, 'struct2freesurfer.mat')
            mirror_mask_t1w_to_fs_node.inputs.in_matrix_file = t1w_to_fs_xfm
            fdt_workflow.connect(tractography_node, 'fs_orig', mirror_mask_t1w_to_fs_node, 'reference')
            mirror_mask_t1w_to_fs_node.inputs.out_file = os.path.join(fs_processing_dir, 'seed_mask_in_fs_flipped.nii.gz')
            mirror_mask_t1w_to_fs_node.inputs.apply_xfm = True
            mirror_mask_t1w_to_fs_node.inputs.interp = 'nearestneighbour'

            mirror_tractography_node = Node(Tractography(), name='mirror_tractography')
            mirror_tractography_node.inputs.fs_output = fs_output
            fdt_workflow.connect(tractography_node, 'fs_processing_dir', mirror_tractography_node, 'fs_processing_dir')
            fdt_workflow.connect(inputnode, 't1w_file', mirror_tractography_node, 't1w_file')
            mirror_tractography_node.inputs.skip_fs_preprocess = True
            fdt_workflow.connect(mirror_mask_t1w_to_fs_node, 'out_file', mirror_tractography_node, 'seed_mask_fsspace')
            mirror_tractography_node.inputs.bedpostx_output_dir = os.path.join(self.output_path, 'bedpostX_input.bedpostX')
            mirror_tractography_node.inputs.probtrackx_output_dir1 = os.path.join(self.output_path, 'mirror_probtrackx_output')
            mirror_tractography_node.inputs.probtrackx_output_dir2 = os.path.join(self.output_path, 'mirror_probtrackx_output_PathLengthCorrected')

            mirror_extract_parameters_node = Node(ExtractSurfaceParameters(), name='mirror_extract_parameters')
            fdt_workflow.connect([
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

        ############
        # DTI-ALPS #
        ############
        if self.dtialps:
            # script: alps.sh
            dtialps_output_dir = os.path.join(self.output_path, 'DTI-ALPS')

            inputnode.inputs.output_path_dtialps = dtialps_output_dir
            
            dtialps_node = Node(DTIALPSsimple(), name='dtialps')
            dtialps_node.inputs.perform_roi_analysis = '1'
            dtialps_node.inputs.use_templete = '1'
            fdt_workflow.connect(inputnode, 'output_path', dtialps_node, 'dtifit_output_dir')
            fdt_workflow.connect(inputnode, 'output_path_dtialps', dtialps_node, 'alps_input_dir')
            dtialps_node.inputs.skip_preprocessing = '1'
            dtialps_node.inputs.alps_script_path = self.alps_script_path
        
        ##########################
        # Single Shell Freewater #
        ##########################
        if self.single_shell_freewater:
            single_shell_freewater_node = Node(SingleShellFW(), name='single_shell_freewater')

            fdt_workflow.connect(inputnode, 'bval_file', single_shell_freewater_node, 'fbval')

            eddy_correct_dwi = os.path.join(self.output_path, 'eddy_corrected_data.nii.gz')
            eddy_correct_bvec = os.path.join(self.output_path, 'eddy_corrected_data.eddy_rotated_bvecs')
            dwi_mask = os.path.join(self.output_path, 'dwi_b0_brain_mask.nii.gz')
            fw_output_path = os.path.join(self.output_path, 'single_shell_freewater')

            single_shell_freewater_node.inputs.fdwi = eddy_correct_dwi
            single_shell_freewater_node.inputs.fbvec = eddy_correct_bvec
            single_shell_freewater_node.inputs.mask_file = dwi_mask
            single_shell_freewater_node.inputs.working_directory = fw_output_path
            single_shell_freewater_node.inputs.output_directory = fw_output_path
            single_shell_freewater_node.inputs.crop_shells = False
            
        return fdt_workflow
    
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

