import os
import pandas as pd
import json
import nibabel as nib
import numpy as np
from cvdproc.config.paths import get_package_path
from nipype import Node, Workflow, MapNode, Function
from nipype.interfaces.utility import IdentityInterface, Merge, Select
from cvdproc.bids_data.rename_bids_file import rename_bids_file

# nipype built-in interfaces
from nipype.interfaces.fsl import FLIRT, ExtractROI, ConvertXFM, MultiImageMaths
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.mrtrix3 import Tractography, MRConvert, DWIPreproc, BuildConnectome, LabelConvert

# custom interfaces
from cvdproc.pipelines.common.files import FilterExisting, MergeFilename
from cvdproc.pipelines.common.pad_dwi import PadDWI
from cvdproc.pipelines.common.flip_b_table import FlipBTable
from cvdproc.pipelines.common.register import Tkregister2fs2t1w, MRIConvertApplyWarp, MRIVol2Vol

from cvdproc.pipelines.dmri.fdt.fdt_nipype import B0AllAndAcqparam, IndexTxt, Topup, EddyCuda, OrderEddyOutputs, B0RefAndBrainMask, DTIFitBIDS, BedpostxGPUCustom, Probtrackx, ReadSingleValue, ExtractSurfaceParameters, ApplyGiiMaskToMgh, ApplyIndexTxtToMgh, DefineConnectionLevel
from cvdproc.pipelines.dmri.synb0.synb0_nipype import Synb0
from cvdproc.pipelines.dmri.psmd.psmd_nipype import PSMDCommandLine
from cvdproc.pipelines.dmri.alps.alps_nipype import ALPS
from cvdproc.pipelines.dmri.dsistudio.create_src_nipype import CreateSRC
from cvdproc.pipelines.dmri.dsistudio.reconstruction_nipype import DSIstudioReconstruction
from cvdproc.pipelines.dmri.dsistudio.tracking_nipype import DSIstudioTracking
from cvdproc.pipelines.dmri.dsistudio.tract_analysis import TractStatsInterface
from cvdproc.pipelines.dmri.pved.pved_nipype import PVeD
from cvdproc.pipelines.dmri.mrtrix3.tcksample_nipype import TckSampleCommand, CalculateMeanTckSample
from cvdproc.pipelines.dmri.mrtrix3.denoise_degibbs_nipype import MrtrixDenoise, MrtrixDegibbs
from cvdproc.pipelines.dmri.mrtrix3.connectome_nipype import RemoveWMH, ConnectomePrepare
from cvdproc.pipelines.dmri.mrtrix3.utils import RemoveTractRegion
from cvdproc.pipelines.dmri.stats.dti_scalar_maps import GenerateNAWMMask, CalculateScalarMaps, CalculateTDIWeightedScalars, TckSampleMultiScalarProfile, TckSampleMultiScalarBundle
from cvdproc.pipelines.dmri.dipy.dipy_freewater_dti import FreeWaterTensor
from cvdproc.pipelines.dmri.dipy.dipy_degibbs import DipyDegibbs
from cvdproc.pipelines.dmri.freewater.single_shell_freewater import SingleShellFW
from cvdproc.pipelines.dmri.freewater.markvcid_freewater import MarkVCIDFreeWater
from cvdproc.pipelines.dmri.qsiprep.register import QSIPrepOrigToACPC

from cvdproc.pipelines.smri.mirror.mirror_nipype import MirrorMask
from cvdproc.pipelines.smri.fsl.fsl_anat_nipype import FSLANAT
from cvdproc.pipelines.smri.freesurfer.synthstrip import SynthStrip
from cvdproc.pipelines.smri.freesurfer.utils import MRIBinarize, MRIvol2surf, MergeRibbon

class DWIPipeline:
    def __init__(self, 
                 subject: object, 
                 session: object, 
                 output_path: str, 
                 use_which_dwi: str = None,
                 use_which_t1w: str = None,
                 use_which_flair: str = None,
                 use_freesurfer_longitudinal: bool = False,
                 preprocess: bool = False,
                 output_resolution: float = 2.0,
                 degibbs: bool = True,
                 flip_b_table_axis: list = [],
                 preprocess_method: str = 'fdt',
                 synb0: bool = False,
                 use_which_reverse_b0: str = None,
                 dti_fit: bool = False,
                 dwi_t1w_register: bool = False,
                 dsistudio_gqi: bool = False,
                 dsistudio_qsdr: bool = False,
                 amico_noddi: bool = False,
                 connectome: list = [],
                 tractography: list = [],
                 seed_mask: str = 'lesion_mask',
                 use_which_mask: str = None,
                 dtialps: bool = False,
                 dtialps_register_method: int = 1,
                 pved: bool = False,
                 freewater: list = [],
                 psmd: bool = False,
                 psmd_exclude_seed_mask: bool = False,
                 visual_pathway_analysis: bool = False,
                 calculate_dwi_metrics: bool = False,
                 exclude_seed_mask: bool = True,
                 exclude_wmh_mask: bool = False,
                 extract_from: str = None,
                 **kwargs):
        """
        DWI pipeline

        Args:
            subject (object): Subject object
            session (object): Session object
            output_path (str): Output path
            use_which_dwi (str, optional): Which DWI file to use. Defaults to None.
            use_which_t1w (str, optional): Which T1w file to use. Defaults to None.
            use_which_flair (str, optional): Which FLAIR file to use. Defaults to None.
            use_freesurfer_longitudinal (bool, optional): Whether to use FreeSurfer longitudinal processing outpus. Defaults to False.
            preprocess (bool, optional): Whether to preprocess the DWI data. Defaults to False.
            output_resolution (float, optional): Output resolution in mm. Defaults to 2.0.
            degibbs (bool, optional): Whether to perform degibbsing. Defaults to True.
            flip_b_table_axis (list, optional): List of axes to flip the b-table. Defaults to []. [0], [1], [2] for x, y, z axes.
            preprocess_method (str, optional): Preprocessing method. 'fdt', 'mrtrix3' or 'post_qsiprep'. Defaults to 'fdt'.
            synb0 (bool, optional): Whether to use Synb0 for TOPUP. Defaults to False.
            use_which_reverse_b0 (str, optional): Which reverse coded b0 image to use. Defaults to None.
            dti_fit (bool, optional): Whether to fit DTI model using FSL dtifit. Defaults to True.
            dwi_t1w_register (bool, optional): Whether to register DWI to T1w space. Defaults to False.
            dsistudio_gqi (bool, optional): Whether to fit GQI model using DSI Studio. Defaults to False.
            dsistudio_qsdr (bool, optional): Whether to fit QSDR model using DSI Studio. Defaults to False.
            amico_noddi (bool, optional): Whether to fit NODDI model using AMICO. Defaults to False.
            connectome (list, optional): Whether to create a connectome (Global). Defaults to []. Subsets of ['fdt', 'mrtrix3', 'dsistudio'].
            tractography (list, optional): Tractography method (ROI-based). Defaults to []. Subsets of ['fdt', 'mrtrix3', 'dsistudio'].
            seed_mask (str, optional): Folder name for seed mask. Expected to find <bids_dir>/derivatives/<seed_mask>/sub-<id>/ses-<id>/*<use_which_mask>*.nii.gz. Defaults to 'lesion_mask'. Must be in the T1w space.
            use_which_mask (str, optional): Which mask file to use (in the processed DWI space). Defaults to None.
            dtialps (bool, optional): Whether to perform DTI-ALPS analysis. Defaults to False.
            dtialps_register_method (int, optional): Registration method for DTI-ALPS. 1 for FA + FLIRT, 2 for FA + synthmorph.
            pved (bool, optional): Whether to perform PVeD analysis. Defaults to False.
            freewater (list, optional): List of freewater estimation methods. Defaults to []. Subsets of ['single_shell_freewater', 'markvcid_freewater', 'dti_freewater'].
            psmd (bool, optional): Whether to perform PSMD analysis. Defaults to False.
            psmd_exclude_seed_mask (bool, optional): Whether to exclude seed mask in PSMD calculation. Defaults to False.
            visual_pathway_analysis (bool, optional): Whether to perform visual pathway analysis. Defaults to False. (An ongoing project! Experimental feature.)
            calculate_dwi_metrics (bool, optional): Whether to calculate DWI scalar maps. Defaults to False.
            exclude_seed_mask (bool, optional): Whether to exclude seed mask in DWI metrics calculation (For NAWM). Defaults to True.
            exclude_wmh_mask (bool, optional): Whether to exclude WMH mask in DWI metrics calculation (For NAWM). Defaults to False.
            extract_from (str, optional): Folder name to extract results from.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_dwi = use_which_dwi # which dwi file to use
        self.use_which_t1w = use_which_t1w # which t1w file to use
        self.use_which_flair = use_which_flair # which flair file to use
        self.use_freesurfer_longitudinal = use_freesurfer_longitudinal # whether to use FreeSurfer longitudinal processing outputs

        # Preprocessing
        self.preprocess = preprocess # whether to preprocess
        self.output_resolution = output_resolution # output resolution in mm
        self.degibbs = degibbs # whether to perform degibbsing
        self.flip_b_table_axis = flip_b_table_axis # axes to flip the b-table
        self.preprocess_method = preprocess_method # preprocessing method
        self.synb0 = synb0 # whether to use synthetic b0 image for TOPUP
        self.use_which_reverse_b0 = use_which_reverse_b0 # whether to use reverse b0 image for TOPUP. if synb0 is True, this will be ignored

        # Renconstruction
        self.dti_fit = dti_fit # whether to fit DTI model using FSL FDT
        self.dwi_t1w_register = dwi_t1w_register # whether to register DWI to T1w space
        self.dsistudio_gqi = dsistudio_gqi # whether to fit GQI model using DSI Studio
        self.dsistudio_qsdr = dsistudio_qsdr # whether to fit QSDR model using DSI Studio
        self.amico_noddi = amico_noddi # whether to fit NODDI model using AMICO

        # Global connectome and connectivity analysis
        self.connectome = connectome # whether to create a connectome

        # ROI-based tractography
        self.tractography = tractography
        self.seed_mask = seed_mask # seed mask for probabilistic tractography
        self.use_which_mask = use_which_mask # which mask to use (in the processed DTI space)

        # DTI-ALPS
        self.dtialps = dtialps
        self.dtialps_register_method = dtialps_register_method # 1 for FA + FLIRT, 2 for T1 + FNIRT, 3 for T1 + Synthmorph

        # PVeD
        self.pved = pved

        # Freewater
        self.freewater = freewater

        # PSMD
        self.psmd = psmd
        self.psmd_exclude_seed_mask = psmd_exclude_seed_mask

        # Visual Pathway Analysis (an ongoing project!)
        self.visual_pathway_analysis = visual_pathway_analysis

        # Calculate scalar maps
        self.calculate_dwi_metrics = calculate_dwi_metrics # whether to calculate DWI metrics
        self.exclude_seed_mask = exclude_seed_mask # whether to exclude seed mask in DWI metrics calculation
        self.exclude_wmh_mask = exclude_wmh_mask # whether to exclude WMH mask in DWI metrics calculation

        # extract results
        self.extract_from = extract_from

        #### Not user configurable ####
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.script_path2 = os.path.join(base_dir, 'bash', 'fdt_fs_processing.sh') # FreeSurfer processing
        self.psmd_skeleton_mask = os.path.join(base_dir, 'external', 'psmd', 'skeleton_mask_2019.nii.gz')

    def check_data_requirements(self):
        """
        check if the required data is available
        :return: bool
        """
        return self.session.get_dwi_files() is not None and self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # Find Image Files
        os.makedirs(self.output_path, exist_ok=True)

        dwi_files = self.session.get_dwi_files()
        phase_encoding_direction_dict = {
            "i": "1 0 0",
            "i-": "-1 0 0",
            "j": "0 1 0",
            "j-": "0 -1 0",
            "k": "0 0 1",
            "k-": "0 0 -1"
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
        print(f"[DWI Pipeline] Using DWI files: {dwi_files_dict}")

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
            print("[Warn] Seems TotalReadoutTime or PhaseEncodingDirection not found in the json file.")
            if not total_readout_time:
                total_readout_time = '0.05'
                print("[Warn] Set TotalReadoutTime to 0.05. But make sure the value is the same in all acquisitions.")

        # Find Reverse b0 Image
        if not self.synb0 and self.use_which_reverse_b0 is not None:
            # TODO: Use the 'IntendedFor' field in the json file to find the reverse b0 image
            fmap_files = self.session.get_fmap_files()
            reverse_b0_files = [file for file in fmap_files if self.use_which_reverse_b0 in file['path']]
            
            if len(reverse_b0_files) != 4:
                raise FileNotFoundError(f"No specific reverse b0 file found for {self.use_which_reverse_b0} or more than one found.")
            
            reverse_b0_files_dict = {file['type']: file['path'] for file in reverse_b0_files}
            print(f"Using reverse b0 files: {reverse_b0_files_dict}")

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
                print("Seems TotalReadoutTime or PhaseEncodingDirection not found in the json file.")
                if not reverse_b0_total_readout_time:
                    reverse_b0_total_readout_time = '0.05'
                    print("Set TotalReadoutTime to 0.05. But make sure the value is the same in all acquisitions.")
        else:
            reverse_b0_image = None

        # get T1w image
        t1w_files = self.session.get_t1w_files()
        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
            print(f"[DWI Pipeline] Using T1w: {t1w_file}")
        else:
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]
            print(f"No specific T1w file selected. Using the first one: {t1w_file}.")
        
        # get FLAIR image
        flair_files = self.session.get_flair_files()
        if self.use_which_flair:
            flair_files = [f for f in flair_files if self.use_which_flair in f]
            if len(flair_files) != 1:
                #raise FileNotFoundError(f"No specific FLAIR file found for {self.use_which_flair} or more than one found.")
                print(f"No specific FLAIR file found for {self.use_which_flair} or more than one found.")
            flair_file = flair_files[0]
            print(f"[DWI Pipeline] Using FLAIR: {flair_file}")
        else:
            # flair_files = [flair_files[0]]
            # flair_file = flair_files[0]
            print(f"No specific FLAIR file selected. If you want to include a FLAIR file, please specify it.")

        fs_output = self.session.freesurfer_dir
        # if fs_output exists
        if fs_output is None:
            fs_subjects_dir = ''
            fs_subject_id = ''
            fs_output_process = False
            print("[DWI Pipeline] No FreeSurfer output found. Skipping related processing.")
        else:
            fs_subjects_dir = os.path.dirname(fs_output)
            fs_subject_id = os.path.basename(fs_output)
            # automatically do related anat preprocessing
            fs_output_process = True
            print(f"[DWI Pipeline] FreeSurfer output found: {fs_output}. Will do related processing.")

            # if use longitudinal outputs, check if the longitudinal output exists (fs_subjects_dir/ses-<session_id>.long.sub-<subject_id>)
            if self.use_freesurfer_longitudinal:
                session_ids = self.subject.sessions_id
                fs_long_output_dir = os.path.join(fs_subjects_dir, f'ses-{self.session.session_id}.long.sub-{self.subject.subject_id}')
                if not os.path.exists(fs_long_output_dir):
                    raise FileNotFoundError(f"FreeSurfer longitudinal output not found: {fs_long_output_dir}. Please check the directory or set use_freesurfer_longitudinal to False.")
                else:
                    print(f"[DWI Pipeline] FreeSurfer longitudinal output found: {fs_long_output_dir}. Related to surface metrics extraction in tractography.")

                # for session_id in session_ids:
                #     fs_long_output_dir = os.path.join(fs_subjects_dir, f'ses-{session_id}.long.sub-{self.subject.subject_id}')
                #     if not os.path.exists(fs_long_output_dir):
                #         raise FileNotFoundError(f"FreeSurfer longitudinal output not found: {fs_long_output_dir}. Please check the directory or set use_freesurfer_longitudinal to False.")
                #     else:
                #         print(f"[DWI Pipeline] FreeSurfer longitudinal output found: {fs_long_output_dir}. Related to surface metrics extraction in tractography.")

        if self.pved:
            # must run QSDR reconstruction first
            self.dsistudio_qsdr = True
        
        if self.visual_pathway_analysis and self.preprocess_method != 'post_qsiprep':
            self.dsistudio_gqi = True
        
        if self.dwi_t1w_register:
            # must run DTI fitting first
            self.dti_fit = True
        
        # Decide one seed mask to use
        lesion_mask_dir = self.session._find_output(self.seed_mask)
        seed_mask = [file for file in os.listdir(lesion_mask_dir) if f'{self.use_which_mask}' in file]
        if seed_mask is not None and len(seed_mask) == 1:
            print(f"[DWI Pipeline] Using seed mask: {seed_mask[0]}.")
            seed_mask = os.path.join(lesion_mask_dir, seed_mask[0])
        elif seed_mask is not None and len(seed_mask) > 1:
            #print(f"Using the first mask found: {seed_mask[0]}.")
            seed_mask = os.path.join(lesion_mask_dir, seed_mask[0])
            print(f"[DWI Pipeline] Using seed mask: {seed_mask}.")
        else:
            seed_mask = ''
            print("No seed mask found.")
            #raise FileNotFoundError(f"No seed mask found in {lesion_mask_dir}. Please check the directory.")

        # Whether need mrtrix3 preproc dwi (nifti to mif)
        if 'mrtrix3' in self.connectome or 'mrtrix3' in self.tractography:
            mrtrix3_preproc = True
        else:
            mrtrix3_preproc = False
        
        # Whether need fdt bedpostx
        if 'fdt' in self.connectome or 'fdt' in self.tractography:
            fdt_bedpostx = True
        else:
            fdt_bedpostx = False
        
        t1w_to_mni_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz')
        if not os.path.exists(t1w_to_mni_warp):
            t1w_to_mni_warp = ""
        mni_to_t1w_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')
        if not os.path.exists(mni_to_t1w_warp):
            mni_to_t1w_warp = ""

        # Whether need to exclude WMH mask in DWI metrics calculation
        wmh_mask_file = None
        wmh_probmap_file = None
        if self.exclude_wmh_mask:
            wmh_output_dir = self.session._find_output('wmh_quantification')
            if wmh_output_dir is None:
                raise FileNotFoundError("WMH quantification output not found. Cannot exclude WMH mask.")
            
            # sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-*_mask.nii.gz
            wmh_mask_files = [file for file in os.listdir(wmh_output_dir) if f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH" in file and file.endswith('_mask.nii.gz')]
            if len(wmh_mask_files) == 1:
                print(f"[DWI Pipeline] Exclude WMH mask: {wmh_mask_files[0]}.")
                wmh_mask_file = os.path.join(wmh_output_dir, wmh_mask_files[0])

                # sub-XXX_ses-XXX_space-T1w_label-WMH_desc-XXX_probmap.nii.gz
                wmh_probmap_files = [file for file in os.listdir(wmh_output_dir) if f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-" in file and file.endswith('_probmap.nii.gz')]
                if len(wmh_probmap_files) == 1:
                    print(f"[DWI Pipeline] WMH probmap: {wmh_probmap_files[0]}.")
                    wmh_probmap_file = os.path.join(wmh_output_dir, wmh_probmap_files[0])
                else:
                    raise FileNotFoundError("WMH probmap in T1w space not found or multiple files found. Cannot exclude WMH probmap.")

        # Space entity ('preprocdwi' for our custom pipeline, 'ACPC' for QSIPrep)
        if self.preprocess_method == 'post_qsiprep':
            space_entity = 'ACPC'
        else:
            space_entity = 'preprocdwi'

        # ============================================
        # Main Workflow
        # ============================================
        dwi_workflow = Workflow(name='dwi_workflow')

        inputnode = Node(IdentityInterface(fields=['bids_dir', 't1w_file',
                                                   'dwi_file', 'bval_file',
                                                   'bvec_file', 'json_file',
                                                   'output_path', 
                                                   'phase_encoding_number', 'total_readout_time', 
                                                   'fs_output', 'seed_mask_dtispace', 'seed_mask_t1wspace',
                                                   'fs_subjects_dir', 'fs_subject_id',
                                                   'fs_processing_dir', 'qsiprep_out_bvec',
                                                   'mni_to_t1w_warp', 't1w_to_mni_warp']), name="inputnode")

        inputnode.inputs.bids_dir = self.subject.bids_dir
        inputnode.inputs.t1w_file = t1w_file
        inputnode.inputs.dwi_file = dwi_image
        inputnode.inputs.bval_file = dwi_bval
        inputnode.inputs.bvec_file = dwi_bvec
        inputnode.inputs.json_file = dwi_json
        inputnode.inputs.output_path = self.output_path
        inputnode.inputs.phase_encoding_number = phase_encoding_direction_dict[phase_encoding_direction]
        inputnode.inputs.total_readout_time = total_readout_time
        inputnode.inputs.fs_output = fs_output
        inputnode.inputs.fs_subjects_dir = fs_subjects_dir
        inputnode.inputs.fs_subject_id = fs_subject_id
        inputnode.inputs.seed_mask_t1wspace = seed_mask
        inputnode.inputs.mni_to_t1w_warp = mni_to_t1w_warp
        inputnode.inputs.t1w_to_mni_warp = t1w_to_mni_warp

        # ============================================
        # Preprocessing (DWI)
        # ============================================
        # Directories for intermediate results
        preproc_intermediate_dir = os.path.join(self.output_path, 'preproc_intermediate')

        # Node to store the b0_all and acqparam for TOPUP
        b0_all_node = Node(IdentityInterface(fields=['b0_all', 'acqparam']), name='b0_all')

        # Node to store preprocessed DWI data
        preproc_dwi_node = Node(IdentityInterface(fields=['preproc_dwi', 'bvec', 'bval', 'dwi_mask',
                                                          'b0', 'preproc_t1w']) # These two used for QSIPrep output
                                                          , name='preproc_dwi')
        
        if self.preprocess:
            print("[DWI Pipeline] Output resolution set to {} mm.".format(self.output_resolution))
            print("[DWI Pipeline] Flip b-table axis: {}.".format(self.flip_b_table_axis))

            # Padding: make sure the slices in the z direction are even (TOPUP requirement)
            # If no need to pad, the output will be the same as input
            padding_node = Node(PadDWI(), name='padding')
            dwi_workflow.connect(inputnode, 'dwi_file', padding_node, 'in_dwi')
            dwi_workflow.connect(inputnode, 'bvec_file', padding_node, 'in_bvec')
            dwi_workflow.connect(inputnode, 'bval_file', padding_node, 'in_bval')
            dwi_workflow.connect(inputnode, 'json_file', padding_node, 'in_json')
            padding_node.inputs.out_file = os.path.join(preproc_intermediate_dir, rename_bids_file(dwi_image, {"desc": "cropped"}, 'dwi', '.nii.gz'))
            
            # Denoise using MRtrix3
            denoise_node = Node(MrtrixDenoise(), name='denoise')
            dwi_workflow.connect(padding_node, 'out_file', denoise_node, 'dwi_img')
            dwi_workflow.connect(padding_node, 'out_bvec', denoise_node, 'dwi_bvec')
            dwi_workflow.connect(padding_node, 'out_bval', denoise_node, 'dwi_bval')
            denoise_node.inputs.output_dir = preproc_intermediate_dir

            # # Degibbs using MRtrix3
            # degibbs_node = Node(MrtrixDegibbs(), name='degibbs')
            # dwi_workflow.connect(denoise_node, 'output_dwi_img', degibbs_node, 'dwi_img')
            # dwi_workflow.connect(denoise_node, 'output_dwi_bvec', degibbs_node, 'dwi_bvec')
            # dwi_workflow.connect(denoise_node, 'output_dwi_bval', degibbs_node, 'dwi_bval')
            # degibbs_node.inputs.output_dir = preproc_intermediate_dir

            # Degibbs using DIPY
            if self.degibbs:
                degibbs_node = Node(DipyDegibbs(), name='degibbs')
                dwi_workflow.connect(denoise_node, 'output_dwi_img', degibbs_node, 'dwi_file')
                degibbs_node.inputs.output_dwi = os.path.join(preproc_intermediate_dir, 'dwi_degibbs.nii.gz')
            else:
                print("[DWI Pipeline] Skip degibbs.")
                degibbs_node = Node(IdentityInterface(fields=['output_dwi_img', 'output_dwi_bvec', 'output_dwi_bval']), name='pesudo_degibbs')
                dwi_workflow.connect(denoise_node, 'output_dwi_img', degibbs_node, 'output_dwi_img')
                dwi_workflow.connect(denoise_node, 'output_dwi_bvec', degibbs_node, 'output_dwi_bvec')
                dwi_workflow.connect(denoise_node, 'output_dwi_bval', degibbs_node, 'output_dwi_bval')

            # Get b0 image for TOPUP
            # Need a b0_all image and acqparam.txt file for FSL TOPUP
            if self.synb0:
                print("[DWI Pipeline] Use Synb0 for TOPUP.")
                synb0_node = Node(Synb0(), name='synb0')
                dwi_workflow.connect([
                    (inputnode, synb0_node, [('t1w_file', 't1w_img'),
                                                ('json_file', 'dwi_json')]),
                    (degibbs_node, synb0_node, [('output_dwi_img', 'dwi_img')])
                ])

                synb0_node.inputs.fmap_output_dir = os.path.join(self.subject.bids_dir, f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}", 'fmap')
                synb0_node.inputs.output_path_synb0 = os.path.join(self.subject.bids_dir, 'derivatives', 'synb0', f'sub-{self.subject.subject_id}', f"ses-{self.session.session_id}")

                dwi_workflow.connect(synb0_node, 'acqparam', b0_all_node, 'acqparam')
                dwi_workflow.connect(synb0_node, 'b0_all', b0_all_node, 'b0_all')
            else:
                print("[DWI Pipeline] Use reverse b0 image for TOPUP.")
                if reverse_b0_image is not None:
                    create_b0_acqparam_node = Node(B0AllAndAcqparam(), name='create_b0_acqparam')
                    dwi_workflow.connect([
                        (inputnode, create_b0_acqparam_node, [('phase_encoding_number', 'phase_encoding_number'),
                                                            ('total_readout_time', 'total_readout_time')]),
                        (degibbs_node, create_b0_acqparam_node, [('output_dwi_img', 'dwi_img'),
                                                                        ('output_dwi_bval', 'dwi_bval')])
                    ])
                    create_b0_acqparam_node.inputs.output_path = preproc_intermediate_dir
                    create_b0_acqparam_node.inputs.reverse_dwi_img = reverse_b0_image
                    create_b0_acqparam_node.inputs.reverse_dwi_bval = reverse_b0_bval
                    create_b0_acqparam_node.inputs.reverse_phase_encoding_number = phase_encoding_direction_dict[reverse_b0_phase_encoding_direction]
                    create_b0_acqparam_node.inputs.reverse_total_readout_time = reverse_b0_total_readout_time

                    dwi_workflow.connect(create_b0_acqparam_node, 'b0_all', b0_all_node, 'b0_all')
                    dwi_workflow.connect(create_b0_acqparam_node, 'acqparam', b0_all_node, 'acqparam')

            # Main preprocessing workflow, according to the preprocess method
            if self.preprocess_method == 'fdt':
                print("[DWI Pipeline] Use FSL FDT for DWI preprocessing.")
                topup_node = Node(Topup(), name='topup')
                dwi_workflow.connect([
                    (b0_all_node, topup_node, [('b0_all', 'b0_all_file'),
                                                ('acqparam', 'acqparam_file')])
                ])
                topup_node.inputs.output_basename = os.path.join(preproc_intermediate_dir, 'topup_results')
                topup_node.inputs.output_b0_basename = os.path.join(preproc_intermediate_dir, 'hifi_b0')
                topup_node.inputs.config_file = 'b02b0.cnf'

                extract_b0_node = Node(ExtractROI(), name='extract_b0')
                dwi_workflow.connect(topup_node, 'b0_image', extract_b0_node, 'in_file')
                extract_b0_node.inputs.roi_file = os.path.join(preproc_intermediate_dir, 'dwi_b0.nii.gz')
                extract_b0_node.inputs.t_min = 0
                extract_b0_node.inputs.t_size = 1

                create_dwi_mask_node = Node(SynthStrip(), name='create_dwi_mask')
                dwi_workflow.connect([
                    (extract_b0_node, create_dwi_mask_node, [('roi_file', 'image')])
                ])
                dwi_mask_filepath = os.path.join(preproc_intermediate_dir, 'dwi_brain_mask.nii.gz')
                create_dwi_mask_node.inputs.mask_file = dwi_mask_filepath

                create_index_node = Node(IndexTxt(), name='create_index')
                dwi_workflow.connect(denoise_node, 'output_dwi_bval', create_index_node, 'bval_file')
                create_index_node.inputs.output_dir = preproc_intermediate_dir

                eddy_node = Node(EddyCuda(), name='eddy')
                dwi_workflow.connect([
                    (degibbs_node, eddy_node, [('output_dwi_img', 'dwi_file')]),
                    (denoise_node, eddy_node, [('output_dwi_bvec', 'bvec_file'), ('output_dwi_bval', 'bval_file')]),
                    (b0_all_node, eddy_node, [('acqparam', 'acqparam_file')]),
                    (create_index_node, eddy_node, [('index_file', 'index_file')]),
                    (create_dwi_mask_node, eddy_node, [('mask_file', 'mask_file')]),
                    (topup_node, eddy_node, [('topup_basename', 'topup_basename')]),
                ])
                eddy_node.inputs.output_basename = os.path.join(preproc_intermediate_dir, 'eddy_corrected_data')

                order_eddy_outputs_node = Node(OrderEddyOutputs(), name='order_eddy_outputs')
                dwi_workflow.connect(eddy_node, 'output_filename', order_eddy_outputs_node, 'eddy_output_filename')
                dwi_workflow.connect(eddy_node, 'bvals', order_eddy_outputs_node, 'bval')
                order_eddy_outputs_node.inputs.output_resolution = self.output_resolution
                dwi_workflow.connect(eddy_node, 'eddy_output_dir', order_eddy_outputs_node, 'eddy_output_dir')
                dwi_workflow.connect(inputnode, 'output_path', order_eddy_outputs_node, 'new_output_dir')
                order_eddy_outputs_node.inputs.new_output_filename = rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '')

                flip_b_table_node = Node(FlipBTable(), name='flip_b_table')
                dwi_workflow.connect(order_eddy_outputs_node, 'ordered_bvec', flip_b_table_node, 'in_bvec')
                flip_b_table_node.inputs.flip_axis = self.flip_b_table_axis
                flip_b_table_node.inputs.out_bvec = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.bvec'))

                create_b0ref_and_brainmask_node = Node(B0RefAndBrainMask(), name='create_b0ref_and_brainmask')
                dwi_workflow.connect(order_eddy_outputs_node, 'ordered_dwi', create_b0ref_and_brainmask_node, 'input_dwi')
                dwi_workflow.connect(order_eddy_outputs_node, 'ordered_bval', create_b0ref_and_brainmask_node, 'input_bval')
                create_b0ref_and_brainmask_node.inputs.output_dir = self.output_path
                create_b0ref_and_brainmask_node.inputs.output_b0_filename = rename_bids_file(dwi_image, {"space": space_entity}, 'dwiref', '.nii.gz')
                create_b0ref_and_brainmask_node.inputs.output_b0_mask_filename = rename_bids_file(dwi_image, {"space": space_entity, "desc": "brain"}, 'mask', '.nii.gz')

                dwi_workflow.connect(order_eddy_outputs_node, 'ordered_dwi', preproc_dwi_node, 'preproc_dwi')
                dwi_workflow.connect(flip_b_table_node, 'out_bvec', preproc_dwi_node, 'bvec')
                dwi_workflow.connect(order_eddy_outputs_node, 'ordered_bval', preproc_dwi_node, 'bval')
                dwi_workflow.connect(create_b0ref_and_brainmask_node, 'b0_brain_mask', preproc_dwi_node, 'dwi_mask')
                dwi_workflow.connect(create_b0ref_and_brainmask_node, 'b0_image', preproc_dwi_node, 'b0')

                preproc_dwi_filename = rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.nii.gz')
            elif self.preprocess_method == 'mrtrix3':
                print("[DWI Pipeline] Use MRtrix3 for DWI preprocessing.")
                # Get b0_all.mif
                convert_b0_all_mif_node = Node(MRConvert(), name='convert_b0_all_mif')
                dwi_workflow.connect(b0_all_node, 'b0_all', convert_b0_all_mif_node, 'in_file')
                convert_b0_all_mif_node.inputs.out_file = os.path.join(preproc_intermediate_dir, 'b0_all.mif')
                convert_b0_all_mif_node.inputs.args = '-force'
                #convert_raw_dwi_mif_node.inputs.args = f'-import_pe_table {os.path.join(self.output_path, "acqparam.txt")}'

                # Convert denoised and degibbsed DWI to mif
                convert_denoised_degibbs_mif_node = Node(MRConvert(), name='convert_denoised_degibbs_mif')
                dwi_workflow.connect(degibbs_node, 'output_dwi_img', convert_denoised_degibbs_mif_node, 'in_file')
                dwi_workflow.connect(denoise_node, 'output_dwi_bvec', convert_denoised_degibbs_mif_node, 'in_bvec')
                dwi_workflow.connect(denoise_node, 'output_dwi_bval', convert_denoised_degibbs_mif_node, 'in_bval')
                convert_denoised_degibbs_mif_node.inputs.out_file = os.path.join(preproc_intermediate_dir, 'dwi_denoise_degibbs.mif')
                convert_denoised_degibbs_mif_node.inputs.args = '-force'

                # TOPUP and Eddy 
                # I don't know how to integrate TOPUP and Eddy in mrtrix3 if use synb0 :(
                # preprocess
                dwi_preprocess_node = Node(DWIPreproc(), name='dwi_preprocess')
                dwi_workflow.connect(convert_denoised_degibbs_mif_node, 'out_file', dwi_preprocess_node, 'in_file')
                dwi_workflow.connect(convert_b0_all_mif_node, 'out_file', dwi_preprocess_node, 'in_epi')
                dwi_preprocess_node.inputs.out_file = os.path.join(preproc_intermediate_dir, 'dwi_preproc.mif')
                dwi_preprocess_node.inputs.rpe_options = 'pair'
                dwi_preprocess_node.inputs.pe_dir = phase_encoding_direction

                # alternative of preprocess
                # https://community.mrtrix.org/t/synb0-for-dwifslpreproc-how/6386/2
                # TODO

                # convert to nifti
                convert_preproc_dwi_node = Node(MRConvert(), name='convert_preproc_dwi')
                dwi_workflow.connect(dwi_preprocess_node, 'out_file', convert_preproc_dwi_node, 'in_file')
                convert_preproc_dwi_node.inputs.out_file = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.nii.gz'))
                convert_preproc_dwi_node.inputs.out_bvec = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.bvec'))
                convert_preproc_dwi_node.inputs.out_bval = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.bval'))
                convert_preproc_dwi_node.inputs.args = '-force'

                # get the mask
                extract_b0_node = Node(ExtractROI(), name='extract_b0')
                dwi_workflow.connect(convert_preproc_dwi_node, 'out_file', extract_b0_node, 'in_file')
                extract_b0_node.inputs.roi_file = os.path.join(preproc_intermediate_dir, 'dwi_b0.nii.gz')
                extract_b0_node.inputs.t_min = 0
                extract_b0_node.inputs.t_size = 1

                create_dwi_mask_node = Node(SynthStrip(), name='create_dwi_mask')
                dwi_workflow.connect(extract_b0_node, 'roi_file', create_dwi_mask_node, 'image')
                create_dwi_mask_node.inputs.mask_file = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "brain"}, 'mask', '.nii.gz'))

                dwi_workflow.connect(convert_preproc_dwi_node, 'out_file', preproc_dwi_node, 'preproc_dwi')
                dwi_workflow.connect(convert_preproc_dwi_node, 'out_bvec', preproc_dwi_node, 'bvec')
                dwi_workflow.connect(convert_preproc_dwi_node, 'out_bval', preproc_dwi_node, 'bval')
                dwi_workflow.connect(create_dwi_mask_node, 'mask_file', preproc_dwi_node, 'dwi_mask')

                preproc_dwi_filename = rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.nii.gz')
        elif not self.preprocess and self.preprocess_method == 'post_qsiprep':
            # search for preprocessed dwi in qsiprep output
            qsiprep_dwi_path = os.path.join(self.subject.bids_dir, 'derivatives', 'qsiprep', f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}", 'dwi')
            # *space-ACPC_desc-preproc_dwi.nii.gz *space-ACPC_desc-preproc_dwi.bvec *space-ACPC_desc-preproc_dwi.bval *space-ACPC_desc-brain_mask.nii.gz
            target_dwi = None
            for file in os.listdir(qsiprep_dwi_path):
                if file.endswith('_desc-preproc_dwi.nii.gz'):
                    target_dwi = os.path.join(qsiprep_dwi_path, file)
                    break
            if target_dwi is None:
                raise FileNotFoundError(f"No preprocessed DWI file found in qsiprep output: {qsiprep_dwi_path}")
            target_bvec = target_dwi.replace('_dwi.nii.gz', '_dwi.bvec')
            target_bval = target_dwi.replace('_dwi.nii.gz', '_dwi.bval')
            target_mask = target_dwi.replace('_desc-preproc_dwi.nii.gz', '_desc-brain_mask.nii.gz')
            if not os.path.exists(target_bvec) or not os.path.exists(target_bval) or not os.path.exists(target_mask):
                raise FileNotFoundError(f"bvec, bval or brain mask file not found in qsiprep output: {qsiprep_dwi_path}")
                
            # search for preprocessed t1w in qsiprep output
            # situation 1: qsiprep/sub-XXX/anat/sub-XXX_space-ACPC_desc-preproc_T1w.nii.gz (subject-wise)
            # situation 2: qsiprep/sub-XXX/ses-YYY/anat/sub-XXX_ses-YYY_space-ACPC_desc-preproc_T1w.nii.gz (session-wise)
            qsiprep_t1w_path1 = os.path.join(self.subject.bids_dir, 'derivatives', 'qsiprep', f"sub-{self.subject.subject_id}", 'anat', f"sub-{self.subject.subject_id}_space-ACPC_desc-preproc_T1w.nii.gz")
            qsiprep_t1w_path2 = os.path.join(self.subject.bids_dir, 'derivatives', 'qsiprep', f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}", 'anat', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-ACPC_desc-preproc_T1w.nii.gz")
            if os.path.exists(qsiprep_t1w_path1):
                target_t1w = qsiprep_t1w_path1
            elif os.path.exists(qsiprep_t1w_path2):
                target_t1w = qsiprep_t1w_path2
            else:
                raise FileNotFoundError(f"No preprocessed T1w file found in qsiprep output: {qsiprep_t1w_path1} or {qsiprep_t1w_path2}")
                
            # check QSIPrep outputs
            print(f"[DWI Pipeline] You selected to use QSIPrep preprocessed DWI and T1w images.")
            print(f"[DWI Pipeline] Found preprocessed DWI: {target_dwi}")
            print(f"[DWI Pipeline] Found preprocessed T1w: {target_t1w}")

            # Flip bvec
            print("[DWI Pipeline] Flip b-table axis: {}.".format(self.flip_b_table_axis))
            inputnode.inputs.qsiprep_out_bvec = target_bvec

            flip_b_table_node = Node(FlipBTable(), name='flip_b_table_qsiprep')
            dwi_workflow.connect([(inputnode, flip_b_table_node, [('qsiprep_out_bvec', 'in_bvec')])])
            flip_b_table_node.inputs.flip_axis = self.flip_b_table_axis
            flip_b_table_node.inputs.out_bvec = target_bvec

            preproc_dwi_node.inputs.preproc_dwi = target_dwi
            #preproc_dwi_node.inputs.bvec = target_bvec
            dwi_workflow.connect([(flip_b_table_node, preproc_dwi_node, [('out_bvec', 'bvec')])])
            preproc_dwi_node.inputs.bval = target_bval
            preproc_dwi_node.inputs.dwi_mask = target_mask
            preproc_dwi_node.inputs.b0 = target_dwi.replace('_desc-preproc_dwi.nii.gz', '_dwiref.nii.gz')
            preproc_dwi_node.inputs.preproc_t1w = target_t1w

            preproc_dwi_filename = rename_bids_file(dwi_image, {"space": "ACPC", "desc": "preproc"}, 'dwi', '.nii.gz')
        else:
            print("[DWI Pipeline] Assuming DWI has been preprocessed and arching to output directory.")
            preproc_dwi_node.inputs.preproc_dwi = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.nii.gz'))
            preproc_dwi_node.inputs.bvec = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.bvec'))
            preproc_dwi_node.inputs.bval = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.bval'))
            preproc_dwi_node.inputs.dwi_mask = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity, "desc": "brain"}, 'mask', '.nii.gz'))
            preproc_dwi_node.inputs.b0 = os.path.join(self.output_path, rename_bids_file(dwi_image, {"space": space_entity}, 'dwiref', '.nii.gz'))
            preproc_dwi_filename = rename_bids_file(dwi_image, {"space": space_entity, "desc": "preproc"}, 'dwi', '.nii.gz')

        # MRtrix3 conversion (nifti to mif): preprocessed DWI and brain mask
        if mrtrix3_preproc:
            mrtrix3_output_dir = os.path.join(self.output_path, 'mrtrix3')
            os.makedirs(mrtrix3_output_dir, exist_ok=True)

            preproc_dwi_mif_node = Node(IdentityInterface(fields=['preproc_dwi', 'dwi_mask']), name='preproc_dwi_mif_node')

            preproc_dwi_mif = os.path.join(mrtrix3_output_dir, rename_bids_file(preproc_dwi_filename, {}, 'dwi', '.mif'))
            if not os.path.exists(preproc_dwi_mif):
                mri_convert_node = Node(MRConvert(), name='dwi_mif_convert')
                mri_convert_node.inputs.out_file = preproc_dwi_mif
                mri_convert_node.inputs.args = '-force'
                dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', mri_convert_node, 'in_file')
                dwi_workflow.connect(preproc_dwi_node, 'bvec', mri_convert_node, 'in_bvec')
                dwi_workflow.connect(preproc_dwi_node, 'bval', mri_convert_node, 'in_bval')

                dwi_workflow.connect(mri_convert_node, 'out_file', preproc_dwi_mif_node, 'preproc_dwi')
            else:
                preproc_dwi_mif_node.inputs.preproc_dwi = preproc_dwi_mif

            brain_mask_mif = os.path.join(mrtrix3_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': 'brain'}, 'mask', '.mif'))
            if not os.path.exists(brain_mask_mif):
                mask_mif_convert = Node(MRConvert(), name='mask_mif_convert')
                mask_mif_convert.inputs.out_file = brain_mask_mif
                mask_mif_convert.inputs.args = '-force'
                dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', mask_mif_convert, 'in_file')

                dwi_workflow.connect(mask_mif_convert, 'out_file', preproc_dwi_mif_node, 'dwi_mask')
            else:
                preproc_dwi_mif_node.inputs.dwi_mask = brain_mask_mif

        # ===========================================
        # DWI scalar maps storage node
        dwi_scalarmaps_output_node = Node(IdentityInterface(fields=[
            'fa_img', 'md_img', 'markvcid2_fw_img', # Tensor model
            'odi_img', 'icvf_img', 'isovf_img',  # NODDI model
            'gqi_gfa_img', 'gqi_iso_img', 'gqi_qa_img', # GQI model
            'chidia_img' # QSM!
        ]), name='dwi_scalarmaps_output')
        # default all to ''
        dwi_scalarmaps_output_node.inputs.fa_img = ''
        dwi_scalarmaps_output_node.inputs.md_img = ''
        dwi_scalarmaps_output_node.inputs.markvcid2_fw_img = ''
        dwi_scalarmaps_output_node.inputs.odi_img = ''
        dwi_scalarmaps_output_node.inputs.icvf_img = ''
        dwi_scalarmaps_output_node.inputs.isovf_img = ''
        dwi_scalarmaps_output_node.inputs.gqi_gfa_img = ''
        dwi_scalarmaps_output_node.inputs.gqi_iso_img = ''
        dwi_scalarmaps_output_node.inputs.gqi_qa_img = ''
        dwi_scalarmaps_output_node.inputs.chidia_img = ''
        # ===========================================

        # ===========================================
        # Reconstruction (FSL FDT dtifit)
        # ===========================================
        dti_fit_output_node = Node(
            IdentityInterface(fields=['fa_img', 'md_img', 'tensor_img']),
            name='dti_fit_output'
        )

        if self.dti_fit:
            print("[DWI Pipeline] DTI fitting using FSL dtifit: True")
            dtifit_output_dir = os.path.join(self.output_path, 'dtifit')
            os.makedirs(dtifit_output_dir, exist_ok=True)

            target_fa_output = os.path.join(dtifit_output_dir, rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "tensor", "param": "fa"}, "dwimap", ".nii.gz"))

            if os.path.exists(target_fa_output):
                print(f"[DWI Pipeline] DTI FA image already exists: {target_fa_output}. Skipping DTI fit.")

                dti_fit_output_node.inputs.fa_img = target_fa_output
                dti_fit_output_node.inputs.md_img = os.path.join(dtifit_output_dir,rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "tensor", "param": "md"}, "dwimap", ".nii.gz"))
                dti_fit_output_node.inputs.tensor_img = os.path.join(dtifit_output_dir, rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "tensor", "param": "tensor"}, "dwimap", ".nii.gz"))
            else:
                dti_fit_node = Node(DTIFitBIDS(), name='dti_fit')

                dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', dti_fit_node, 'dwi_file')
                dwi_workflow.connect(preproc_dwi_node, 'bvec',       dti_fit_node, 'bvec_file')
                dwi_workflow.connect(preproc_dwi_node, 'bval',       dti_fit_node, 'bval_file')
                dwi_workflow.connect(preproc_dwi_node, 'dwi_mask',   dti_fit_node, 'mask_file')
                dti_fit_node.inputs.output_basename = os.path.join(dtifit_output_dir, 'dti')
                dti_fit_node.inputs.bids_rename = True
                dti_fit_node.inputs.overwrite = False

                dwi_workflow.connect(dti_fit_node, 'dti_fa',     dti_fit_output_node, 'fa_img')
                dwi_workflow.connect(dti_fit_node, 'dti_md',     dti_fit_output_node, 'md_img')
                dwi_workflow.connect(dti_fit_node, 'dti_tensor', dti_fit_output_node, 'tensor_img')
            
        # Connect DTI outputs to dwi_scalarmaps_output_node
        dwi_workflow.connect(dti_fit_output_node, 'fa_img', dwi_scalarmaps_output_node, 'fa_img')
        dwi_workflow.connect(dti_fit_output_node, 'md_img', dwi_scalarmaps_output_node, 'md_img')
        
        # ============================================
        # DWI and T1w registration (And other anat preprocessing)
        # ============================================
        # We put it here because we prefer to use FA image for registration (FLIRT), while QSIPrep uses b0 image !
        if self.dwi_t1w_register:
            print("[DWI Pipeline] Register DWI to T1w image: True")
            xfm_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}")
            if self.session.xfm_dir is None:
                os.makedirs(os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}"), exist_ok=True)

            if self.preprocess_method == 'post_qsiprep':
                dwi_to_t1w_reg_node = Node(QSIPrepOrigToACPC(), name='dwi_to_t1w_reg_qsiprep')
                dwi_workflow.connect(inputnode, 't1w_file', dwi_to_t1w_reg_node, 'orig_t1w_file')
                dwi_workflow.connect(dti_fit_output_node, 'fa_img', dwi_to_t1w_reg_node, 'dwimap_file')

                dwi_to_t1w_reg_node.inputs.subject_id = self.subject.subject_id
                dwi_to_t1w_reg_node.inputs.session_id = self.session.session_id
                dwi_to_t1w_reg_node.inputs.output_dir = xfm_dir
                dwi_workflow.connect(preproc_dwi_node, 'preproc_t1w', dwi_to_t1w_reg_node, 'preprocess_t1w_file')
            else:
                dwi_to_t1w_reg_node = Node(FLIRT(), name='dwi_to_t1w_reg')
                
                t1w_stripped = os.path.join(xfm_dir, rename_bids_file(t1w_file, {'desc': 'brain', 'space': None}, 'T1w', '.nii.gz'))
                dwi_workflow.connect(dti_fit_output_node, 'fa_img', dwi_to_t1w_reg_node, 'in_file')
                if os.path.exists(t1w_stripped):
                    dwi_to_t1w_reg_node.inputs.reference = t1w_stripped
                else:
                    t1w_strip_node = Node(SynthStrip(), name='t1w_strip_for_dwi_reg')
                    dwi_workflow.connect(inputnode, 't1w_file', t1w_strip_node, 'image')
                    t1w_strip_node.inputs.out_file = t1w_stripped
                    dwi_workflow.connect(t1w_strip_node, 'out_file', dwi_to_t1w_reg_node, 'reference')
                dwi_to_t1w_reg_node.inputs.out_file = os.path.join(xfm_dir, rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "tensor", "param": "fa", 'space': 'T1w'}, 'dwimap', '.nii.gz'))
                dwi_to_t1w_reg_node.inputs.out_matrix_file = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-{space_entity}_to-T1w_xfm.mat")
            
            # need to invert the matrix for later use
            invert_dwi_to_t1w_reg_node = Node(ConvertXFM(), name='invert_dwi_to_t1w_reg')
            dwi_workflow.connect(dwi_to_t1w_reg_node, 'out_matrix_file', invert_dwi_to_t1w_reg_node, 'in_file')
            invert_dwi_to_t1w_reg_node.inputs.out_file = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-{space_entity}_xfm.mat")
            invert_dwi_to_t1w_reg_node.inputs.invert_xfm = True
            
            # If seed_mask_t1wspace is provided, we need to transform it to DWI space
            if seed_mask != '':
                seed_mask_to_dwi_node = Node(FLIRT(), name='seed_mask_to_dwi')
                dwi_workflow.connect(inputnode, 'seed_mask_t1wspace', seed_mask_to_dwi_node, 'in_file')
                dwi_workflow.connect(invert_dwi_to_t1w_reg_node, 'out_file', seed_mask_to_dwi_node, 'in_matrix_file')
                dwi_workflow.connect(preproc_dwi_node, 'b0', seed_mask_to_dwi_node, 'reference')
                seed_mask_to_dwi_node.inputs.interp = 'nearestneighbour'
                seed_mask_to_dwi_node.inputs.apply_xfm = True
                seed_mask_to_dwi_node.inputs.out_file = os.path.join(os.path.dirname(seed_mask), rename_bids_file(seed_mask, {'space': space_entity}, 'mask', '.nii.gz'))
            
            if fs_output_process:
                anat_output_dir = os.path.join(self.output_path, 'anat')
                os.makedirs(anat_output_dir, exist_ok=True)

                # Freesurfer to T1w registration
                fs_to_t1w_xfm_node = Node(Tkregister2fs2t1w(), name='fs_to_t1w_xfm')
                dwi_workflow.connect(inputnode, 'fs_subjects_dir', fs_to_t1w_xfm_node, 'fs_subjects_dir')
                dwi_workflow.connect(inputnode, 'fs_subject_id', fs_to_t1w_xfm_node, 'fs_subject_id')
                fs_to_t1w_xfm_node.inputs.output_matrix = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-fs_to-T1w_xfm.mat")
                fs_to_t1w_xfm_node.inputs.output_inverse_matrix = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-fs_xfm.mat")

                concat_fs_to_dwi_xfm_node = Node(ConvertXFM(), name='concat_fs_to_dwi_xfm')
                dwi_workflow.connect(fs_to_t1w_xfm_node, 'output_matrix', concat_fs_to_dwi_xfm_node, 'in_file')
                concat_fs_to_dwi_xfm_node.inputs.concat_xfm = True
                dwi_workflow.connect(invert_dwi_to_t1w_reg_node, 'out_file', concat_fs_to_dwi_xfm_node, 'in_file2')
                concat_fs_to_dwi_xfm_node.inputs.out_file = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-fs_to-{space_entity}_xfm.mat")

                # Seed mask to fs space registration
                if seed_mask != '':
                    seed_mask_to_fs_node = Node(MRIVol2Vol(), name='seed_mask_to_fs')
                    seed_mask_to_fs_node.inputs.moving_image = seed_mask # in T1w space
                    dwi_workflow.connect(fs_to_t1w_xfm_node, 'output_inverse_matrix', seed_mask_to_fs_node, 'fsl_matrix')
                    seed_mask_to_fs_node.inputs.target_image = os.path.join(fs_output, 'mri', 'orig.mgz')  # fs native space
                    seed_mask_to_fs_node.inputs.interp = 'nearestneighbour'
                    seed_mask_to_fs_node.inputs.output_image = os.path.join(os.path.dirname(seed_mask), rename_bids_file(seed_mask, {'space': 'fs'}, 'mask', '.nii.gz'))

                # Refine freesurfer aparc+aseg parcellation
                def get_aseg_file(fs_subjects_dir, fs_subject_id):
                    import os
                    return os.path.join(fs_subjects_dir, fs_subject_id, 'mri', 'aparc+aseg.mgz')

                get_aseg_node = Node(Function(input_names=['fs_subjects_dir', 'fs_subject_id'],
                                            output_names=['aseg_file'],
                                            function=get_aseg_file), name='get_aseg_file')
                dwi_workflow.connect(inputnode, 'fs_subjects_dir', get_aseg_node, 'fs_subjects_dir')
                dwi_workflow.connect(inputnode, 'fs_subject_id', get_aseg_node, 'fs_subject_id')

                aseg_mgz_to_nifti_node = Node(MRIConvert(), name='aseg_mgz_to_nifti')
                dwi_workflow.connect(get_aseg_node, 'aseg_file', aseg_mgz_to_nifti_node, 'in_file')
                aseg_mgz_to_nifti_node.inputs.out_file = os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-fs_aparcaseg.nii.gz")

                # convert to DWI space
                fs_aparcaseg_to_dwi_node = Node(FLIRT(), name='fs_aparcaseg_to_dwi')
                dwi_workflow.connect(aseg_mgz_to_nifti_node, 'out_file', fs_aparcaseg_to_dwi_node, 'in_file')
                dwi_workflow.connect(concat_fs_to_dwi_xfm_node, 'out_file', fs_aparcaseg_to_dwi_node, 'in_matrix_file')
                dwi_workflow.connect(preproc_dwi_node, 'b0', fs_aparcaseg_to_dwi_node, 'reference')
                fs_aparcaseg_to_dwi_node.inputs.interp = 'nearestneighbour'
                fs_aparcaseg_to_dwi_node.inputs.apply_xfm = True
                fs_aparcaseg_to_dwi_node.inputs.out_file = os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_aparcaseg.nii.gz")

                # Remove WMH from aseg
                refine_aseg_node = Node(RemoveWMH(), name='refine_aseg')
                dwi_workflow.connect(fs_aparcaseg_to_dwi_node, 'out_file', refine_aseg_node, 'in_aseg')
                refine_aseg_node.inputs.out_aseg = os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_desc-WMHremoved_aparcaseg.nii.gz")

                # Create Tissue masks
                create_WM_masks_node = MapNode(MRIBinarize(), name='create_WM_masks', iterfield=['output_volume', 'match'])
                dwi_workflow.connect(refine_aseg_node, 'out_aseg', create_WM_masks_node, 'input_volume')
                create_WM_masks_node.inputs.match = [
                    [2, 41, 77, 251, 252, 253, 254, 255],  # Cerebral WM
                    [2, 41, 77, 251, 252, 253, 254, 255, 7, 46], # All WM
                    # [2, 41, 251, 252, 253, 254, 255, 7, 46] # NAWM (raw)
                    [2, 41, 251, 252, 253, 254, 255] # NAWM (raw): only cerebral WM
                ]
                create_WM_masks_node.inputs.output_volume = [
                    os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_label-cerebralWM_mask.nii.gz"),
                    os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_label-WM_mask.nii.gz"),
                    os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_label-rawNAWM_mask.nii.gz"),
                ]

                create_cortical_GM_mask_node = Node(MergeRibbon(), name='create_cortical_GM_mask')
                dwi_workflow.connect(inputnode, 'fs_subjects_dir', create_cortical_GM_mask_node, 'subjects_dir')
                dwi_workflow.connect(inputnode, 'fs_subject_id', create_cortical_GM_mask_node, 'subject_id')
                create_cortical_GM_mask_node.inputs.output_gm_mask = os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-fs_label-corticalGM_mask.nii.gz")

                cortical_GM_to_dwi_node = Node(FLIRT(), name='cortical_GM_to_dwi')
                dwi_workflow.connect(create_cortical_GM_mask_node, 'output_gm_mask', cortical_GM_to_dwi_node, 'in_file')
                dwi_workflow.connect(concat_fs_to_dwi_xfm_node, 'out_file', cortical_GM_to_dwi_node, 'in_matrix_file')
                dwi_workflow.connect(preproc_dwi_node, 'b0', cortical_GM_to_dwi_node, 'reference')
                cortical_GM_to_dwi_node.inputs.interp = "nearestneighbour"
                cortical_GM_to_dwi_node.inputs.apply_xfm = True
                cortical_GM_to_dwi_node.inputs.out_file = os.path.join(
                    anat_output_dir, rename_bids_file(create_cortical_GM_mask_node.inputs.output_gm_mask, {"space": space_entity}, "mask", ".nii.gz")
                )

                # refine NAWM mask by excluding WMH and seed mask (if configured)
                get_raw_nawm_mask_node = Node(Select(index=2), name="select_raw_nawm_mask")
                dwi_workflow.connect(create_WM_masks_node, 'output_volume', get_raw_nawm_mask_node, 'inlist')

                def collect_exclusion_masks(seed_mask, wmh_mask, use_seed, use_wmh):
                    import os
                    masks = []
                    if use_seed and seed_mask and os.path.exists(seed_mask):
                        masks.append(seed_mask)
                    if use_wmh and wmh_mask and os.path.exists(wmh_mask):
                        masks.append(wmh_mask)
                    return masks


                collect_excl_node = Node(
                    Function(
                        input_names=["seed_mask", "wmh_mask", "use_seed", "use_wmh"],
                        output_names=["masks"],
                        function=collect_exclusion_masks,
                    ),
                    name="collect_exclusion_masks",
                )

                collect_excl_node.inputs.use_seed = bool(self.exclude_seed_mask and seed_mask != "")
                collect_excl_node.inputs.use_wmh = bool(self.exclude_wmh_mask)

                if self.exclude_seed_mask and seed_mask != "":
                    dwi_workflow.connect(seed_mask_to_dwi_node, "out_file", collect_excl_node, "seed_mask")
                else:
                    collect_excl_node.inputs.seed_mask = ""

                if self.exclude_wmh_mask:
                    wmh_mask_to_dwi_node = Node(FLIRT(), name="wmh_mask_to_dwi")
                    wmh_mask_to_dwi_node.inputs.in_file = wmh_mask_file
                    dwi_workflow.connect(invert_dwi_to_t1w_reg_node, "out_file", wmh_mask_to_dwi_node, "in_matrix_file")
                    dwi_workflow.connect(preproc_dwi_node, "b0", wmh_mask_to_dwi_node, "reference")
                    wmh_mask_to_dwi_node.inputs.interp = "nearestneighbour"
                    wmh_mask_to_dwi_node.inputs.apply_xfm = True
                    wmh_mask_to_dwi_node.inputs.out_file = os.path.join(
                        anat_output_dir, rename_bids_file(wmh_mask_file, {"space": space_entity}, "mask", ".nii.gz")
                    )
                    dwi_workflow.connect(wmh_mask_to_dwi_node, "out_file", collect_excl_node, "wmh_mask")
                else:
                    collect_excl_node.inputs.wmh_mask = ""


                def merge_exclusion_masks(mask_files, ref_file, out_file):
                    import os
                    import subprocess

                    if mask_files is None:
                        mask_files = []
                    mask_files = [m for m in mask_files if m]

                    os.makedirs(os.path.dirname(out_file), exist_ok=True)

                    if len(mask_files) == 0:
                        subprocess.check_call(["fslmaths", ref_file, "-mul", "0", out_file])
                        return out_file

                    cmd = ["fslmaths", mask_files[0]]
                    for m in mask_files[1:]:
                        cmd += ["-add", m]
                    cmd += ["-thr", "0.5", "-bin", out_file]
                    subprocess.check_call(cmd)
                    return out_file


                merge_masks_to_exclude_node = Node(
                    Function(
                        input_names=["mask_files", "ref_file", "out_file"],
                        output_names=["out_file"],
                        function=merge_exclusion_masks,
                    ),
                    name="merge_masks_to_exclude",
                )

                dwi_workflow.connect(collect_excl_node, "masks", merge_masks_to_exclude_node, "mask_files")

                # Use a reference image in the same target space to generate an all-zero mask if needed
                dwi_workflow.connect(preproc_dwi_node, "b0", merge_masks_to_exclude_node, "ref_file")

                merge_masks_to_exclude_node.inputs.out_file = os.path.join(
                    anat_output_dir,
                    f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_desc-exclusionMerged_mask.nii.gz",
                )


                get_final_nawm_node = Node(GenerateNAWMMask(), name="get_final_nawm_mask")
                dwi_workflow.connect(get_raw_nawm_mask_node, "out", get_final_nawm_node, "wm_mask")
                dwi_workflow.connect(merge_masks_to_exclude_node, "out_file", get_final_nawm_node, "exclude_mask")
                get_final_nawm_node.inputs.erode_mm = 2
                get_final_nawm_node.inputs.output_mask = os.path.join(
                    anat_output_dir,
                    f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_label-NAWM_mask.nii.gz",
                )

                final_nawm_mask = os.path.join(
                    anat_output_dir,
                    f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-{space_entity}_label-NAWM_mask.nii.gz",
                )

                if self.visual_pathway_analysis:
                    # check QSM output
                    chidia_in_t1w_file = os.path.join(
                        self.subject.bids_dir, 'derivatives', 'qsm_pipeline', f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}", "QSM_registered",
                        f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_Chidia.nii.gz",
                    )
                    if os.path.exists(chidia_in_t1w_file):
                        chidia_to_dwi_node = Node(FLIRT(), name='chidia_to_dwi')
                        chidia_to_dwi_node.inputs.in_file = chidia_in_t1w_file
                        dwi_workflow.connect(invert_dwi_to_t1w_reg_node, 'out_file', chidia_to_dwi_node, 'in_matrix_file')
                        dwi_workflow.connect(preproc_dwi_node, 'b0', chidia_to_dwi_node, 'reference')
                        chidia_to_dwi_node.inputs.interp = 'trilinear'
                        chidia_to_dwi_node.inputs.apply_xfm = True
                        chidia_to_dwi_node.inputs.out_file = os.path.join(anat_output_dir, rename_bids_file(chidia_in_t1w_file, {'space': space_entity}, "Chidia", ".nii.gz"))
                        dwi_workflow.connect(chidia_to_dwi_node, 'out_file', dwi_scalarmaps_output_node, 'chidia_img')

                        wmhprobmap_to_dwi_node = Node(FLIRT(), name='wmhprobmap_to_dwi')
                        wmhprobmap_to_dwi_node.inputs.in_file = wmh_probmap_file
                        dwi_workflow.connect(invert_dwi_to_t1w_reg_node, 'out_file', wmhprobmap_to_dwi_node, 'in_matrix_file')
                        dwi_workflow.connect(preproc_dwi_node, 'b0', wmhprobmap_to_dwi_node, 'reference')
                        wmhprobmap_to_dwi_node.inputs.interp = 'trilinear'
                        wmhprobmap_to_dwi_node.inputs.apply_xfm = True
                        wmhprobmap_to_dwi_node.inputs.out_file = os.path.join(anat_output_dir, rename_bids_file(wmh_probmap_file, {'space': space_entity}, "probmap", ".nii.gz"))

        # ============================================
        # DSI Studio Reconstruction
        # ============================================
        if self.dsistudio_gqi or self.dsistudio_qsdr:
            # first need to convert to .src.gz format
            os.makedirs(os.path.join(self.output_path, 'dsistudio'), exist_ok=True)
            create_src_node = Node(CreateSRC(), name='create_src')
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', create_src_node, 'source')
            create_src_node.inputs.output = os.path.join(self.output_path, 'dsistudio', rename_bids_file(preproc_dwi_filename, {"desc": None, "desc": "preproc"}, 'dwi', '.src.gz'))

            gqi_out = os.path.join(self.output_path, 'dsistudio', rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "gqi"}, 'dwimap', '.fib.gz'))
            qsdr_out = os.path.join(self.output_path, 'dsistudio', rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "qsdr"}, 'dwimap', '.fib.gz'))

            if self.dsistudio_gqi:
                gqi_reconstruction_node = Node(DSIstudioReconstruction(), name='gqi_reconstruction')
                dwi_workflow.connect(create_src_node, 'out_file', gqi_reconstruction_node, 'source')
                gqi_reconstruction_node.inputs.method = 4
                gqi_reconstruction_node.inputs.other_output = 'fa,ad,rd,md,iso,rdi,nrdi,tensor'
                gqi_reconstruction_node.inputs.output = gqi_out

            if self.dsistudio_qsdr:
                qsdr_reconstruction_node = Node(DSIstudioReconstruction(), name='qsdr_reconstruction')
                dwi_workflow.connect(create_src_node, 'out_file', qsdr_reconstruction_node, 'source')
                qsdr_reconstruction_node.inputs.method = 7
                qsdr_reconstruction_node.inputs.qsdr_reso = 2.0
                qsdr_reconstruction_node.inputs.other_output = 'fa,ad,rd,md,iso,rdi,nrdi,tensor'
                qsdr_reconstruction_node.inputs.output = qsdr_out
            
        if self.preprocess_method == 'post_qsiprep':
            qsirecon_output_dir = self.session._find_output('qsirecon-DSIStudio')
            gqi_fib = os.path.join(qsirecon_output_dir, 'dwi', rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "gqi"}, 'dwimap', '.fib.gz'))
            if not os.path.exists(gqi_fib):
                raise FileNotFoundError(f"GQI .fib.gz file not found in {qsirecon_output_dir}. Please run QSIRecon with DSIStudio reconstruction first.")
            
            dwi_scalarmaps_output_node.inputs.gqi_gfa_img = os.path.join(qsirecon_output_dir, 'dwi', rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "gqi", "param": "gfa"}, 'dwimap', '.nii.gz'))
            dwi_scalarmaps_output_node.inputs.gqi_iso_img = os.path.join(qsirecon_output_dir, 'dwi', rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "gqi", "param": "iso"}, 'dwimap', '.nii.gz'))
            dwi_scalarmaps_output_node.inputs.gqi_qa_img = os.path.join(qsirecon_output_dir, 'dwi', rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "gqi", "param": "qa"}, 'dwimap', '.nii.gz'))

        # ===========================================
        # AMICO NODDI Reconstruction
        # ===========================================
        if self.amico_noddi:
            from cvdproc.pipelines.dmri.amico.amico_nipype import AmicoNoddi
            
            amico_output_dir = os.path.join(self.output_path, 'NODDI')

            amico_noddi_node = Node(AmicoNoddi(), name='amico_noddi')
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', amico_noddi_node, 'dwi')
            dwi_workflow.connect(preproc_dwi_node, 'bval', amico_noddi_node, 'bval')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', amico_noddi_node, 'bvec')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', amico_noddi_node, 'mask')
            amico_noddi_node.inputs.output_dir = amico_output_dir
            amico_noddi_node.inputs.direction_filename = rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "noddi", 'param': 'direction'}, 'dwimap', '.nii.gz')
            amico_noddi_node.inputs.icvf_filename = rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "noddi", 'param': 'icvf'}, 'dwimap', '.nii.gz')
            amico_noddi_node.inputs.isovf_filename = rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "noddi", 'param': 'isovf'}, 'dwimap', '.nii.gz')
            amico_noddi_node.inputs.od_filename = rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "noddi", 'param': 'odi'}, 'dwimap', '.nii.gz')
            amico_noddi_node.inputs.modulated_icvf_filename = rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "noddi", 'param': 'ModulatedICVF'}, 'dwimap', '.nii.gz')
            amico_noddi_node.inputs.modulated_od_filename = rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "noddi", 'param': 'ModulatedODI'}, 'dwimap', '.nii.gz')
            amico_noddi_node.inputs.config_filename = rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "noddi"}, 'config', '.pickle')

            # Connect outputs to dwi_scalarmaps_output_node
            dwi_workflow.connect(amico_noddi_node, 'icvf', dwi_scalarmaps_output_node, 'icvf_img')
            dwi_workflow.connect(amico_noddi_node, 'isovf', dwi_scalarmaps_output_node, 'isovf_img')
            dwi_workflow.connect(amico_noddi_node, 'od', dwi_scalarmaps_output_node, 'odi_img')

        # ===========================================
        # FSL FDT bedpostx
        # ===========================================
        if fdt_bedpostx:
            bedpostx_output_dir = os.path.join(self.output_path, 'bedpostx')
            bedpostx_node = Node(BedpostxGPUCustom(), name="bedpostx")

            dwi_workflow.connect(preproc_dwi_node, "preproc_dwi", bedpostx_node, "dwi_img")
            dwi_workflow.connect(preproc_dwi_node, "bvec",       bedpostx_node, "bvec")
            dwi_workflow.connect(preproc_dwi_node, "bval",       bedpostx_node, "bval")
            dwi_workflow.connect(preproc_dwi_node, "dwi_mask",   bedpostx_node, "mask")

            bedpostx_node.inputs.out_dir = bedpostx_output_dir

        # ===========================================
        # Connectome
        # ===========================================
        if self.connectome:
            connectome_output_dir = os.path.join(self.output_path, 'connectome')
            os.makedirs(connectome_output_dir, exist_ok=True)

            # Prepare connectome node
            prepare_connectome_node = Node(ConnectomePrepare(), name='prepare_connectome')
            dwi_workflow.connect(preproc_dwi_mif_node, 'preproc_dwi', prepare_connectome_node, 'preproc_dwi_mif')
            dwi_workflow.connect(preproc_dwi_mif_node, 'dwi_mask', prepare_connectome_node, 'dwi_mask_mif')
            prepare_connectome_node.inputs.output_dir = connectome_output_dir
            dwi_workflow.connect(refine_aseg_node, 'out_aseg', prepare_connectome_node, 'aseg')
            prepare_connectome_node.inputs.wm_response = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'WM', 'desc': None}, 'dwimap', '.txt')
            prepare_connectome_node.inputs.wm_fod = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'WM', 'desc': None}, 'dwimap', '.mif')
            prepare_connectome_node.inputs.wm_fod_norm = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'WM', 'desc': 'norm'}, 'dwimap', '.mif')
            prepare_connectome_node.inputs.gm_response = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'GM', 'desc': None}, 'dwimap', '.txt')
            prepare_connectome_node.inputs.gm_fod = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'GM', 'desc': None}, 'dwimap', '.mif')
            prepare_connectome_node.inputs.gm_fod_norm = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'GM', 'desc': 'norm'}, 'dwimap', '.mif')
            prepare_connectome_node.inputs.csf_response = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'CSF', 'desc': None}, 'dwimap', '.txt')
            prepare_connectome_node.inputs.csf_fod = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'CSF', 'desc': None}, 'dwimap', '.mif')
            prepare_connectome_node.inputs.csf_fod_norm = rename_bids_file(preproc_dwi_filename, {'param': 'fod', 'label': 'CSF', 'desc': 'norm'}, 'dwimap', '.mif')
            prepare_connectome_node.inputs.sift_mu = rename_bids_file(preproc_dwi_filename, {'model': 'sift2', 'desc': None}, 'mu', '.txt')
            prepare_connectome_node.inputs.five_tt_dwi = rename_bids_file(preproc_dwi_filename, {'desc': None}, '5tt', '.mif')
            prepare_connectome_node.inputs.gmwmSeed_dwi = rename_bids_file(preproc_dwi_filename, {'desc': None}, 'gmwmSeed', '.mif')
            prepare_connectome_node.inputs.streamlines = rename_bids_file(preproc_dwi_filename, {'desc': None}, 'streamlines', '.tck')
            prepare_connectome_node.inputs.sift_weights = rename_bids_file(preproc_dwi_filename, {'model': 'sift2', 'desc': None}, 'streamlineweights', '.txt')

            # Prepare parcellation
            fs_parcellation_config = Node(Merge(2), name='merge_fs_parcellation_config')
            fs_parcellation_config.inputs.in1 = get_package_path('data', 'labelconvert', 'fs_default.txt')
            fs_parcellation_config.inputs.in2 = get_package_path('data', 'labelconvert', 'custom', 'fs_aparc.txt')

            prepare_fs_parcellation_node = MapNode(LabelConvert(), name='prepare_parcellation', iterfield=['in_config', 'out_file'])
            dwi_workflow.connect(prepare_connectome_node, 'aseg_dwi', prepare_fs_parcellation_node, 'in_file')
            prepare_fs_parcellation_node.inputs.in_lut = get_package_path('data', 'labelconvert_in', 'FreeSurferColorLUT.txt')
            dwi_workflow.connect(fs_parcellation_config, 'out', prepare_fs_parcellation_node, 'in_config')
            prepare_fs_parcellation_node.inputs.out_file = [
                os.path.join(connectome_output_dir, rename_bids_file(preproc_dwi_filename, {'seg': 'aparcaseg', 'desc': None}, 'parcellation', '.mif')),
                os.path.join(connectome_output_dir, rename_bids_file(preproc_dwi_filename, {'seg': 'aparc', 'desc': None}, 'parcellation', '.mif'))
            ]

            # Connectome construction
            construct_connectome_node = MapNode(BuildConnectome(), name='construct_connectome', iterfield=['in_parc', 'out_file'])
            dwi_workflow.connect(prepare_connectome_node, 'global_streamlines', construct_connectome_node, 'in_file')
            construct_connectome_node.inputs.out_file = [
                os.path.join(connectome_output_dir, rename_bids_file(preproc_dwi_filename, {'seg': 'aparcaseg', 'desc': None}, 'connectivity', '.csv')),
                os.path.join(connectome_output_dir, rename_bids_file(preproc_dwi_filename, {'seg': 'aparc', 'desc': None}, 'connectivity', '.csv'))
            ]
            dwi_workflow.connect(prepare_fs_parcellation_node, 'out_file', construct_connectome_node, 'in_parc')
            dwi_workflow.connect(prepare_connectome_node, 'sift_weights', construct_connectome_node, 'in_weights')
            construct_connectome_node.inputs.args = '-zero_diagonal -symmetric -scale_invnodevol'

        # ===========================================
        # Tractography: Designed for lesion-based probabilistic tractography
        # ===========================================
        tractography_output_dir = os.path.join(self.output_path, 'tractography')
        if 'fdt' in self.tractography:
            fdt_tractography_output_dir = os.path.join(tractography_output_dir, 'fdt')
            os.makedirs(fdt_tractography_output_dir, exist_ok=True)

            uncorrected_tractography_output_dir = os.path.join(fdt_tractography_output_dir, 'PathLengthUncorrected')
            seed_mask_tractography = Node(Probtrackx(), name='seed_mask_tractography')
            dwi_workflow.connect(bedpostx_node, 'source_for_probtrackx', seed_mask_tractography, 'source')
            dwi_workflow.connect(bedpostx_node, 'mask_for_probtrackx', seed_mask_tractography, 'dwi_mask')
            dwi_workflow.connect(seed_mask_to_fs_node, 'output_image', seed_mask_tractography, 'seed_mask')  # in fs space
            dwi_workflow.connect(concat_fs_to_dwi_xfm_node, 'out_file', seed_mask_tractography, 'seed_to_dwi_xfm')
            seed_mask_tractography.inputs.seed_ref = os.path.join(fs_output, 'mri', 'orig.mgz')  # fs native space
            dwi_workflow.connect(create_cortical_GM_mask_node, 'output_gm_mask', seed_mask_tractography, 'waypoints')  # in fs space
            seed_mask_tractography.inputs.path_length_correction = False
            seed_mask_tractography.inputs.output_dir = uncorrected_tractography_output_dir
            seed_mask_tractography.inputs.nsamples = 10000
            seed_mask_tractography.inputs.args = '-l --forcedir --opd --ompl'

            waytotal_node = Node(ReadSingleValue(), name='waytotal_node')
            dwi_workflow.connect(seed_mask_tractography, 'waytotal', waytotal_node, 'input_file')

            lh_fdtpaths_to_surf = Node(MRIvol2surf(), name='lh_fdtpaths_to_surf')
            dwi_workflow.connect(seed_mask_tractography, 'fdt_paths', lh_fdtpaths_to_surf, 'volume')
            lh_fdtpaths_to_surf.inputs.subjects_dir = fs_subjects_dir
            lh_fdtpaths_to_surf.inputs.regheader = fs_subject_id
            lh_fdtpaths_to_surf.inputs.hemi = 'lh'
            lh_fdtpaths_to_surf.inputs.output_surf = os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-fsaverage_TDI.mgh')
            lh_fdtpaths_to_surf.inputs.proj_frac = 0.5
            lh_fdtpaths_to_surf.inputs.target = 'fsaverage'

            mask_lh_fdtpaths_surf = Node(ApplyIndexTxtToMgh(), name='mask_lh_fdtpaths_surf')
            dwi_workflow.connect(lh_fdtpaths_to_surf, 'output_surf', mask_lh_fdtpaths_surf, 'measure_mgh')
            mask_lh_fdtpaths_surf.inputs.index_txt = get_package_path('data', 'standard', 'fsaverage', 'lh.aparc.label_medial_wall.txt')
            mask_lh_fdtpaths_surf.inputs.output_mgh = os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-fsaverage_desc-NoMedialWall_TDI.mgh')

            rh_fdtpaths_to_surf = Node(MRIvol2surf(), name='rh_fdtpaths_to_surf')
            dwi_workflow.connect(seed_mask_tractography, 'fdt_paths', rh_fdtpaths_to_surf, 'volume')
            rh_fdtpaths_to_surf.inputs.subjects_dir = fs_subjects_dir
            rh_fdtpaths_to_surf.inputs.regheader = fs_subject_id
            rh_fdtpaths_to_surf.inputs.hemi = 'rh'
            rh_fdtpaths_to_surf.inputs.output_surf = os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-fsaverage_TDI.mgh')
            rh_fdtpaths_to_surf.inputs.proj_frac = 0.5
            rh_fdtpaths_to_surf.inputs.target = 'fsaverage'

            mask_rh_fdtpaths_surf = Node(ApplyIndexTxtToMgh(), name='mask_rh_fdtpaths_surf')
            dwi_workflow.connect(rh_fdtpaths_to_surf, 'output_surf', mask_rh_fdtpaths_surf, 'measure_mgh')
            mask_rh_fdtpaths_surf.inputs.index_txt = get_package_path('data', 'standard', 'fsaverage', 'rh.aparc.label_medial_wall.txt')
            mask_rh_fdtpaths_surf.inputs.output_mgh = os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-fsaverage_desc-NoMedialWall_TDI.mgh')

            define_connection_level_node = Node(DefineConnectionLevel(), name='define_connection_levels')
            dwi_workflow.connect(mask_lh_fdtpaths_surf, 'output_mgh', define_connection_level_node, 'lh_mgh')
            dwi_workflow.connect(mask_rh_fdtpaths_surf, 'output_mgh', define_connection_level_node, 'rh_mgh')
            define_connection_level_node.inputs.low_threshold = 3.8e-5
            dwi_workflow.connect(waytotal_node, 'value', define_connection_level_node, 'divisor')
            define_connection_level_node.inputs.output_files = {
                'lh_unconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-fsaverage_desc-UnConn_mask.mgh'),
                'rh_unconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-fsaverage_desc-UnConn_mask.mgh'),
                'lh_lowconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-fsaverage_desc-LowConn_mask.mgh'),
                'rh_lowconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-fsaverage_desc-LowConn_mask.mgh'),
                'lh_medconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-fsaverage_desc-MediumConn_mask.mgh'),
                'rh_medconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-fsaverage_desc-MediumConn_mask.mgh'),
                'lh_highconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-fsaverage_desc-HighConn_mask.mgh'),
                'rh_highconn': os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-fsaverage_desc-HighConn_mask.mgh'),
            }

            # remask unconn regions
            mask_lh_unconn_node = Node(ApplyGiiMaskToMgh(), name='mask_lh_unconn_regions')
            dwi_workflow.connect(define_connection_level_node, 'lh_unconn_mask', mask_lh_unconn_node, 'measure_mgh')
            mask_lh_unconn_node.inputs.mask_gii = get_package_path('data', 'standard', 'fsaverage', 'lh.thickness.shape.gii')
            mask_lh_unconn_node.inputs.output_mgh = os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-fsaverage_desc-UnConn_mask.mgh')
            mask_rh_unconn_node = Node(ApplyGiiMaskToMgh(), name='mask_rh_unconn_regions')
            dwi_workflow.connect(define_connection_level_node, 'rh_unconn_mask', mask_rh_unconn_node, 'measure_mgh')
            mask_rh_unconn_node.inputs.mask_gii = get_package_path('data', 'standard', 'fsaverage', 'rh.thickness.shape.gii')
            mask_rh_unconn_node.inputs.output_mgh = os.path.join(uncorrected_tractography_output_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-fsaverage_desc-UnConn_mask.mgh')

            # extract surface parameters
            uncorr_extract_surf_params = Node(ExtractSurfaceParameters(), name='uncorr_extract_surf_params')
            uncorr_extract_surf_params.inputs.fs_subjects_dir = fs_subjects_dir
            uncorr_extract_surf_params.inputs.sessions = self.subject.sessions_id
            if self.use_freesurfer_longitudinal:
                uncorr_extract_surf_params.inputs.sessions = self.subject.sessions_id + [f'long.sub-{self.subject.subject_id}']
            dwi_workflow.connect(mask_lh_unconn_node, 'output_mgh', uncorr_extract_surf_params, 'lh_unconn_mask')
            dwi_workflow.connect(mask_rh_unconn_node, 'output_mgh', uncorr_extract_surf_params, 'rh_unconn_mask')
            dwi_workflow.connect(define_connection_level_node, 'lh_lowconn_mask', uncorr_extract_surf_params, 'lh_low_conn_mask')
            dwi_workflow.connect(define_connection_level_node, 'rh_lowconn_mask', uncorr_extract_surf_params, 'rh_low_conn_mask')
            dwi_workflow.connect(define_connection_level_node, 'lh_medconn_mask', uncorr_extract_surf_params, 'lh_medium_conn_mask')
            dwi_workflow.connect(define_connection_level_node, 'rh_medconn_mask', uncorr_extract_surf_params, 'rh_medium_conn_mask')
            dwi_workflow.connect(define_connection_level_node, 'lh_highconn_mask', uncorr_extract_surf_params, 'lh_high_conn_mask')
            dwi_workflow.connect(define_connection_level_node, 'rh_highconn_mask', uncorr_extract_surf_params, 'rh_high_conn_mask')
            uncorr_extract_surf_params.inputs.output_dir = uncorrected_tractography_output_dir
            uncorr_extract_surf_params.inputs.csv_file_name = f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-ConnectivityLevels_surfpars.csv'

        if 'mrtrix3' in self.tractography:
            seed_based_track_node = Node(IdentityInterface(fields=['seed_based_track']), name='seed_based_track')
            tckgen_output_dir = os.path.join(tractography_output_dir, 'mrtrix3')

            tckgen_node = Node(Tractography(), name='tckgen')
            os.makedirs(tckgen_output_dir, exist_ok=True)
            dwi_workflow.connect(preproc_dwi_mif_node, 'preproc_dwi', tckgen_node, 'in_file')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', tckgen_node, 'roi_mask')
            dwi_workflow.connect(seed_mask_to_dwi_node, 'out_file', tckgen_node, 'seed_image')
            tckgen_node.inputs.out_file = os.path.join(tckgen_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': 'ROIseed'}, 'streamlines', '.tck'))
            tckgen_node.inputs.algorithm = 'Tensor_Prob'
            tckgen_node.inputs.select = 10000
            tckgen_node.inputs.nthreads = 4
            tckgen_node.inputs.args = '-force'

            dwi_workflow.connect(tckgen_node, 'out_file', seed_based_track_node, 'seed_based_track')

            if fs_output_process:
                nawm_without_tract = os.path.join(tckgen_output_dir, rename_bids_file(final_nawm_mask, {'desc': 'ConnTractRemoved'}, 'mask', '.nii.gz'))

                remove_tract_region_node = Node(RemoveTractRegion(), name='remove_tract_region')
                dwi_workflow.connect(seed_based_track_node, 'seed_based_track', remove_tract_region_node, 'tck_file')
                dwi_workflow.connect(get_final_nawm_node, 'nawm_mask', remove_tract_region_node, 'wm_mask')
                remove_tract_region_node.inputs.out_tract_mask = os.path.join(tckgen_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': 'ROIseedTract'}, 'mask', '.nii.gz'))
                remove_tract_region_node.inputs.out_tdi_norm = os.path.join(tckgen_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': 'ROIseedTract'}, 'tdi', '.nii.gz'))
                remove_tract_region_node.inputs.out_wm_mask = nawm_without_tract

        # ===========================================
        # DTI-ALPS
        # ===========================================
        if self.dtialps and self.preprocess_method != 'post_qsiprep':
            print("[DWI Pipeline] DTI-ALPS analysis: True")
            dti_alps_node = Node(ALPS(), name='dti_alps')
            dwi_workflow.connect(dti_fit_output_node, 'fa_img', dti_alps_node, 'fa_img')
            dti_alps_node.inputs.output_dir = os.path.join(self.output_path, 'DTI-ALPS')
            dti_alps_node.inputs.alps_dir = get_package_path("pipelines", "external", "alps")
            # self.dtialps_register_method: 1 for 'flirt', 2 for 'synthmorph'
            dti_alps_node.inputs.register_method = 'flirt' if self.dtialps_register_method == 1 else 'synthmorph' if self.dtialps_register_method == 2 else 'ants'
            dwi_workflow.connect(dti_fit_output_node, 'tensor_img', dti_alps_node, 'tensor_img')

        # ===========================================
        # PVeD 
        # ===========================================
        if self.pved:
            print("[DWI Pipeline] PVeD analysis: True")
            pved_node = Node(PVeD(), name='pved')

            dwi_workflow.connect(qsdr_reconstruction_node, 'out_file', pved_node, 'qsdr_fib_file')
            pved_node.inputs.output_dir = os.path.join(self.output_path, 'PVeD')
            pved_node.inputs.script_path = get_package_path("pipelines", "matlab", "pved", "pved_single.m")
            pved_node.inputs.spm_path = get_package_path("data", "matlab_toolbox", "spm12")
            pved_node.inputs.pved_path = get_package_path("data", "matlab_toolbox", "EstPVeD", 'MATLAB', 'pkg_pved_est')
        
        # ===========================================
        # Free Water
        # ===========================================
        freewater_node = Node(IdentityInterface(fields=['single_shell_fw_img', 'markvcid_fw_img', 'dti_fw_img']), name='freewater_node')

        if 'single_shell_freewater' in self.freewater:
            single_shell_freewater_node = Node(SingleShellFW(), name='single_shell_freewater')

            dwi_workflow.connect(inputnode, 'bval_file', single_shell_freewater_node, 'fbval')

            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', single_shell_freewater_node, 'fdwi')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', single_shell_freewater_node, 'fbvec')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', single_shell_freewater_node, 'mask_file')

            fw_output_path = os.path.join(self.output_path, 'freewater', 'single_shell_freewater')
            single_shell_freewater_node.inputs.working_directory = fw_output_path
            single_shell_freewater_node.inputs.output_directory = fw_output_path
            single_shell_freewater_node.inputs.crop_shells = False

            dwi_workflow.connect(single_shell_freewater_node, 'output_fw', freewater_node, 'fw_img')
        else:
            freewater_node.inputs.fw_img = os.path.join(self.output_path, 'single_shell_freewater', 'freewater.nii.gz')
        
        if 'markvcid_freewater' in self.freewater:
            markvcid_freewater_node = Node(MarkVCIDFreeWater(), name='markvcid_freewater')

            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', markvcid_freewater_node, 'in_dwi')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', markvcid_freewater_node, 'in_dwi_bvec')
            dwi_workflow.connect(preproc_dwi_node, 'bval', markvcid_freewater_node, 'in_dwi_bval')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', markvcid_freewater_node, 'in_dwi_mask')

            fw_output_path = os.path.join(self.output_path, 'freewater', 'markvcid_freewater')
            markvcid_freewater_node.inputs.output_dir = fw_output_path
            markvcid_freewater_node.inputs.script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'MarkVCID2', 'scripts_FW_CONSORTIUM'))
            markvcid_freewater_node.inputs.output_fw_img = os.path.join(os.path.dirname(fw_output_path), rename_bids_file(preproc_dwi_filename, {'model': 'tensor', 'param': 'freewater', 'desc': 'MarkVCID2'}, 'dwimap', '.nii.gz'))

            dwi_workflow.connect(markvcid_freewater_node, 'out_fw', freewater_node, 'markvcid_fw_img')

            dwi_workflow.connect(markvcid_freewater_node, 'out_fw', dwi_scalarmaps_output_node, 'markvcid2_fw_img')
        
        if 'dti_freewater' in self.freewater:
            dti_freewater_node = Node(FreeWaterTensor(), name='dti_freewater')
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', dti_freewater_node, 'dwi_file')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', dti_freewater_node, 'bvec_file')
            dwi_workflow.connect(preproc_dwi_node, 'bval', dti_freewater_node, 'bval_file')
            dwi_workflow.connect(preproc_dwi_node, 'dwi_mask', dti_freewater_node, 'mask_file')

            fw_output_path = os.path.join(self.output_path, 'freewater', 'tensor_model_freewater')
            dti_freewater_node.inputs.output_dir = fw_output_path

            dwi_workflow.connect(dti_freewater_node, 'freewater_file', freewater_node, 'dti_fw_img')
        else:
            freewater_node.inputs.dti_fw_img = os.path.join(self.output_path, 'tensor_model_freewater', 'dti_freewater.nii.gz')
        
        # ===========================================
        # PSMD
        # ===========================================
        if self.psmd:
            print("[DWI Pipeline] PSMD analysis: True")
            psmd_output_dir = os.path.join(self.output_path, 'psmd')

            psmd_node = Node(PSMDCommandLine(), name="psmd_node")
            dwi_workflow.connect(preproc_dwi_node, 'preproc_dwi', psmd_node, 'dwi_data')
            dwi_workflow.connect(preproc_dwi_node, 'bval', psmd_node, 'bval_file')
            dwi_workflow.connect(preproc_dwi_node, 'bvec', psmd_node, 'bvec_file')
            psmd_node.inputs.mask_file = self.psmd_skeleton_mask
            if self.psmd_exclude_seed_mask:
                dwi_workflow.connect(seed_mask_to_dwi_node, 'out_file', psmd_node, 'lesion_mask')
            psmd_node.inputs.output_dir = psmd_output_dir
        
        ###########################
        # Visual Pathway Analysis #
        ###########################
        if self.visual_pathway_analysis:
            vpa_output_dir = os.path.join(self.output_path, 'visual_pathway_analysis')
            os.makedirs(vpa_output_dir, exist_ok=True)

            gather_gqi_fib_node = Node(IdentityInterface(fields=['gqi_fib']), name='gqi_fib_gather')

            if self.preprocess_method == 'post_qsiprep':
                qsirecon_output_dir = self.session._find_output('qsirecon-DSIStudio')
                gqi_fib = os.path.join(qsirecon_output_dir, 'dwi', rename_bids_file(preproc_dwi_filename, {"desc": None, "model": "gqi"}, 'dwimap', '.fib.gz'))
                if not os.path.exists(gqi_fib):
                    raise FileNotFoundError(f"GQI .fib.gz file not found in {qsirecon_output_dir}. Please run QSIRecon with DSIStudio reconstruction first.")
                gather_gqi_fib_node.inputs.gqi_fib = gqi_fib
            else:
                dwi_workflow.connect(gqi_reconstruction_node, 'out_file', gather_gqi_fib_node, 'gqi_fib')

            # 1. Prepare ROIs
            from cvdproc.pipelines.dmri.vp_project.prepare_roi_nipype import PrepareVPROI
            prepare_vp_roi_node = Node(PrepareVPROI(), name='prepare_vp_roi')
            dwi_workflow.connect(inputnode, 'fs_output', prepare_vp_roi_node, 'fs_output_dir')
            prepare_vp_roi_node.inputs.subject_id = self.subject.subject_id
            prepare_vp_roi_node.inputs.session_id = self.session.session_id
            dwi_workflow.connect(concat_fs_to_dwi_xfm_node, 'out_file', prepare_vp_roi_node, 'fs_to_dwi_xfm')
            dwi_workflow.connect(preproc_dwi_node, 'b0', prepare_vp_roi_node, 'dwi_ref')
            prepare_vp_roi_node.inputs.output_dir = anat_output_dir
            prepare_vp_roi_node.inputs.space_entity = space_entity

            # ROI for Meyer's Loop
            roi4meyersloop_to_t1w_node = MapNode(MRIConvertApplyWarp(), name='roi4meyersloop_to_t1w', iterfield=['input_image', 'output_image'])
            dwi_workflow.connect(inputnode, 'mni_to_t1w_warp', roi4meyersloop_to_t1w_node, 'warp_image')
            roi4meyersloop_to_t1w_node.inputs.input_image = [
                get_package_path('data', 'standard', 'MNI152', 'meyers_loop_roi', 'meyers_loop_roi_L.nii.gz'),
                get_package_path('data', 'standard', 'MNI152', 'meyers_loop_roi', 'meyers_loop_roi_R.nii.gz')
            ]
            roi4meyersloop_to_t1w_node.inputs.output_image = [
                os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-T1w_desc-roi4meyersloop_mask.nii.gz"),
                os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-T1w_desc-roi4meyersloop_mask.nii.gz")
            ]
            roi4meyersloop_to_t1w_node.inputs.interp = 'nearest'

            roi4meyersloop_to_dwi_node = MapNode(FLIRT(), name='roi4meyersloop_to_dwi', iterfield=['in_file', 'out_file'])
            dwi_workflow.connect(invert_dwi_to_t1w_reg_node, 'out_file', roi4meyersloop_to_dwi_node, 'in_matrix_file')
            dwi_workflow.connect(roi4meyersloop_to_t1w_node, 'output_image', roi4meyersloop_to_dwi_node, 'in_file')
            roi4meyersloop_to_dwi_node.inputs.out_file = [
                os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_space-{space_entity}_desc-roi4meyersloop_mask.nii.gz"),
                os.path.join(anat_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_space-{space_entity}_desc-roi4meyersloop_mask.nii.gz")
            ]
            dwi_workflow.connect(preproc_dwi_node, 'b0', roi4meyersloop_to_dwi_node, 'reference')
            roi4meyersloop_to_dwi_node.inputs.interp = "nearestneighbour"
            roi4meyersloop_to_dwi_node.inputs.apply_xfm = True

            # select left and right roi separately
            fetch_roi4meyersloop_L = Node(Select(index=0), name="fetch_roi4meyersloop_L")
            fetch_roi4meyersloop_R = Node(Select(index=1), name="fetch_roi4meyersloop_R")

            dwi_workflow.connect(roi4meyersloop_to_dwi_node, "out_file", fetch_roi4meyersloop_L, "inlist")
            dwi_workflow.connect(roi4meyersloop_to_dwi_node, "out_file", fetch_roi4meyersloop_R, "inlist")

            # 2. ROI Tractography
            raw_track_output_dir = os.path.join(vpa_output_dir, 'raw_tracts')
            os.makedirs(raw_track_output_dir, exist_ok=True)
            lh_or_track_node = Node(DSIstudioTracking(), name='lh_or_track')
            dwi_workflow.connect(gather_gqi_fib_node, 'gqi_fib', lh_or_track_node, 'source')
            lh_or_track_node.inputs.output = os.path.join(raw_track_output_dir, 'lh_OR.tt.gz')
            lh_or_track_node.inputs.thread_count = 8
            lh_or_track_node.inputs.tract_count = 2500
            #lh_or_track_node.inputs.turning_angle = 45
            lh_or_track_node.inputs.max_length = 200
            lh_or_track_node.inputs.args = '--tip_iteration=0 --method=0'
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_lgn_dil3x_roi', lh_or_track_node, 'seed')
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_v1_ext2_roi', lh_or_track_node, 'end')
            #dwi_workflow.connect(fetch_roi4meyersloop_L, 'out', lh_or_track_node, 'ter')

            rh_or_track_node = Node(DSIstudioTracking(), name='rh_or_track')
            dwi_workflow.connect(gather_gqi_fib_node, 'gqi_fib', rh_or_track_node, 'source')
            rh_or_track_node.inputs.output = os.path.join(raw_track_output_dir, 'rh_OR.tt.gz')
            rh_or_track_node.inputs.thread_count = 8
            rh_or_track_node.inputs.tract_count = 2500
            #rh_or_track_node.inputs.turning_angle = 45
            rh_or_track_node.inputs.max_length = 200
            rh_or_track_node.inputs.args = '--tip_iteration=0 --method=0'
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_lgn_dil3x_roi', rh_or_track_node, 'seed')
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_v1_ext2_roi', rh_or_track_node, 'end')
            #dwi_workflow.connect(fetch_roi4meyersloop_R, 'out', rh_or_track_node, 'ter')

            lh_ot_track_node = Node(DSIstudioTracking(), name='lh_ot_track')
            dwi_workflow.connect(gather_gqi_fib_node, 'gqi_fib', lh_ot_track_node, 'source')
            lh_ot_track_node.inputs.output = os.path.join(raw_track_output_dir, 'lh_OT.tt.gz')
            lh_ot_track_node.inputs.thread_count = 8
            lh_ot_track_node.inputs.tract_count = 2500
            lh_ot_track_node.inputs.max_length = 60
            lh_ot_track_node.inputs.args = '--tip_iteration=2 --method=1'
            dwi_workflow.connect(prepare_vp_roi_node, 'optic_chiasm_dil1_roi', lh_ot_track_node, 'seed')
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_lgn_dil1_roi', lh_ot_track_node, 'end')

            rh_ot_track_node = Node(DSIstudioTracking(), name='rh_ot_track')
            dwi_workflow.connect(gather_gqi_fib_node, 'gqi_fib', rh_ot_track_node, 'source')
            rh_ot_track_node.inputs.output = os.path.join(raw_track_output_dir, 'rh_OT.tt.gz')
            rh_ot_track_node.inputs.thread_count = 8
            rh_ot_track_node.inputs.tract_count = 2500
            rh_ot_track_node.inputs.max_length = 60
            rh_ot_track_node.inputs.args = '--tip_iteration=2 --method=1'
            dwi_workflow.connect(prepare_vp_roi_node, 'optic_chiasm_dil1_roi', rh_ot_track_node, 'seed')
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_lgn_dil1_roi', rh_ot_track_node, 'end')

            # Experimental: Meyer's Loop
            lr_ml_track_node = Node(DSIstudioTracking(), name='lr_meyersloop_track')
            dwi_workflow.connect(gather_gqi_fib_node, 'gqi_fib', lr_ml_track_node, 'source')
            lr_ml_track_node.inputs.output = os.path.join(raw_track_output_dir, 'lr_MeyersLoop.tt.gz')
            lr_ml_track_node.inputs.thread_count = 8
            lr_ml_track_node.inputs.tract_count = 2500
            lr_ml_track_node.inputs.turning_angle = 45
            lr_ml_track_node.inputs.args = '--tip_iteration=2 --method=1'
            dwi_workflow.connect(fetch_roi4meyersloop_L, 'out', lr_ml_track_node, 'seed')
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_v1_ext2_roi', lr_ml_track_node, 'end')

            rh_ml_track_node = Node(DSIstudioTracking(), name='rh_meyersloop_track')
            dwi_workflow.connect(gather_gqi_fib_node, 'gqi_fib', rh_ml_track_node, 'source')
            rh_ml_track_node.inputs.output = os.path.join(raw_track_output_dir, 'rh_MeyersLoop.tt.gz')
            rh_ml_track_node.inputs.thread_count = 8
            rh_ml_track_node.inputs.tract_count = 2500
            rh_ml_track_node.inputs.turning_angle = 45
            rh_ml_track_node.inputs.args = '--tip_iteration=2 --method=1'
            dwi_workflow.connect(fetch_roi4meyersloop_R, 'out', rh_ml_track_node, 'seed')
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_v1_ext2_roi', rh_ml_track_node, 'end')

            # 3. Refine tracts with ROIs
            from cvdproc.pipelines.dmri.vp_project.refine_vp_nipype import RefineVP
            vp_refine_node = Node(RefineVP(), name='vp_refine')
            dwi_workflow.connect(lh_or_track_node, 'output', vp_refine_node, 'lh_or')
            dwi_workflow.connect(lh_ot_track_node, 'output', vp_refine_node, 'lh_ot')
            dwi_workflow.connect(rh_or_track_node, 'output', vp_refine_node, 'rh_or')
            dwi_workflow.connect(rh_ot_track_node, 'output', vp_refine_node, 'rh_ot')
            dwi_workflow.connect(lr_ml_track_node, 'output', vp_refine_node, 'lh_ml')
            dwi_workflow.connect(rh_ml_track_node, 'output', vp_refine_node, 'rh_ml')
            dwi_workflow.connect(prepare_vp_roi_node, 'optic_chiasm_dil1_roi', vp_refine_node, 'cho_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_lgn_dil1_roi', vp_refine_node, 'lh_lgn_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_lgn_dil3x_roi', vp_refine_node, 'lh_lgn_dia_x_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_lgn_extendpart_roi', vp_refine_node, 'lh_lgn_extendpart_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_lgn_dil1_roi', vp_refine_node, 'rh_lgn_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_lgn_dil3x_roi', vp_refine_node, 'rh_lgn_dia_x_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_lgn_extendpart_roi', vp_refine_node, 'rh_lgn_extendpart_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'lh_v1_ext2_roi', vp_refine_node, 'lh_v1_roi')
            dwi_workflow.connect(prepare_vp_roi_node, 'rh_v1_ext2_roi', vp_refine_node, 'rh_v1_roi')
            dwi_workflow.connect(fetch_roi4meyersloop_L, 'out', vp_refine_node, 'lh_meyersloop_roi')
            dwi_workflow.connect(fetch_roi4meyersloop_R, 'out', vp_refine_node, 'rh_meyersloop_roi')
            vp_refine_node.inputs.output_dir = vpa_output_dir
            vp_refine_node.inputs.output_lh_ot = rename_bids_file(preproc_dwi_filename, {'bundle': 'OT', 'hemi': 'L', 'desc': 'voxelspace'}, 'streamlines', '.tt.gz')
            vp_refine_node.inputs.output_rh_ot = rename_bids_file(preproc_dwi_filename, {'bundle': 'OT', 'hemi': 'R', 'desc': 'voxelspace'}, 'streamlines', '.tt.gz')
            vp_refine_node.inputs.output_lh_or = rename_bids_file(preproc_dwi_filename, {'bundle': 'OR', 'hemi': 'L', 'desc': 'voxelspace'}, 'streamlines', '.tt.gz')
            vp_refine_node.inputs.output_rh_or = rename_bids_file(preproc_dwi_filename, {'bundle': 'OR', 'hemi': 'R', 'desc': 'voxelspace'}, 'streamlines', '.tt.gz')
            vp_refine_node.inputs.output_lh_ml = rename_bids_file(preproc_dwi_filename, {'bundle': 'ML', 'hemi': 'L', 'desc': 'voxelspace'}, 'streamlines', '.tt.gz')
            vp_refine_node.inputs.output_rh_ml = rename_bids_file(preproc_dwi_filename, {'bundle': 'ML', 'hemi': 'R', 'desc': 'voxelspace'}, 'streamlines', '.tt.gz')

            from cvdproc.pipelines.dmri.vp_project.tt_process_nipype import TTGZToTCK, TTGZToTDI
            gather_refined_tracts = Node(Merge(6), name='gather_refined_tracts')
            dwi_workflow.connect(vp_refine_node, 'refined_lh_ot', gather_refined_tracts, 'in1')
            dwi_workflow.connect(vp_refine_node, 'refined_rh_ot', gather_refined_tracts, 'in2')
            dwi_workflow.connect(vp_refine_node, 'refined_lh_or', gather_refined_tracts, 'in3')
            dwi_workflow.connect(vp_refine_node, 'refined_rh_or', gather_refined_tracts, 'in4')
            dwi_workflow.connect(vp_refine_node, 'refined_lh_ml', gather_refined_tracts, 'in5')
            dwi_workflow.connect(vp_refine_node, 'refined_rh_ml', gather_refined_tracts, 'in6')

            # tract stats
            tract_stats_node = MapNode(TractStatsInterface(), name='tract_stats', iterfield=['tract', 'output_txt'])
            dwi_workflow.connect(gather_refined_tracts, 'out', tract_stats_node, 'tract')
            dwi_workflow.connect(gather_gqi_fib_node, 'gqi_fib', tract_stats_node, 'source')
            tract_stats_node.inputs.output_txt = [
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OT', 'hemi': 'L'}, 'stats', '.txt')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OT', 'hemi': 'R'}, 'stats', '.txt')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OR', 'hemi': 'L'}, 'stats', '.txt')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OR', 'hemi': 'R'}, 'stats', '.txt')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'ML', 'hemi': 'L'}, 'stats', '.txt')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'ML', 'hemi': 'R'}, 'stats', '.txt'))
            ]
            tract_stats_node.inputs.export = 'stat'

            # tt.gz to tck
            tt_to_tck_node = MapNode(TTGZToTCK(), name='tt_to_tck', iterfield=['tt_gz_path', 'out_tck'])
            tt_to_tck_node.synchronize = True
            dwi_workflow.connect(gather_refined_tracts, 'out', tt_to_tck_node, 'tt_gz_path')
            dwi_workflow.connect(preproc_dwi_node, 'b0', tt_to_tck_node, 'ref_nii')
            tt_to_tck_node.inputs.out_tck = [
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OT', 'hemi': 'L'}, 'streamlines', '.tck')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OT', 'hemi': 'R'}, 'streamlines', '.tck')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OR', 'hemi': 'L'}, 'streamlines', '.tck')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OR', 'hemi': 'R'}, 'streamlines', '.tck')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'ML', 'hemi': 'L'}, 'streamlines', '.tck')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'ML', 'hemi': 'R'}, 'streamlines', '.tck'))
            ]

            # tt.gz to TDI image
            tt_to_tdi_node = MapNode(TTGZToTDI(), name='tt_to_tdi', iterfield=['tt_gz_path', 'out_tdi'])
            tt_to_tdi_node.synchronize = True
            dwi_workflow.connect(gather_refined_tracts, 'out', tt_to_tdi_node, 'tt_gz_path')
            dwi_workflow.connect(preproc_dwi_node, 'b0', tt_to_tdi_node, 'ref_nii')
            tt_to_tdi_node.inputs.out_tdi = [
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OT', 'hemi': 'L'}, 'tdi', '.nii.gz')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OT', 'hemi': 'R'}, 'tdi', '.nii.gz')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OR', 'hemi': 'L'}, 'tdi', '.nii.gz')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'OR', 'hemi': 'R'}, 'tdi', '.nii.gz')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'ML', 'hemi': 'L'}, 'tdi', '.nii.gz')),
                os.path.join(vpa_output_dir, rename_bids_file(preproc_dwi_filename, {'desc': None, 'bundle': 'ML', 'hemi': 'R'}, 'tdi', '.nii.gz'))
            ]

        #########################
        # Calculate DWI metrics #
        #########################
        if self.calculate_dwi_metrics:
            dwi_metrics_output_dir = os.path.join(self.output_path, 'dwi_metrics_stats')
            os.makedirs(dwi_metrics_output_dir, exist_ok=True)

            dwi_metrics_node = Node(Merge(10), name='dwi_metrics_node')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'fa_img', dwi_metrics_node, 'in1')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'md_img', dwi_metrics_node, 'in2')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'markvcid2_fw_img', dwi_metrics_node, 'in3')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'odi_img', dwi_metrics_node, 'in4')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'icvf_img', dwi_metrics_node, 'in5')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'isovf_img', dwi_metrics_node, 'in6')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'gqi_gfa_img', dwi_metrics_node, 'in7')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'gqi_iso_img', dwi_metrics_node, 'in8')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'gqi_qa_img', dwi_metrics_node, 'in9')
            dwi_workflow.connect(dwi_scalarmaps_output_node, 'chidia_img', dwi_metrics_node, 'in10')

            exist_dwi_metrics_node = Node(FilterExisting(), name='exist_dwi_metrics')
            dwi_workflow.connect(dwi_metrics_node, 'out', exist_dwi_metrics_node, 'input_file_list')

            if self.visual_pathway_analysis:
                # Add WMH probmap to scalar maps
                vp_dwi_metrics_node = Node(Merge(2), name='vp_dwi_metrics_node')
                dwi_workflow.connect(dwi_metrics_node, 'out', vp_dwi_metrics_node, 'in1')
                dwi_workflow.connect(wmhprobmap_to_dwi_node, 'out_file', vp_dwi_metrics_node, 'in2')

            # -------------------------------------- #
            # Calculate DWI metrics using .tck files #
            # -------------------------------------- #
            if 'mrtrix3' in self.tractography:
                seed_tract_scalar_maps_node = Node(TckSampleMultiScalarBundle(), name='seed_tract_scalar_maps')
                dwi_workflow.connect(seed_based_track_node, 'seed_based_track', seed_tract_scalar_maps_node, 'tck_file')
                dwi_workflow.connect(dwi_metrics_node, 'out', seed_tract_scalar_maps_node, 'scalar_files')
                seed_tract_scalar_maps_node.inputs.scalar_names = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF", "GQI_GFA", "GQI_ISO", "GQI_QA", "CHIDIA"]
                seed_tract_scalar_maps_node.inputs.stat_tck = 'mean'
                seed_tract_scalar_maps_node.inputs.output_csv = os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-ROIseedTract_desc-mean_dwimap.csv")

            if self.visual_pathway_analysis:
                vp_tract_scalar_maps_node = MapNode(TckSampleMultiScalarProfile(), name='vp_tract_scalar_maps', iterfield=['tck_file', 'output_csv'])
                dwi_workflow.connect(tt_to_tck_node, 'out_tck', vp_tract_scalar_maps_node, 'tck_file')
                dwi_workflow.connect(vp_dwi_metrics_node, 'out', vp_tract_scalar_maps_node, 'scalar_files')
                vp_tract_scalar_maps_node.inputs.scalar_names = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF", "GQI_GFA", "GQI_ISO", "GQI_QA", "CHIDIA", "WMHprobmap"]
                vp_tract_scalar_maps_node.inputs.output_csv = [
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_label-OT_desc-alongtract_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_label-OT_desc-alongtract_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_label-OR_desc-alongtract_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_label-OR_desc-alongtract_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_label-ML_desc-alongtract_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_label-ML_desc-alongtract_dwimap.csv")
                ]

            # ------------------------------------------------ #
            # Calculate DWI metrics using volumetric masks     #
            # ------------------------------------------------ #
            # 1. Seed mask
            if seed_mask != '' and seed_mask is not None:
                calc_seedmask_node = Node(CalculateScalarMaps(), name="scalar_maps_for_seedmask")
                calc_seedmask_node.inputs.colnames = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF", "GQI_GFA", "GQI_ISO", "GQI_QA", "CHIDIA"]
                calc_seedmask_node.inputs.roi_label = 1
                calc_seedmask_node.inputs.output_csv = os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-seedmask_desc-mean_dwimap.csv")
                dwi_workflow.connect(seed_mask_to_dwi_node, 'out_file', calc_seedmask_node, 'mask_file')
                dwi_workflow.connect(dwi_metrics_node, 'out', calc_seedmask_node, 'data_files')

            # 2. WMH mask
            if wmh_mask_file != '' and wmh_mask_file is not None:
                scalar_maps_for_wmhmask_node = Node(CalculateScalarMaps(), name='scalar_maps_for_wmhmask')
                scalar_maps_for_wmhmask_node.inputs.colnames = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF", "GQI_GFA", "GQI_ISO", "GQI_QA", "CHIDIA"]
                scalar_maps_for_wmhmask_node.inputs.roi_label = 1
                scalar_maps_for_wmhmask_node.inputs.output_csv = os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-WMH_desc-mean_dwimap.csv")
                dwi_workflow.connect(wmh_mask_to_dwi_node, 'out_file', scalar_maps_for_wmhmask_node, 'mask_file')
                dwi_workflow.connect(dwi_metrics_node, 'out', scalar_maps_for_wmhmask_node, 'data_files')
            
            # 3. NAWM mask
            if self.dwi_t1w_register and fs_output_process:
                scalar_maps_for_nawmmask_node = Node(CalculateScalarMaps(), name='scalar_maps_for_nawmmask')
                scalar_maps_for_nawmmask_node.inputs.colnames = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF", "GQI_GFA", "GQI_ISO", "GQI_QA", "CHIDIA"]
                scalar_maps_for_nawmmask_node.inputs.roi_label = 1
                scalar_maps_for_nawmmask_node.inputs.output_csv = os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-NAWM_desc-mean_dwimap.csv")
                dwi_workflow.connect(get_final_nawm_node, 'nawm_mask', scalar_maps_for_nawmmask_node, 'mask_file')
                dwi_workflow.connect(dwi_metrics_node, 'out', scalar_maps_for_nawmmask_node, 'data_files')

            # 4. NAWM without tract mask
            if self.dwi_t1w_register and fs_output_process and 'mrtrix3' in self.tractography:
                scalar_maps_for_nawm_without_tract_node = Node(CalculateScalarMaps(), name='scalar_maps_for_nawm_without_tract')
                scalar_maps_for_nawm_without_tract_node.inputs.colnames = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF", "GQI_GFA", "GQI_ISO", "GQI_QA", "CHIDIA"]
                scalar_maps_for_nawm_without_tract_node.inputs.roi_label = 1
                scalar_maps_for_nawm_without_tract_node.inputs.output_csv = os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-NAWMwithoutTract_desc-mean_dwimap.csv")
                dwi_workflow.connect(remove_tract_region_node, 'out_wm_mask', scalar_maps_for_nawm_without_tract_node, 'mask_file')
                dwi_workflow.connect(dwi_metrics_node, 'out', scalar_maps_for_nawm_without_tract_node, 'data_files')

            # visual pathway analysis: TDI image
            if self.visual_pathway_analysis:
                scalar_maps_for_vptdi_node = MapNode(CalculateTDIWeightedScalars(), name='scalar_maps_for_vptdi', iterfield=['weight_file', 'output_csv'])
                scalar_maps_for_vptdi_node.inputs.colnames = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF", "GQI_GFA", "GQI_ISO", "GQI_QA", "CHIDIA", "WMHprobmap"]
                dwi_workflow.connect(vp_dwi_metrics_node, 'out', scalar_maps_for_vptdi_node, 'data_files')
                dwi_workflow.connect(tt_to_tdi_node, 'out_tdi', scalar_maps_for_vptdi_node, 'weight_file')
                scalar_maps_for_vptdi_node.inputs.output_csv = [
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_label-OT_desc-TDIweighted_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_label-OT_desc-TDIweighted_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_label-OR_desc-TDIweighted_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_label-OR_desc-TDIweighted_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-L_label-ML_desc-TDIweighted_dwimap.csv"),
                    os.path.join(dwi_metrics_output_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_hemi-R_label-ML_desc-TDIweighted_dwimap.csv")
                ]
                scalar_maps_for_vptdi_node.inputs.background_value = -1

        return dwi_workflow
    
    def extract_results(self):
        def _read_one_row_mask_metrics(csv_path):
            """
            Read a single-row metrics CSV produced by CalculateScalarMaps.
            Returns a dict with keys: FA, MD, FW (MarkVCID2), ODI, ICVF, ISOVF
            Missing file or missing columns -> np.nan
            """
            keys = ["FA", "MD", "FW (MarkVCID2)", "ODI", "ICVF", "ISOVF"]
            out = {k: np.nan for k in keys}

            if csv_path is None or not os.path.exists(csv_path):
                return out

            df = pd.read_csv(csv_path)
            if df.shape[0] < 1:
                return out

            row0 = df.iloc[0]
            for k in keys:
                if k in df.columns:
                    out[k] = row0[k]
            return out


        def _safe_get(d, k):
            v = d.get(k, np.nan)
            try:
                return float(v)
            except Exception:
                return np.nan
    
        os.makedirs(self.output_path, exist_ok=True)

        dwi_output_path = self.extract_from

        # DTI-ALPS
        alps_columns = ['Subject', 'Session', 'ALPS_L', 'ALPS_R', 'ALPS_mean']
        alps_results_df = pd.DataFrame(columns=alps_columns)

        # PSMD
        psmd_columns = ['Subject', 'Session', 'PSMD', 'PSMD_Left', 'PSMD_Right']
        psmd_results_df = pd.DataFrame(columns=psmd_columns)

        # PVeD
        pved_columns = ['Subject', 'Session', 'PVeD', 'PVeD_L', 'PVeD_R', 'QA_index']
        pved_results_df = pd.DataFrame(columns=pved_columns)
        
        # Track-based DWI metrics
        track_dwi_metrics_columns = ['Subject', 'Session', 'Track_DTI_FA', 'Track_DTI_MD', 'Track_SS_FW', 'Track_DTI_FW'] # SS for single shell
        track_dwi_metrics_df = pd.DataFrame(columns=track_dwi_metrics_columns)

        # Mask-based DWI metrics
        mask_tensor_metrics_columns = ['Subject', 'Session',
                                      'FA_seedmask', 'MD_seedmask', 'markvcid_FW_seedmask',
                                      'FA_WMH', 'MD_WMH', 'markvcid_FW_WMH',
                                      'FA_NAWM', 'MD_NAWM', 'markvcid_FW_NAWM']
        mask_tensor_metrics_df = pd.DataFrame(columns=mask_tensor_metrics_columns)

        mask_noddi_metrics_columns = ['Subject', 'Session',
                                     'ODI_seedmask', 'ICVF_seedmask', 'ISOVF_seedmask',
                                     'ODI_WMH', 'ICVF_WMH', 'ISOVF_WMH',
                                     'ODI_NAWM', 'ICVF_NAWM', 'ISOVF_NAWM']
        mask_noddi_metrics_df = pd.DataFrame(columns=mask_noddi_metrics_columns)

        # Surface parameters
        surface_parameters_df = pd.DataFrame()
        mirror_surface_parameters_df = pd.DataFrame()

        # Assume have sub + ses
        # Loop through all subjects (start with sub-) and sessions (start with ses-)
        print(f"Reading results from {dwi_output_path}...")
        for subject_folder in os.listdir(dwi_output_path):
            if not subject_folder.startswith('sub-'):
                continue

            subject_id = subject_folder.split('-')[1]
            subject_folder_path = os.path.join(dwi_output_path, subject_folder)

            for session_folder in os.listdir(subject_folder_path):
                if not session_folder.startswith('ses-'):
                    continue

                session_id = session_folder.split('-')[1]
                session_path = os.path.join(subject_folder_path, session_folder)

                dti_alps_csv = os.path.join(session_path, 'DTI-ALPS', 'alps.stat', 'alps.csv')
                psmd_txt = os.path.join(session_path, 'psmd', 'psmd_out.txt')
                pved_csv = os.path.join(session_path, 'PVeD', 'PVeD_metrics.csv')

                tract_fa_csv = os.path.join(session_path, 'dwi_metrics_stats', 'track_in_dti_FA_mean.csv')
                tract_md_csv = os.path.join(session_path, 'dwi_metrics_stats', 'track_in_dti_MD_mean.csv')
                tract_ss_fw_csv = os.path.join(session_path, 'dwi_metrics_stats', 'track_in_freewater_mean.csv')
                tract_dti_fw_csv = os.path.join(session_path, 'dwi_metrics_stats', 'track_in_dti_freewater_mean.csv')
                
                seedmask_scalarmaps_csv = os.path.join(session_path, 'dwi_metrics_stats', f"sub-{subject_id}_ses-{session_id}_label-seedmask_desc-mean_dwimap.csv")
                wmh_scalarmaps_csv = os.path.join(session_path, 'dwi_metrics_stats', f"sub-{subject_id}_ses-{session_id}_label-WMH_desc-mean_dwimap.csv")
                nawm_scalarmaps_csv = os.path.join(session_path, 'dwi_metrics_stats', f"sub-{subject_id}_ses-{session_id}_label-NAWM_desc-mean_dwimap.csv")

                surface_csv = os.path.join(session_path, 'surface_parameters.csv')
                mirror_surface_csv = os.path.join(session_path, 'mirror_surface_parameters.csv')
                
                # DTI-ALPS
                if os.path.exists(dti_alps_csv):
                    alps_df = pd.read_csv(dti_alps_csv)
                    if {'alps_L', 'alps_R', 'alps'}.issubset(alps_df.columns):
                        new_data = pd.DataFrame([{
                            'Subject': f"sub-{subject_id}",
                            'Session': f"ses-{session_id}",
                            'ALPS_L': alps_df['alps_L'].values[0],
                            'ALPS_R': alps_df['alps_R'].values[0],
                            'ALPS_mean': alps_df['alps'].values[0]
                        }])
                        alps_results_df = pd.concat([alps_results_df, new_data], ignore_index=True)
                
                # PSMD
                if os.path.exists(psmd_txt):
                    # if have only on number, assign to PSMD (XXXX)
                    # if have two number, assign to PSMD_L and PSMD_R (XXXX XXXX)
                    with open(psmd_txt, 'r') as f:
                        line = f.readline().strip()
                        parts = line.split()
                        if len(parts) == 1:
                            new_data = pd.DataFrame([{
                                'Subject': f"sub-{subject_id}",
                                'Session': f"ses-{session_id}",
                                'PSMD': float(parts[0]),
                                'PSMD_Left': np.nan,
                                'PSMD_Right': np.nan
                            }])
                        elif len(parts) == 2:
                            new_data = pd.DataFrame([{
                                'Subject': f"sub-{subject_id}",
                                'Session': f"ses-{session_id}",
                                'PSMD': np.nan,
                                'PSMD_Left': float(parts[0]),
                                'PSMD_Right': float(parts[1])
                            }])
                        else:
                            new_data = pd.DataFrame([{
                                'Subject': f"sub-{subject_id}",
                                'Session': f"ses-{session_id}",
                                'PSMD': np.nan,
                                'PSMD_Left': np.nan,
                                'PSMD_Right': np.nan
                            }])
                        psmd_results_df = pd.concat([psmd_results_df, new_data], ignore_index=True)

                # PVeD
                if os.path.exists(pved_csv):
                    pved_df = pd.read_csv(pved_csv)
                    if {'PVeD_total', 'PVeD_L', 'PVeD_R'}.issubset(pved_df.columns):
                        new_data = pd.DataFrame([{
                            'Subject': f"sub-{subject_id}",
                            'Session': f"ses-{session_id}",
                            'PVeD': pved_df['PVeD_total'].values[0],
                            'PVeD_L': pved_df['PVeD_L'].values[0],
                            'PVeD_R': pved_df['PVeD_R'].values[0],
                            'QA_index': pved_df['QA_index'].values[0]
                        }])
                        pved_results_df = pd.concat([pved_results_df, new_data], ignore_index=True)

                # Track-based DWI metrics
                new_data = pd.DataFrame([{
                    'Subject': f"sub-{subject_id}",
                    'Session': f"ses-{session_id}",
                    'Track_DTI_FA': pd.read_csv(tract_fa_csv)['mean'].values[0] if os.path.exists(tract_fa_csv) else None,
                    'Track_DTI_MD': pd.read_csv(tract_md_csv)['mean'].values[0] if os.path.exists(tract_md_csv) else None,
                    'Track_SS_FW': pd.read_csv(tract_ss_fw_csv)['mean'].values[0] if os.path.exists(tract_ss_fw_csv) else None,
                    'Track_DTI_FW': pd.read_csv(tract_dti_fw_csv)['mean'].values[0] if os.path.exists(tract_dti_fw_csv) else None
                }])
                track_dwi_metrics_df = pd.concat([track_dwi_metrics_df, new_data], ignore_index=True)

                # Mask-based DWI metrics
                seedmask_metrics = _read_one_row_mask_metrics(seedmask_scalarmaps_csv)
                wmh_metrics = _read_one_row_mask_metrics(wmh_scalarmaps_csv)
                nawm_metrics = _read_one_row_mask_metrics(nawm_scalarmaps_csv)

                tensor_row = {
                    "Subject": f"sub-{subject_id}",
                    "Session": f"ses-{session_id}",

                    "FA_seedmask": _safe_get(seedmask_metrics, "FA"),
                    "MD_seedmask": _safe_get(seedmask_metrics, "MD"),
                    "markvcid_FW_seedmask": _safe_get(seedmask_metrics, "FW (MarkVCID2)"),

                    "FA_WMH": _safe_get(wmh_metrics, "FA"),
                    "MD_WMH": _safe_get(wmh_metrics, "MD"),
                    "markvcid_FW_WMH": _safe_get(wmh_metrics, "FW (MarkVCID2)"),

                    "FA_NAWM": _safe_get(nawm_metrics, "FA"),
                    "MD_NAWM": _safe_get(nawm_metrics, "MD"),
                    "markvcid_FW_NAWM": _safe_get(nawm_metrics, "FW (MarkVCID2)"),
                }
                mask_tensor_metrics_df = pd.concat([mask_tensor_metrics_df, pd.DataFrame([tensor_row])], ignore_index=True)

                noddi_row = {
                    "Subject": f"sub-{subject_id}",
                    "Session": f"ses-{session_id}",

                    "ODI_seedmask": _safe_get(seedmask_metrics, "ODI"),
                    "ICVF_seedmask": _safe_get(seedmask_metrics, "ICVF"),
                    "ISOVF_seedmask": _safe_get(seedmask_metrics, "ISOVF"),

                    "ODI_WMH": _safe_get(wmh_metrics, "ODI"),
                    "ICVF_WMH": _safe_get(wmh_metrics, "ICVF"),
                    "ISOVF_WMH": _safe_get(wmh_metrics, "ISOVF"),

                    "ODI_NAWM": _safe_get(nawm_metrics, "ODI"),
                    "ICVF_NAWM": _safe_get(nawm_metrics, "ICVF"),
                    "ISOVF_NAWM": _safe_get(nawm_metrics, "ISOVF"),
                }
                mask_noddi_metrics_df = pd.concat([mask_noddi_metrics_df, pd.DataFrame([noddi_row])], ignore_index=True)

                # Surface parameters
                if os.path.exists(surface_csv):
                    df = pd.read_csv(surface_csv)
                    df.insert(0, 'DWI_Pipeline_ID', f'sub-{subject_id}_ses-{session_id}')
                    df.insert(1, 'Subject', f'sub-{subject_id}')
                    df.insert(2, 'Session', f'ses-{session_id}')
                    surface_parameters_df = pd.concat([surface_parameters_df, df], ignore_index=True)
                
                if os.path.exists(mirror_surface_csv):
                    df_mirror = pd.read_csv(mirror_surface_csv)
                    df_mirror.insert(0, 'DWI_Pipeline_ID', f'sub-{subject_id}_ses-{session_id}')
                    df_mirror.insert(1, 'Subject', f'sub-{subject_id}')
                    df_mirror.insert(2, 'Session', f'ses-{session_id}')
                    mirror_surface_parameters_df = pd.concat([mirror_surface_parameters_df, df_mirror], ignore_index=True)

        # Save results
        alps_output_excel = os.path.join(self.output_path, 'alps_results.xlsx')
        psmd_output_excel = os.path.join(self.output_path, 'psmd_results.xlsx')
        pved_output_excel = os.path.join(self.output_path, 'pved_results.xlsx')
        track_dwi_metrics_output_excel = os.path.join(self.output_path, 'track_dwi_metrics_results.xlsx')
        mask_tensor_metrics_output_csv = os.path.join(self.output_path, "mask_tensor_metrics_results.csv")
        mask_noddi_metrics_output_csv = os.path.join(self.output_path, "mask_noddi_metrics_results.csv")
        surface_output_excel = os.path.join(self.output_path, 'surface_parameters_results.xlsx')
        mirror_output_excel = os.path.join(self.output_path, 'mirror_surface_parameters_results.xlsx')
        if not alps_results_df.empty:
            alps_results_df.to_excel(alps_output_excel, header=True, index=False)
            print(f"DTI-ALPS results saved to {alps_output_excel}")
        else:
            print("No DTI-ALPS results found.")
        if not psmd_results_df.empty:
            psmd_results_df.to_excel(psmd_output_excel, header=True, index=False)
            print(f"PSMD results saved to {psmd_output_excel}")
        else:
            print("No PSMD results found.")
        if not pved_results_df.empty:
            pved_results_df.to_excel(pved_output_excel, header=True, index=False)
            print(f"PVeD results saved to {pved_output_excel}")
        else:
            print("No PVeD results found.")
        if not track_dwi_metrics_df.empty:
            track_dwi_metrics_df.to_excel(track_dwi_metrics_output_excel, header=True, index=False)
            print(f"Track-based DWI metrics results saved to {track_dwi_metrics_output_excel}")
        else:
            print("No track-based DWI metrics results found.")
        if not mask_tensor_metrics_df.empty:
            mask_tensor_metrics_df.to_csv(mask_tensor_metrics_output_csv, index=False)
            print(f"Mask-based tensor metrics saved to {mask_tensor_metrics_output_csv}")
        else:
            print("No mask-based tensor metrics results found.")

        if not mask_noddi_metrics_df.empty:
            mask_noddi_metrics_df.to_csv(mask_noddi_metrics_output_csv, index=False)
            print(f"Mask-based NODDI metrics saved to {mask_noddi_metrics_output_csv}")
        else:
            print("No mask-based NODDI metrics results found.")

        # Save surface parameters
        if 'session' in surface_parameters_df.columns:
            surface_parameters_df.drop(columns=['session'], inplace=True)
        if 'session' in mirror_surface_parameters_df.columns:
            mirror_surface_parameters_df.drop(columns=['session'], inplace=True)

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