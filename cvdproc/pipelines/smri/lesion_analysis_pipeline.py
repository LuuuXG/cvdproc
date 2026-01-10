import os
import subprocess
import nibabel as nib
import scipy.ndimage
import numpy as np
import pandas as pd
from nipype.interfaces.fsl import FLIRT
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from cvdproc.pipelines.smri.lesion_preprocess.lesion_preprocess import LeftRightLesionFill, SymmetricMniLesionFill, LesionSizeAnalysis, LIT, LesionFillBidsJson

from cvdproc.config.paths import get_package_path
from cvdproc.bids_data.rename_bids_file import rename_bids_file

class LesionAnalysisPipeline:
    def __init__(self,
                 subject: object,
                 session: object,
                 output_path: str,
                 use_which_t1w: str = None,
                 use_which_lesion_mask: str = None,
                 lesion_fill: bool = False,
                 lesion_fill_method: str = 'LIT',
                 out_contra_mask: bool = False,
                 lesion_size_analysis: bool = True,
                 normalize: bool = False,
                 extract_from: str = None,
                 **kwargs):
        """
        Lesion filling pipeline.

        Args:
            subject (object): Subject object
            session (object): Session object
            output_path (str): Output path
            use_which_t1w (str, optional): Which T1w file to use. Defaults to None.
            use_which_lesion_mask (str, optional): Which lesion mask (T1w space) to use. Defaults to None. Seeks for files in `<bids_dir>/derivatives/lesion_mask/sub-<subject_id>/ses-<session_id>`
            lesion_fill (bool, optional): Whether to perform lesion filling. Defaults to False. Need to be True if `out_contra_mask` is True.
            lesion_fill_method (str, optional): Lesion filling method. Defaults to 'LIT'. can be 'left-right', 'sym_MNI', or 'LIT'.
            out_contra_mask (bool, optional): Whether to save contralateral lesion mask when using 'sym_MNI' method. Defaults to False.
            lesion_size_analysis (bool, optional): Whether to perform lesion size analysis (MUST be one cluster). Defaults to True.
            normalize (bool, optional): Whether to normalize lesion mask to MNI space. Defaults to False.
            extract_from (str, optional): Folder name to extract results from. Defaults to None.
        """
        self.subject = subject
        self.session = session
        #self.output_path = output_path
        self.use_which_t1w = use_which_t1w
        self.use_which_lesion_mask = use_which_lesion_mask
        self.lesion_fill = lesion_fill
        self.lesion_fill_method = lesion_fill_method
        self.out_contra_mask = out_contra_mask
        self.lesion_size_analysis = lesion_size_analysis
        self.normalize = normalize
        self.extract_from = extract_from

        self.output_path = self.session._find_output('lesion_mask') if self.session is not None else output_path

    def check_data_requirements(self):
        return self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # get T1w image
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            # ensure that there is only 1 suitable file
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]
        print(f"[LESION FILLING] Using T1w file: {t1w_file}")

        # get lesion mask
        lesion_mask_dir = self.session._find_output('lesion_mask')
        if lesion_mask_dir is None:
            raise FileNotFoundError(f"No lesion mask derivative found in expected directory: {lesion_mask_dir}")
        lesion_mask_files = sorted([os.path.join(lesion_mask_dir, f) for f in os.listdir(lesion_mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        if self.use_which_lesion_mask:
            lesion_mask_files = [f for f in lesion_mask_files if self.use_which_lesion_mask in f]
            # ensure that there is only 1 suitable file
            if len(lesion_mask_files) != 1:
                raise FileNotFoundError(f"No specific lesion mask file found for {self.use_which_lesion_mask} or more than one found.")
            lesion_mask_file = lesion_mask_files[0]
        else:
            print("No specific lesion mask file selected. Using the first one.")
            lesion_mask_files = [lesion_mask_files[0]]
            lesion_mask_file = lesion_mask_files[0]
        print(f"[LESION FILLING] Using lesion mask file: {lesion_mask_file}")

        # get nonlinear warp files if needed
        # if self.normalize:
        #     target_warp_fwd = os.path.join(self.session.xfm_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz')
        #     if not os.path.exists(target_warp_fwd):
        #         print(f"[WARNING] Non-linear warp file not found: {target_warp_fwd}. Normalization will be skipped.")
        #         self.normalize = False
        
        if self.session.xfm_dir is None:
            xfm_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}' if self.session else '')
        else:
            xfm_dir = self.session.xfm_dir

        # Create the workflow
        lesion_filling_wf = Workflow(name='lesion_filling_workflow')

        inputnode = Node(IdentityInterface(fields=['t1w', 'lesion_mask']),
                         name='inputnode')
        inputnode.inputs.t1w = t1w_file
        inputnode.inputs.lesion_mask = lesion_mask_file

        gather_lesionfilled_t1w_node = Node(IdentityInterface(fields=['lesion_filled_t1w']),
                                                name='gather_lesionfilled_t1w_node')
        if self.lesion_fill:
            if self.lesion_fill_method == 'left-right':
                lesion_fill_node = Node(LeftRightLesionFill(),
                                        name='lesion_fill_node')
                
                lesion_filling_wf.connect([(inputnode, lesion_fill_node, [('t1w', 't1w_file'),
                                                                        ('lesion_mask', 'lesion_mask')])])
                lesion_fill_node.inputs.output_file = os.path.join(self.subject.bids_dir, f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}' if self.session else '',
                                                                'anat', rename_bids_file(t1w_file, {'desc': 'lesionfilled'}, 'T1w', '.nii.gz'))
                lesion_fill_node.inputs.output_json = lesion_fill_node.inputs.output_file.replace('.nii.gz', '.json')

                lesion_filling_wf.connect(lesion_fill_node, 'output_file', gather_lesionfilled_t1w_node, 'lesion_filled_t1w')
            elif self.lesion_fill_method == 'LIT':
                lit_lesion_fill_node = Node(LIT(),
                                            name='lit_lesion_fill_node')
                lesion_filling_wf.connect([(inputnode, lit_lesion_fill_node, [('t1w', 'input_image'),
                                                                             ('lesion_mask', 'mask_image')])])
                lit_lesion_fill_node.inputs.output_directory = os.path.join(self.output_path, 'LIT')
                lit_lesion_fill_node.inputs.lit_data_dir = get_package_path('data', 'models', 'lit')
                lit_lesion_fill_node.inputs.output_image = os.path.join(self.subject.bids_dir, f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}' if self.session else '',
                                                                       'anat', rename_bids_file(t1w_file, {'desc': 'lesionfilled'}, 'T1w', '.nii.gz'))
                
                lit_create_json_node = Node(LesionFillBidsJson(),
                                            name='lit_create_json_node')
                lesion_filling_wf.connect(inputnode, 'lesion_mask', lit_create_json_node, 'lesion_mask')
                lesion_filling_wf.connect(inputnode, 't1w', lit_create_json_node, 't1w_file')
                lit_create_json_node.inputs.json_out = os.path.join(self.subject.bids_dir, f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}' if self.session else '',
                                                                   'anat', rename_bids_file(t1w_file, {'desc': 'lesionfilled'}, 'T1w', '.json'))
                lit_create_json_node.inputs.bids_dir = self.subject.bids_dir

                # make sure the inpainted T1w has the same dimension with orig T1w
                resample_node = Node(FLIRT(), name='lit_resample')
                lesion_filling_wf.connect([(inputnode, resample_node, [('t1w', 'reference')]),
                                             (lit_lesion_fill_node, resample_node, [('output_image', 'in_file')])])
                resample_node.inputs.out_file = os.path.join(self.subject.bids_dir, f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}' if self.session else '',
                                                                       'anat', rename_bids_file(t1w_file, {'desc': 'lesionfilled'}, 'T1w', '.nii.gz'))
                resample_node.inputs.args = '-applyxfm -usesqform -interp trilinear'

                lesion_filling_wf.connect(resample_node, 'out_file', gather_lesionfilled_t1w_node, 'lesion_filled_t1w')

            if self.lesion_fill_method == 'sym_MNI' or self.out_contra_mask:
                # get original desc- entity of lesion mask file
                lesion_mask_basename = os.path.basename(lesion_mask_file)
                desc_entity = ''
                if 'desc-' in lesion_mask_basename:
                    parts = lesion_mask_basename.split('_')
                    for part in parts:
                        if part.startswith('desc-'):
                            desc_entity = part.replace('desc-', '')
                            break
                
                # contra: add 'contra' in original entity
                contra_entity = desc_entity + 'contra'

                sym_mni_lesion_fill_node = Node(SymmetricMniLesionFill(),
                                               name='sym_mni_lesion_fill_node')
                lesion_filling_wf.connect([(inputnode, sym_mni_lesion_fill_node, [('t1w', 't1w_file'),
                                                                                  ('lesion_mask', 'lesion_mask')])])
                sym_mni_lesion_fill_node.inputs.mni_template = get_package_path('data', 'standard', 'MNI152', 'mni_icbm152_nlin_sym_09a_nifti', 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
                sym_mni_lesion_fill_node.inputs.warp_fwd = os.path.join(xfm_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin2009aSym_warp.nii.gz')
                sym_mni_lesion_fill_node.inputs.warp_inv = os.path.join(xfm_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin2009aSym_to-T1w_warp.nii.gz')
                sym_mni_lesion_fill_node.inputs.t1_mni = os.path.join(xfm_dir, rename_bids_file(t1w_file, {'space': 'MNI152NLin2009aSym'}, 'T1w', '.nii.gz'))
                sym_mni_lesion_fill_node.inputs.filled_t1 = os.path.join(self.subject.bids_dir, f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}' if self.session else '',
                                                                       'anat', rename_bids_file(t1w_file, {'desc': 'lesionfilled'}, 'T1w', '.nii.gz'))
                sym_mni_lesion_fill_node.inputs.contra_mask = os.path.join(lesion_mask_dir, rename_bids_file(lesion_mask_file, {'desc': contra_entity}, 'mask', '.nii.gz'))
                sym_mni_lesion_fill_node.inputs.bids_dir = self.subject.bids_dir

                if self.out_contra_mask:
                    sym_mni_lesion_fill_node.inputs.contra_only = True
                else:
                    sym_mni_lesion_fill_node.inputs.contra_only = False
                    lesion_filling_wf.connect(sym_mni_lesion_fill_node, 'filled_t1', gather_lesionfilled_t1w_node, 'lesion_filled_t1w')
            
        if self.lesion_size_analysis:
            # check whether only have ONE cluster
            lesion_img = nib.load(lesion_mask_file)
            lesion_data = lesion_img.get_fdata()
            labeled_array, num_features = scipy.ndimage.label(lesion_data)
            if num_features != 1:
                #raise ValueError(f"Lesion size analysis requires exactly one lesion cluster, but found {num_features} clusters.")

                # skip
                print(f"[WARNING] Lesion size analysis requires exactly one lesion cluster, but found {num_features} clusters. Skipping lesion size analysis.")
            else:
                lesion_size_analysis_node = Node(LesionSizeAnalysis(),
                                                name='lesion_size_analysis_node')
                lesion_filling_wf.connect([(inputnode, lesion_size_analysis_node, [('lesion_mask', 'lesion_mask')])])
                lesion_size_analysis_node.inputs.out_csv = os.path.join(self.output_path, f'lesion_metrics.csv')
            
        if self.normalize:
            # add normalization step for lesion mask
            from cvdproc.pipelines.common.register import MRIConvertApplyWarp
            from cvdproc.pipelines.common.register import SynthmorphNonlinear

            target_warp_fwd = os.path.join(xfm_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz')

            if os.path.exists(target_warp_fwd):
                normalize_lesion_node = Node(MRIConvertApplyWarp(),
                                            name='normalize_lesion_node')
                lesion_filling_wf.connect([(inputnode, normalize_lesion_node, [('lesion_mask', 'input_image')])])
                normalize_lesion_node.inputs.warp_image = target_warp_fwd
                normalize_lesion_node.inputs.output_image = os.path.join(self.output_path, rename_bids_file(lesion_mask_file, {'space': 'MNI152NLin6ASym'}, 'mask', '.nii.gz'))
                normalize_lesion_node.inputs.interp = 'nearest'
            else:
                register_node = Node(SynthmorphNonlinear(), name='synthmorph_register_for_normalization')
                lesion_filling_wf.connect([(gather_lesionfilled_t1w_node, register_node, [('lesion_filled_t1w', 't1')])])
                register_node.inputs.mni_template = get_package_path('data', 'standard', 'MNI152', 'MNI152_T1_1mm_brain.nii.gz')
                register_node.inputs.t1_mni_out = os.path.join(xfm_dir, rename_bids_file(t1w_file, {'space': 'MNI152NLin6ASym'}, 'T1w', '.nii.gz'))
                register_node.inputs.t1_2_mni_warp = target_warp_fwd
                register_node.inputs.mni_2_t1_warp = os.path.join(xfm_dir, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')

                normalize_lesion_node = Node(MRIConvertApplyWarp(),
                                            name='normalize_lesion_node')
                lesion_filling_wf.connect([(inputnode, normalize_lesion_node, [('lesion_mask', 'input_image')])])
                lesion_filling_wf.connect(register_node, 't1_2_mni_warp', normalize_lesion_node, 'warp_image')
                normalize_lesion_node.inputs.output_image = os.path.join(self.output_path, rename_bids_file(lesion_mask_file, {'space': 'MNI152NLin6ASym'}, 'mask', '.nii.gz'))
                normalize_lesion_node.inputs.interp = 'nearest'

        return lesion_filling_wf
    
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        lesion_analysis_output_dir = self.extract_from

        lesion_analysis_columes = [
            'Subject',
            'Session',
            'Lesion_Volume_mm3',
            'Lesion_Max_Diameter_tra_mm',
            'Lesion_Max_Diameter_3d_mm'
        ]

        lesion_analysis_df = pd.DataFrame(columns=lesion_analysis_columes)

        for subject_folder in os.listdir(lesion_analysis_output_dir):
            subject_path = os.path.join(lesion_analysis_output_dir, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            for session_folder in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session_folder)
                if not os.path.isdir(session_path):
                    continue
                
                lesion_metrics_file = os.path.join(session_path, 'lesion_metrics.csv')
                if not os.path.exists(lesion_metrics_file):
                    print(f"[WARNING] Lesion metrics file not found: {lesion_metrics_file}")
                    continue
                
                # In order: Subject, Session, Lesion_Volume_mm3, Lesion_Max_Diameter_tra_mm, Lesion_Max_Diameter_3d_mm
                # need to rename csv columns
                temp_df = pd.read_csv(lesion_metrics_file)
                temp_df.rename(columns={
                    'volume_mm3': 'Lesion_Volume_mm3',
                    'largest_diameter_axial_mm': 'Lesion_Max_Diameter_tra_mm',
                    'largest_diameter_3d_mm': 'Lesion_Max_Diameter_3d_mm'
                }, inplace=True)
                temp_df.insert(0, 'Session', session_folder)
                temp_df.insert(0, 'Subject', subject_folder)

                lesion_analysis_df = pd.concat([lesion_analysis_df, temp_df], ignore_index=True)
        # Save combined results
        combined_results_file = os.path.join(self.output_path, 'lesion_size_analysis_summary.csv')
        lesion_analysis_df.to_csv(combined_results_file, index=False)
        print(f"[INFO] Lesion size analysis summary saved to: {combined_results_file}")