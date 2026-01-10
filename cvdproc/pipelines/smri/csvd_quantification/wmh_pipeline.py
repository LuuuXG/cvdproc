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
from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from ....bids_data.rename_bids_file import rename_bids_file

from .wmh.wmh_seg_nipype import LSTSegmentation, LSTAI, WMHSynthSegSingle, PrepareTrueNetData, PrepareTrueNetData2, TrueNetEvaluate, TrueNetPostProcess
from .wmh.wmh_location_nipype import FazekasClassification
from .wmh.wmh_shape_nipype import WMHShape

from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.fsl.preprocess import ApplyXFM
from ..fsl.fsl_anat_nipype import FSLANAT
from ..fsl.distancemap_nipype import DistanceMap
from ...common.copy_file import CopyFileCommandLine
from cvdproc.pipelines.smri.fsl.fslmaths_thr import FSLMathsUnderThr, FSLMathsThr
from cvdproc.pipelines.smri.fsl.make_bianca_mask_nipype import MakeBIANCAMask
from cvdproc.pipelines.common.register import SynthmorphNonlinear, MRIConvertApplyWarp, TwoStepNormalization
from ..freesurfer.synthSR import SynthSR
from ..freesurfer.synthstrip import SynthStrip
from cvdproc.pipelines.smri.freesurfer.synthseg import SynthSeg
from ...common.register import ModalityRegistration, Tkregister2fs2t1w

from cvdproc.pipelines.common.extract_region import ExtractRegion
from cvdproc.pipelines.common.calculate_volume import CalculateVolume
from cvdproc.pipelines.common.calculate_roi_volume import CalculateROIVolume

from cvdproc.config.paths import get_package_path

class WMHSegmentationPipeline:
    """
    WMH Segmentation and Quantification Pipeline
    """
    def __init__(self, 
                 subject, 
                 session, 
                 output_path, 
                 use_which_t1w: str = None,
                 use_which_flair: str = None,
                 seg_method: str = 'LST',
                 seg_threshold: float = 0.5,
                 location_method: list = ['Fazekas'],
                 ventmask_method: str = 'SynthSeg',
                 use_bianca_mask: bool = False,
                 normalize_to_mni: bool = False,
                 shape_features: bool = False,
                 **kwargs):
        """
        WMH Segmentation and Quantification Pipeline
        
        Args:
            subject: BIDSSubject object
            session: BIDSSession object
            output_path: output directory for the pipeline
            use_which_t1w: specific string to select T1w image, e.g. 'acq-highres'. If None, T1w image is not used
            use_which_flair: specific string to select FLAIR image, e.g. 'acq-highres'. If None, FLAIR image is not used
            seg_method: WMH segmentation method, one of ['LST', 'LSTAI', 'WMHSynthSeg', 'truenet']
            seg_threshold: threshold for WMH segmentation (not used for WMHSynthSeg method)
            location_method: list of location method, subset of ['Fazekas', 'bullseye', 'shiva', 'McDonald', 'JHU']
            ventmask_method: method to get ventricle mask, one of ['SynthSeg']
            use_bianca_mask: whether to use BIANCA white matter mask to constrain WMH segmentation
            normalize_to_mni: whether to normalize the WMH mask to MNI space
            shape_features: whether to calculate shape features for each WMH cluster (need normalize_to_mni=True and location_method contains 'Fazekas')
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.use_which_t1w = use_which_t1w
        self.use_which_flair = use_which_flair
        self.seg_method = seg_method
        self.seg_threshold = seg_threshold
        self.location_method = location_method
        self.ventmask_method = ventmask_method
        self.use_bianca_mask = use_bianca_mask
        self.normalize_to_mni = normalize_to_mni
        self.shape_features = shape_features
        self.kwargs = kwargs

    def check_data_requirements(self):
        """
        Check data requirements for the pipeline (At least one FLAIR image or T1w image)
        :return: bool
        """
        return self.session.get_flair_files() or self.session.get_t1w_files()
    
    def create_workflow(self):
        print("[WMH Pipeline] IMPORTANT:")
        print("[WMH Pipeline] If provide T1w image, please make sure it is 3D-T1w image!")

        if self.session.get_flair_files():
            flair_files = self.session.get_flair_files()
            if self.use_which_flair:
                flair_files = [f for f in flair_files if self.use_which_flair in f]
                if len(flair_files) != 1:
                    print(f"Warning: No specific FLAIR file found for {self.use_which_flair} or more than one found. Assume you don't want to use FLAIR image.")
                else:
                    flair_file = flair_files[0]
            else:
                flair_files = [flair_files[0]]
                flair_file = flair_files[0]
                print(f"No specific FLAIR file selected. Using the first one.")
        else:
            flair_file = None
        
        if self.session.get_t1w_files():
            t1w_files = self.session.get_t1w_files()
            if self.use_which_t1w:
                t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
                if len(t1w_files) != 1:
                    #raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
                    print(f"Warning: No specific T1w file found for {self.use_which_t1w} or more than one found. Assume you don't want to use T1w image.")
                    t1w_file = None
                else:
                    t1w_file = t1w_files[0]
            else:
                t1w_files = [t1w_files[0]]
                t1w_file = t1w_files[0]
                print(f"No specific T1w file selected. Using the first one (Assume it is 3D-T1w).")
        else:
            t1w_file = None

        if flair_file is None and t1w_file is None:
            raise FileNotFoundError("No FLAIR or T1w file found for this session. At least one FLAIR or T1w image is required.")
        if flair_file is None:
            flair_file = ''         
        if t1w_file is None:
            t1w_file = ''

        #########################################
        # Display selected files and parameters #
        #########################################
        if flair_file != '':
            print(f"[WMH Pipeline] Using FLAIR file: {flair_file}")
        else:
            print(f"[WMH Pipeline] No FLAIR file provided. Proceeding with T1w only.")
        if t1w_file != '':
            print(f"[WMH Pipeline] Using T1w file: {t1w_file}")
        else:
            print(f"[WMH Pipeline] No T1w file provided. Proceeding with FLAIR only.")
        # seg_method must be one of ['LST', 'LSTAI', 'WMHSynthSeg', 'truenet']
        if self.seg_method not in ['LST', 'LSTAI', 'WMHSynthSeg', 'truenet']:
            raise ValueError(f"seg_method must be one of ['LST', 'LSTAI', 'WMHSynthSeg', 'truenet'], but got {self.seg_method}.")
        print(f"[WMH Pipeline] Using segmentation method: {self.seg_method}")
        # the following seg_methods need T1w image: 'LSTAI'
        if self.seg_method == 'LSTAI' and t1w_file == '':
            raise ValueError(f"seg_method {self.seg_method} requires T1w image, but no T1w image provided.")
        # seg_threshold is not used when seg_method is WMHSynthSeg
        if self.seg_method != 'WMHSynthSeg':
            print(f"[WMH Pipeline] Using segmentation threshold: {self.seg_threshold}")
        else:
            print(f"[WMH Pipeline] Segmentation threshold is not used for WMHSynthSeg method.")
        # location_method must be subset of ['Fazekas', 'bullseye', 'McDonald']
        for method in self.location_method:
            if method not in ['Fazekas', 'bullseye', 'McDonald', 'shiva', 'JHU']:
                raise ValueError(f"location_method must be subset of ['Fazekas', 'bullseye', 'McDonald', 'shiva'], but got {self.location_method}.")
        print(f"[WMH Pipeline] Using location method: {self.location_method}")
        if self.normalize_to_mni:
            print(f"[WMH Pipeline] Will normalize WMH mask to MNI space.")
        
        os.makedirs(self.output_path, exist_ok=True)

        wmh_workflow = Workflow(name="WMHSegmentationPipeline")

        inputnode = Node(IdentityInterface(fields=['flair', 't1w', 'seg_threshold', 'subject_id',
                                                   'subject', 'session']), name='inputnode')
        inputnode.inputs.flair = flair_file
        inputnode.inputs.t1w = t1w_file
        inputnode.inputs.seg_threshold = self.seg_threshold
        inputnode.inputs.subject_id = f"ses-{self.session.session_id}" # used to connect bullseye nested workflow
        inputnode.inputs.subject = f"sub-{self.subject.subject_id}"
        inputnode.inputs.session = f"ses-{self.session.session_id}"

        # ------------------------
        # Part 1: WMH Segmentation
        # ------------------------
        # Segment WMH in T1w space if T1w provided, otherwise in FLAIR space
        # target output in this stage
        thr_string = f'Thr{self.seg_threshold:.2f}'.replace('.', 'p')
        if self.seg_method == 'WMHSynthSeg':
            thr_string = ''  # threshold is not used for WMHSynthSeg

        binarized_wmh_filename_flair = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-WMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"
        binarized_wmh_filename_flair_nothr = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-WMH_desc-{self.seg_method}_mask.nii.gz"
        binarized_wmh_filename_t1w = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"
        binarized_wmh_filename_t1w_nothr = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-{self.seg_method}_mask.nii.gz"
        probmap_filename_flair = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-WMH_desc-{self.seg_method}_probmap.nii.gz"
        probmap_filename_t1w = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-{self.seg_method}_probmap.nii.gz"

        if t1w_file == '':
            binarized_wmh_filename_t1w = ''
            probmap_filename_t1w = ''
        if flair_file == '':
            binarized_wmh_filename_flair = ''
            probmap_filename_flair = ''

        wmh_mask_node = Node(IdentityInterface(fields=["wmh_mask_flair", "wmh_probmap_flair",
                                                       "wmh_mask_t1w", "wmh_probmap_t1w"]), name="wmh_mask_node")
        
        # Preprocess
        # register FLAIR to T1w if T1w provided
        if flair_file != '' and t1w_file != '':
            xfm_output_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
            os.makedirs(xfm_output_dir, exist_ok=True)
            flair_in_t1w_node = Node(IdentityInterface(fields=["flair_in_t1w", "flair_to_t1w_xfm", "t1w_to_flair_xfm"]), name="flair_in_t1w_node")
            flair_in_t1w = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', rename_bids_file(flair_file, {'space': 'T1w', 'desc': 'brain'}, 'FLAIR', '.nii.gz'))
            flair_to_t1w_xfm = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-FLAIR_to-T1w_xfm.mat")
            t1w_to_flair_xfm = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-FLAIR_xfm.mat")
            
            if os.path.exists(flair_in_t1w) and os.path.exists(flair_to_t1w_xfm) and os.path.exists(t1w_to_flair_xfm):
                print(f"[WMH Pipeline] Found existing registered FLAIR in T1w space: {flair_in_t1w}, and existing transform files. Will use them directly.")
                flair_in_t1w_node.inputs.flair_in_t1w = flair_in_t1w
                flair_in_t1w_node.inputs.flair_to_t1w_xfm = flair_to_t1w_xfm
                flair_in_t1w_node.inputs.t1w_to_flair_xfm = t1w_to_flair_xfm
            else:
                flair_to_t1w_reg = Node(ModalityRegistration(), name='flair_to_t1w_reg')
                wmh_workflow.connect(inputnode, 'flair', flair_to_t1w_reg, 'image_source')
                flair_to_t1w_reg.inputs.image_source_strip = 0  # assume FLAIR is not skull-stripped
                wmh_workflow.connect(inputnode, 't1w', flair_to_t1w_reg, 'image_target')
                flair_to_t1w_reg.inputs.image_target_strip = 0  # assume T1w is not skull-stripped
                flair_to_t1w_reg.inputs.flirt_direction = 1 # use T1w as reference
                flair_to_t1w_reg.inputs.output_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')
                flair_to_t1w_reg.inputs.registered_image_filename = rename_bids_file(flair_file, {'space': 'T1w', 'desc': 'brain'}, 'FLAIR', '.nii.gz')
                flair_to_t1w_reg.inputs.source_to_target_mat_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-FLAIR_to-T1w_xfm.mat"
                flair_to_t1w_reg.inputs.target_to_source_mat_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-FLAIR_xfm.mat"
                flair_to_t1w_reg.inputs.dof = 6
                    
                wmh_workflow.connect(flair_to_t1w_reg, 'output_image', flair_in_t1w_node, 'flair_in_t1w')
                wmh_workflow.connect(flair_to_t1w_reg, 'source_to_target_mat', flair_in_t1w_node, 'flair_to_t1w_xfm')
                wmh_workflow.connect(flair_to_t1w_reg, 'target_to_source_mat', flair_in_t1w_node, 't1w_to_flair_xfm')
        else:
            pass
            # currently, no preprocess is needed if only provide FLAIR image (2D/3D)
        
        # Skull-strip T1w if T1w provided
        if t1w_file != '':
            t1w_stripped_node = Node(IdentityInterface(fields=["t1w_stripped", "brain_mask"]), name="t1w_stripped_node")
            t1w_stripped = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', rename_bids_file(t1w_file, {'desc': 'brain'}, 'T1w', '.nii.gz'))
            brain_mask = os.path.join(os.path.dirname(t1w_stripped), rename_bids_file(t1w_file, {'desc': 'brain', 'space': 'T1w'}, 'mask', '.nii.gz'))
            if os.path.exists(t1w_stripped) and os.path.exists(brain_mask):
                print(f"[WMH Pipeline] Found existing skull-stripped T1w: {t1w_stripped}. Will use it directly.")
                t1w_stripped_node.inputs.t1w_stripped = t1w_stripped
                t1w_stripped_node.inputs.brain_mask = brain_mask
            else:
                t1w_synthstrip = Node(SynthStrip(), name='t1w_synthstrip')
                wmh_workflow.connect(inputnode, 't1w', t1w_synthstrip, 'image')
                t1w_synthstrip.inputs.out_file = t1w_stripped
                t1w_synthstrip.inputs.mask_file = os.path.join(os.path.dirname(t1w_stripped), rename_bids_file(t1w_file, {'desc': 'brain', 'space': 'T1w'}, 'mask', '.nii.gz'))
                t1w_synthstrip.inputs.no_csf = True
                
                wmh_workflow.connect(t1w_synthstrip, 'out_file', t1w_stripped_node, 't1w_stripped')
                wmh_workflow.connect(t1w_synthstrip, 'mask_file', t1w_stripped_node, 'brain_mask')
        
        # Determine whether need to run SynthSeg in prior
        # 1: Fazekas + SynthSeg ventricle mask
        # 2: shiva
        # 3: truenet
        run_synthseg = False
        synthseg_outfile = os.path.join(self.subject.bids_dir, 'derivatives', 'anat_seg', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_synthseg.nii.gz')
        if 'Fazekas' in self.location_method and self.ventmask_method == 'SynthSeg':
            run_synthseg = True
        if 'shiva' in self.location_method:
            run_synthseg = True
        if self.seg_method == 'truenet':
            run_synthseg = True
        if os.path.exists(synthseg_outfile):
            run_synthseg = False
            synthseg_img = nib.load(synthseg_outfile)
            if synthseg_img.shape != nib.load(t1w_file).shape:
                print(f"[WMH Pipeline] Sorry... SynthSeg output has different shape with current T1w image.")
                run_synthseg = True
        if run_synthseg:
            print(f"[WMH Pipeline] Will run SynthSeg on T1w image: {t1w_file}.")
            if t1w_file == '':
                raise ValueError("SynthSeg requires T1w image, but no T1w image provided.")
            synthseg_node = Node(SynthSeg(), name='synthseg')
            wmh_workflow.connect(inputnode, 't1w', synthseg_node, 'image')
            synthseg_node.inputs.out = synthseg_outfile
            synthseg_node.inputs.vol = os.path.join(self.subject.bids_dir, 'derivatives', 'anat_seg', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-synthseg_volume.csv')
            synthseg_node.inputs.robust = True
            synthseg_node.inputs.parc = True
            synthseg_node.inputs.keepgeom = True
        
        # check whether already have synthseg output
        check_synthseg = Node(IdentityInterface(fields=["synthseg_output"]), name="synthseg_output")
        if run_synthseg:
            wmh_workflow.connect(synthseg_node, 'out', check_synthseg, 'synthseg_output')
        else:
            print(f"[WMH Pipeline] Found existing SynthSeg output: {synthseg_outfile}. Will use it directly.")
            check_synthseg.inputs.synthseg_output = synthseg_outfile

        if self.seg_method == 'LST':
            # As inplement in the LST segmentation script, still use raw FLAIR image instead of registered one
            # FLAIR -> 'wmh_mask_t1w' and 'wmh_prob_t1w' will be empty ('')
            # FLAIR + T1w -> 'wmh_mask_flair' and 'wmh_probmap_flair' will be empty ('')
            lst_seg = Node(LSTSegmentation(), name='lst_seg')
            wmh_workflow.connect([
                (inputnode, lst_seg, [('flair', 'flair'),
                                      ('t1w', 't1w'),
                                      ('seg_threshold', 'threshold')]),
                (lst_seg, wmh_mask_node, [('wmh_mask_flair', 'wmh_mask_flair'),
                                          ('wmh_prob_flair', 'wmh_probmap_flair'),
                                          ('wmh_mask_t1w', 'wmh_mask_t1w'),
                                          ('wmh_prob_t1w', 'wmh_probmap_t1w')])
            ])
            lst_seg.inputs.WMH_mask_flair = os.path.join(self.output_path, binarized_wmh_filename_flair)
            lst_seg.inputs.WMH_mask_t1w = os.path.join(self.output_path, binarized_wmh_filename_t1w) if t1w_file != '' else ''
            lst_seg.inputs.WMH_prob_flair = os.path.join(self.output_path, probmap_filename_flair)
            lst_seg.inputs.WMH_prob_t1w = os.path.join(self.output_path, probmap_filename_t1w) if t1w_file != '' else ''
            lst_seg.inputs.spm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'matlab_toolbox', 'spm12')
            lst_seg.inputs.lst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'matlab_toolbox', 'LST')
            lst_seg.inputs.output_path = self.output_path
            lst_seg.inputs.script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'matlab', 'lst', 'wmh_seg_lst_script.m')
            if flair_file != '':
                lst_seg.inputs.FLAIR_in_T1w = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', rename_bids_file(flair_file, {'space': 'T1w'}, 'FLAIR', '.nii.gz'))
            else:
                lst_seg.inputs.FLAIR_in_T1w = ''
        elif self.seg_method == 'LSTAI':
            # LST-AI need 3D T1w image and whether 2D/3D FLAIR image
            lst_ai = Node(LSTAI(), name='lst_ai')
            lst_ai_outputdir = os.path.join(self.output_path, 'lst_ai')
            os.makedirs(lst_ai_outputdir, exist_ok=True)

            if flair_file != '' and t1w_file != '':
                wmh_workflow.connect(flair_in_t1w_node, 'flair_in_t1w', lst_ai, 'flair_img')
                wmh_workflow.connect(t1w_stripped_node, 't1w_stripped', lst_ai, 't1w_img')
                wmh_workflow.connect(inputnode, 'seg_threshold', lst_ai, 'threshold')
                lst_ai.inputs.output_dir = lst_ai_outputdir
                lst_ai.inputs.temp_dir = lst_ai_outputdir
                lst_ai.inputs.save_prob_map = True
                lst_ai.inputs.img_stripped = True
            else:
                raise ValueError("LST-AI method requires both FLAIR and T1w images.")
            
            copy_lst_ai_probmap_node = Node(CopyFileCommandLine(), name="copy_lst_ai_probmap")
            copy_lst_ai_probmap_node.inputs.output_file = os.path.join(self.output_path, probmap_filename_t1w)
            wmh_workflow.connect(lst_ai, 'wmh_prob_map', copy_lst_ai_probmap_node, 'input_file')

            copy_lst_ai_wmhmask_node = Node(CopyFileCommandLine(), name="copy_lst_ai_wmhmask")
            copy_lst_ai_wmhmask_node.inputs.output_file = os.path.join(self.output_path, binarized_wmh_filename_t1w)
            wmh_workflow.connect(lst_ai, 'wmh_mask', copy_lst_ai_wmhmask_node, 'input_file')

            wmh_workflow.connect(copy_lst_ai_probmap_node, 'output_file', wmh_mask_node, 'wmh_probmap_t1w')
            wmh_workflow.connect(copy_lst_ai_wmhmask_node, 'output_file', wmh_mask_node, 'wmh_mask_t1w')
        elif self.seg_method == 'WMHSynthSeg':
            # WMHSynthSeg use only 1 image as input (FLAIR or T1w)
            wmh_synthseg = Node(WMHSynthSegSingle(), name='wmh_synthseg')

            if t1w_file != '' and flair_file != '':
                print("[WMH Pipeline] WMHSynthSeg can only take one image as input. Using registered FLAIR image as input.")
                wmh_workflow.connect(flair_in_t1w_node, 'flair_in_t1w', wmh_synthseg, 'input')
                wmh_synthseg.inputs.output = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_WMHSynthSeg.nii.gz")
                wmh_synthseg.inputs.csv_vols = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-WMHSynthSeg_volume.csv")
                wmh_synthseg.inputs.device = 'cpu'
                wmh_synthseg.inputs.threads = 8
                wmh_synthseg.inputs.save_lesion_probabilities = True
                wmh_synthseg.inputs.prob_filepath = os.path.join(self.output_path, probmap_filename_t1w)
                wmh_synthseg.inputs.wmh_filepath = os.path.join(self.output_path, binarized_wmh_filename_t1w_nothr)

                wmh_workflow.connect(wmh_synthseg, 'wmh_filepath', wmh_mask_node, 'wmh_mask_t1w')
                wmh_workflow.connect(wmh_synthseg, 'prob_filepath', wmh_mask_node, 'wmh_probmap_t1w')
            elif t1w_file != '' and flair_file == '':
                print("[WMH Pipeline] Using T1w image for WMHSynthSeg.")
                wmh_workflow.connect(t1w_stripped_node, 't1w_stripped', wmh_synthseg, 'input')
                wmh_synthseg.inputs.output = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_WMHSynthSeg.nii.gz")
                wmh_synthseg.inputs.csv_vols = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-WMHSynthSeg_volume.csv")
                wmh_synthseg.inputs.device = 'cpu'
                wmh_synthseg.inputs.threads = 8
                wmh_synthseg.inputs.save_lesion_probabilities = True
                wmh_synthseg.inputs.prob_filepath = os.path.join(self.output_path, probmap_filename_t1w)
                wmh_synthseg.inputs.wmh_filepath = os.path.join(self.output_path, binarized_wmh_filename_t1w_nothr)

                wmh_workflow.connect(wmh_synthseg, 'wmh_filepath', wmh_mask_node, 'wmh_mask_t1w')
                wmh_workflow.connect(wmh_synthseg, 'prob_filepath', wmh_mask_node, 'wmh_probmap_t1w')
            elif t1w_file == '' and flair_file != '':
                print("[WMH Pipeline] Using FLAIR image for WMHSynthSeg.")
                wmh_workflow.connect(inputnode, 'flair', wmh_synthseg, 'input')
                wmh_synthseg.inputs.output = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_WMHSynthSeg.nii.gz")
                wmh_synthseg.inputs.csv_vols = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_desc-WMHSynthSeg_volume.csv")
                wmh_synthseg.inputs.device = 'cpu'
                wmh_synthseg.inputs.threads = 8
                wmh_synthseg.inputs.save_lesion_probabilities = True
                wmh_synthseg.inputs.prob_filepath = os.path.join(self.output_path, probmap_filename_flair)
                wmh_synthseg.inputs.wmh_filepath = os.path.join(self.output_path, binarized_wmh_filename_flair_nothr)

                wmh_workflow.connect(wmh_synthseg, 'wmh_filepath', wmh_mask_node, 'wmh_mask_flair')
                wmh_workflow.connect(wmh_synthseg, 'prob_filepath', wmh_mask_node, 'wmh_probmap_flair')
        elif self.seg_method == 'truenet':
            truenet_preprocess_dir = os.path.join(self.output_path, 'truenet_preprocess')
            os.makedirs(truenet_preprocess_dir, exist_ok=True)
            truenet_output_dir = os.path.join(self.output_path, 'truenet_output')
            os.makedirs(truenet_output_dir, exist_ok=True)

            if t1w_file != '' and flair_file != '':
                prepare_truenet_data = Node(PrepareTrueNetData2(), name='prepare_truenet_data')
                evaluate_truenet_data = Node(TrueNetEvaluate(), name='evaluate_truenet_data')
                # truenet T1w + FLAIR
                # Let truenet do the preprocess (but in T1w space)
                wmh_workflow.connect(t1w_stripped_node, 't1w_stripped', prepare_truenet_data, 't1w')
                wmh_workflow.connect(flair_in_t1w_node, 'flair_in_t1w', prepare_truenet_data, 'flair')
                wmh_workflow.connect(t1w_stripped_node, 'brain_mask', prepare_truenet_data, 'brain_mask')
                wmh_workflow.connect(check_synthseg, 'synthseg_output', prepare_truenet_data, 'synthseg_img')
                prepare_truenet_data.inputs.output_dir = truenet_preprocess_dir
                prepare_truenet_data.inputs.prefix = 'truenet_preprocess'

                wmh_workflow.connect(prepare_truenet_data, 'output_dir', evaluate_truenet_data, 'inp_dir')
                evaluate_truenet_data.inputs.model_name = 'ukbb'
                evaluate_truenet_data.inputs.output_dir = truenet_output_dir

            elif t1w_file != '' and flair_file == '':
                prepare_truenet_data = Node(PrepareTrueNetData(), name='prepare_truenet_data')
                evaluate_truenet_data = Node(TrueNetEvaluate(), name='evaluate_truenet_data')
                # truenet T1w only
                # Let truenet do the preprocess
                wmh_workflow.connect(inputnode, 't1w', prepare_truenet_data, 'T1')
                prepare_truenet_data.inputs.outname = truenet_preprocess_dir + '/truenet_preprocess'
                prepare_truenet_data.inputs.verbose = True

                wmh_workflow.connect(prepare_truenet_data, 'output_dir', evaluate_truenet_data, 'inp_dir')
                evaluate_truenet_data.inputs.model_name = 'ukbb_t1'
                evaluate_truenet_data.inputs.output_dir = truenet_output_dir
            elif t1w_file == '' and flair_file != '':
                prepare_truenet_data = Node(PrepareTrueNetData(), name='prepare_truenet_data')
                evaluate_truenet_data = Node(TrueNetEvaluate(), name='evaluate_truenet_data')
                # truenet FLAIR only
                # Let truenet do the preprocess (actually can pass original flair image)
                def _copy_flair(flair, outname):
                    import shutil
                    out_flair = f"{outname}_FLAIR.nii.gz"
                    shutil.copy(flair, out_flair)
                    return os.path.dirname(out_flair)
                copy_flair = Node(Function(input_names=['flair', 'outname'],
                                           output_names=['out_dir'],
                                           function=_copy_flair), name='copy_flair')
                copy_flair.inputs.outname = truenet_preprocess_dir + '/truenet_preprocess'
                wmh_workflow.connect(inputnode, 'flair', copy_flair, 'flair')

                wmh_workflow.connect(copy_flair, 'out_dir', prepare_truenet_data, 'inp_dir')
                evaluate_truenet_data.inputs.model_name = 'ukbb_flair'
                evaluate_truenet_data.inputs.output_dir = truenet_output_dir
            
            # truenet postprocess
            truenet_postprocess = Node(TrueNetPostProcess(), name='truenet_postprocess')
            wmh_workflow.connect(evaluate_truenet_data, 'pred_file', truenet_postprocess, 'pred_file')
            wmh_workflow.connect(prepare_truenet_data, 'output_dir', truenet_postprocess, 'preprocess_dir')
            truenet_postprocess.inputs.output_dir = self.output_path
            wmh_workflow.connect(inputnode, 'seg_threshold', truenet_postprocess, 'threshold')
            truenet_postprocess.inputs.output_mask_name = binarized_wmh_filename_t1w if t1w_file != '' else binarized_wmh_filename_flair
            truenet_postprocess.inputs.output_prob_map_name = probmap_filename_t1w if t1w_file != '' else probmap_filename_flair

            # connect to wmh_mask_node
            if t1w_file != '' and flair_file != '':
                wmh_workflow.connect(truenet_postprocess, 'wmh_mask', wmh_mask_node, 'wmh_mask_t1w')
                wmh_workflow.connect(truenet_postprocess, 'wmh_prob_map', wmh_mask_node, 'wmh_probmap_t1w')
            else:
                wmh_workflow.connect(truenet_postprocess, 'wmh_mask', wmh_mask_node, 'wmh_mask_flair')
                wmh_workflow.connect(truenet_postprocess, 'wmh_prob_map', wmh_mask_node, 'wmh_probmap_flair')
            
        final_wmh_mask_node = Node(IdentityInterface(fields=["final_wmh_mask_flair", "final_wmh_mask_t1w"]), name="final_wmh_mask_node")

        if self.use_bianca_mask:
            if t1w_file == '':
                raise ValueError("If use_bianca_mask is True, T1w image must be provided.")
            # first need to run fsl_anat pipeline
            fsl_anat_node = Node(FSLANAT(), name="fsl_anat")
            wmh_workflow.connect(inputnode, "t1w", fsl_anat_node, "input_image")
            fsl_anat_node.inputs.output_directory = os.path.join(self.subject.bids_dir, 'derivatives', 'fsl_anat', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'fsl')

            # then make bianca mask
            make_bianca_mask_node = Node(MakeBIANCAMask(), name="make_bianca_mask")
            wmh_workflow.connect(fsl_anat_node, 'output_directory', make_bianca_mask_node, 'fsl_anat_output')
            make_bianca_mask_node.inputs.bianca_mask_name = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-BIANCA_mask.nii.gz")

            # apply bianca mask to wmh mask
            apply_bianca_mask_node = Node(ApplyMask(), name="apply_bianca_mask")
            wmh_workflow.connect(make_bianca_mask_node, 'bianca_mask', apply_bianca_mask_node, 'mask_file')
            wmh_workflow.connect(wmh_mask_node, 'wmh_mask_t1w', apply_bianca_mask_node, 'in_file')
            apply_bianca_mask_node.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-{self.seg_method}{thr_string}Filtered_mask.nii.gz")

            wmh_workflow.connect(wmh_mask_node, 'wmh_mask_flair', final_wmh_mask_node, 'final_wmh_mask_flair')
            wmh_workflow.connect(apply_bianca_mask_node, 'out_file', final_wmh_mask_node, 'final_wmh_mask_t1w')
        else:
            wmh_workflow.connect(wmh_mask_node, 'wmh_mask_flair', final_wmh_mask_node, 'final_wmh_mask_flair')
            wmh_workflow.connect(wmh_mask_node, 'wmh_mask_t1w', final_wmh_mask_node, 'final_wmh_mask_t1w')
        
        # --------------------------
        # Part 2: WMH Quantification
        # --------------------------
        # If provide T1w, will do quantification in T1w space
        if t1w_file is not None and t1w_file != '':
            # quantification the total WMH
            twmh_quantification = Node(CalculateVolume(), name='twmh_quantification')
            wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', twmh_quantification, 'in_nii')
            twmh_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-{self.seg_method}{thr_string}_volume.csv")

            if 'Fazekas' in self.location_method:
                # results in this section
                fazekas_output = Node(IdentityInterface(fields=["total_wmh", "pwmh", "dwmh"]), name="fazekas_output")

                fazekas_classification = Node(FazekasClassification(), name='fazekas_classification')

                # Preprocess for lateral ventricle mask generation and bianca mask
                # if self.ventmask_method == 'fsl_anat' or self.use_bianca_mask == True, will need to run fsl_anat
                if self.ventmask_method == 'fsl_anat':
                    #TODO
                    print("TODO")
                elif self.ventmask_method == 'SynthSeg':
                    print("[WMH Pipeline] Using SynthSeg to generate lateral ventricle mask.")
                    
                    # Extract lateral ventricle mask from synthseg output
                    # eg vent_mask = extract_roi_from_image(self.inputs.wmh_synthseg, [4, 43], binarize=True, output_path=os.path.join(self.inputs.output_dir, self.inputs.vent_mask_filename))
                    extract_ventmask = Node(ExtractRegion(), name='extract_ventmask')
                    wmh_workflow.connect(check_synthseg, 'synthseg_output', extract_ventmask, 'in_nii')
                    extract_ventmask.inputs.roi_list = [4, 43]
                    extract_ventmask.inputs.binarize = True
                    extract_ventmask.inputs.output_nii = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-LateralVentricle_desc-SynthSeg_mask.nii.gz")

                    # Calculate distancemap
                    distancemap = Node(DistanceMap(), name='distancemap')
                    wmh_workflow.connect(extract_ventmask, 'out_nii', distancemap, 'in_file')
                    distancemap.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-SynthSeg_LateralVentricleDistancemap.nii.gz")

                    # Calculate 3mm and 10mm mask
                    # 3mm mask
                    mask_3mm_vent_mask = Node(FSLMathsUnderThr(), name='mask_3mm_vent_mask')
                    wmh_workflow.connect(distancemap, 'out_file', mask_3mm_vent_mask, 'in_file')
                    mask_3mm_vent_mask.inputs.threshold = 3.0
                    mask_3mm_vent_mask.inputs.binarize = True
                    mask_3mm_vent_mask.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-LateralVentricle3mm_desc-SynthSeg_mask.nii.gz")

                    # 10mm mask
                    mask_10mm_vent_mask = Node(FSLMathsUnderThr(), name='mask_10mm_vent_mask')
                    wmh_workflow.connect(distancemap, 'out_file', mask_10mm_vent_mask, 'in_file')
                    mask_10mm_vent_mask.inputs.threshold = 10.0
                    mask_10mm_vent_mask.inputs.binarize = True
                    mask_10mm_vent_mask.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-LateralVentricle10mm_desc-SynthSeg_mask.nii.gz")

                    wmh_workflow.connect(extract_ventmask, 'out_nii', fazekas_classification, 'vent_mask')
                    wmh_workflow.connect(mask_10mm_vent_mask, 'out_file', fazekas_classification, 'perivent_mask_10mm')
                    wmh_workflow.connect(mask_3mm_vent_mask, 'out_file', fazekas_classification, 'perivent_mask_3mm')
                    fazekas_classification.inputs.output_dir = self.output_path
                    fazekas_classification.inputs.pwmh_mask_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-PWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"
                    fazekas_classification.inputs.dwmh_mask_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-DWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"

                # Fazekas classification results
                wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', fazekas_classification, 'wmh_img')

                wmh_workflow.connect(fazekas_classification, 'pwmh_mask', fazekas_output, 'pwmh')
                wmh_workflow.connect(fazekas_classification, 'dwmh_mask', fazekas_output, 'dwmh')

                # quantification
                pwmh_quantification = Node(CalculateVolume(), name='pwmh_quantification')
                wmh_workflow.connect(fazekas_output, 'pwmh', pwmh_quantification, 'in_nii')
                pwmh_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-PWMH_desc-{self.seg_method}{thr_string}_volume.csv")

                dwmh_quantification = Node(CalculateVolume(), name='dwmh_quantification')
                wmh_workflow.connect(fazekas_output, 'dwmh', dwmh_quantification, 'in_nii')
                dwmh_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-DWMH_desc-{self.seg_method}{thr_string}_volume.csv")
            
            if 'bullseye' in self.location_method:
                #from cvdproc.pipelines.external.bullseye_WMH.bullseye_pipeline_custom import create_bullseye_pipeline
                from cvdproc.pipelines.smri.csvd_quantification.wmh.wmh_location_nipype import Bullseye2

                if self.session.freesurfer_dir is None:
                    raise ValueError("Freesurfer directory is not available. Please run Freesurfer recon-all first.")
                else:
                    print(f"[WMH Pipeline] Found Freesurfer directory: {self.session.freesurfer_dir}. Will run Bullseye WMH location quantification.")
                    bullseye_process = Node(Bullseye2(), name='bullseye_process')
                    wmh_workflow.connect(inputnode, 'session', bullseye_process, 'subject_id')
                    bullseye_process.inputs.source_dir = get_package_path('pipelines', 'external', 'WMH_Bullseye')
                    bullseye_process.inputs.subjects_dir = os.path.dirname(self.session.freesurfer_dir)
                    bullseye_process.inputs.output_dir = os.path.join(self.output_path, 'bullseye')

                    fs_to_t1w_reg = Node(Tkregister2fs2t1w(), name='fs_to_t1w_reg')
                    fs_to_t1w_reg.inputs.fs_subjects_dir = os.path.dirname(self.session.freesurfer_dir)
                    wmh_workflow.connect(inputnode, 'session', fs_to_t1w_reg, 'fs_subject_id')
                    fs_to_t1w_reg.inputs.output_matrix = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-fs_to-T1w_xfm.mat")
                    fs_to_t1w_reg.inputs.output_inverse_matrix = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-fs_xfm.mat")

                    # use 4 separate nodes
                    transform_bullseye_parc_to_t1w_bullseye = Node(ApplyXFM(), name='transform_bullseye_parc_to_t1w_bullseye')
                    wmh_workflow.connect(bullseye_process, 'bullseye_wmparc', transform_bullseye_parc_to_t1w_bullseye, 'in_file')
                    wmh_workflow.connect(inputnode, 't1w', transform_bullseye_parc_to_t1w_bullseye, 'reference')
                    wmh_workflow.connect(fs_to_t1w_reg, 'output_matrix', transform_bullseye_parc_to_t1w_bullseye, 'in_matrix_file')
                    transform_bullseye_parc_to_t1w_bullseye.inputs.interp = 'nearestneighbour'
                    transform_bullseye_parc_to_t1w_bullseye.inputs.apply_xfm = True
                    transform_bullseye_parc_to_t1w_bullseye.inputs.out_file = os.path.join(self.output_path, 'bullseye', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-bullseye_wmparc.nii.gz")

                    transform_bullseye_parc_to_t1w_bullseyebis = Node(ApplyXFM(), name='transform_bullseye_parc_to_t1w_bullseyebis')
                    wmh_workflow.connect(bullseye_process, 'bullseye_wmparc_bis', transform_bullseye_parc_to_t1w_bullseyebis, 'in_file')
                    wmh_workflow.connect(inputnode, 't1w', transform_bullseye_parc_to_t1w_bullseyebis, 'reference')
                    wmh_workflow.connect(fs_to_t1w_reg, 'output_matrix', transform_bullseye_parc_to_t1w_bullseyebis, 'in_matrix_file')
                    transform_bullseye_parc_to_t1w_bullseyebis.inputs.interp = 'nearestneighbour'
                    transform_bullseye_parc_to_t1w_bullseyebis.inputs.apply_xfm = True
                    transform_bullseye_parc_to_t1w_bullseyebis.inputs.out_file = os.path.join(self.output_path, 'bullseye', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-bullseyebis_wmparc.nii.gz")

                    transform_bullseye_parc_to_t1w_lobarseg = Node(ApplyXFM(), name='transform_bullseye_parc_to_t1w_lobarseg')
                    wmh_workflow.connect(bullseye_process, 'lobar_wmparc', transform_bullseye_parc_to_t1w_lobarseg, 'in_file')
                    wmh_workflow.connect(inputnode, 't1w', transform_bullseye_parc_to_t1w_lobarseg, 'reference')
                    wmh_workflow.connect(fs_to_t1w_reg, 'output_matrix', transform_bullseye_parc_to_t1w_lobarseg, 'in_matrix_file')
                    transform_bullseye_parc_to_t1w_lobarseg.inputs.interp = 'nearestneighbour'
                    transform_bullseye_parc_to_t1w_lobarseg.inputs.apply_xfm = True
                    transform_bullseye_parc_to_t1w_lobarseg.inputs.out_file = os.path.join(self.output_path, 'bullseye', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-lobarseg_wmparc.nii.gz")

                    transform_bullseye_parc_to_t1w_lobarsegbis = Node(ApplyXFM(), name='transform_bullseye_parc_to_t1w_lobarsegbis')
                    wmh_workflow.connect(bullseye_process, 'lobar_wmparc_bis', transform_bullseye_parc_to_t1w_lobarsegbis, 'in_file')
                    wmh_workflow.connect(inputnode, 't1w', transform_bullseye_parc_to_t1w_lobarsegbis, 'reference')
                    wmh_workflow.connect(fs_to_t1w_reg, 'output_matrix', transform_bullseye_parc_to_t1w_lobarsegbis, 'in_matrix_file')
                    transform_bullseye_parc_to_t1w_lobarsegbis.inputs.interp = 'nearestneighbour'
                    transform_bullseye_parc_to_t1w_lobarsegbis.inputs.apply_xfm = True
                    transform_bullseye_parc_to_t1w_lobarsegbis.inputs.out_file = os.path.join(self.output_path, 'bullseye', f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-lobarsegbis_wmparc.nii.gz")

                    # quantification (volume) using bullseye_wmparc_bis and lobar_wmparc_bis
                    bullseye_quantification = Node(CalculateROIVolume(), name='bullseye_quantification')
                    wmh_workflow.connect(transform_bullseye_parc_to_t1w_bullseyebis, 'out_file', bullseye_quantification, 'roi_nii')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', bullseye_quantification, 'in_nii')
                    bullseye_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-bullseye_volume.csv")

                    lobar_quantification = Node(CalculateROIVolume(), name='lobar_quantification')
                    wmh_workflow.connect(transform_bullseye_parc_to_t1w_lobarsegbis, 'out_file', lobar_quantification, 'roi_nii')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', lobar_quantification, 'in_nii')
                    lobar_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-lobarseg_volume.csv")

            if 'shiva' in self.location_method:
                from cvdproc.pipelines.smri.csvd_quantification.shiva_segmentation.shiva_parc import ShivaGeneralParcellation, Brain_Seg_for_biomarker
                shiva_general_parc_node = Node(ShivaGeneralParcellation(), name='shiva_general_parc')
                wmh_workflow.connect(check_synthseg, 'synthseg_output', shiva_general_parc_node, 'in_seg')
                shiva_general_parc_node.inputs.out_seg = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-shivaParc_synthseg.nii.gz")

                shiva_wmh_seg_node = Node(Brain_Seg_for_biomarker(), name='shiva_wmh_seg')
                wmh_workflow.connect(shiva_general_parc_node, 'out_seg', shiva_wmh_seg_node, 'brain_seg')
                shiva_wmh_seg_node.inputs.custom_parc = 'wmh'
                shiva_wmh_seg_node.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-shivaParcWMH_synthseg.nii.gz")
                shiva_wmh_seg_node.inputs.parc_json = os.path.join(self.output_path, f"shivaParcWMH.json")

                # calculate volume for shiva parcellation
                shiva_wmh_quantification = Node(CalculateROIVolume(), name='shiva_wmh_quantification')
                wmh_workflow.connect(shiva_wmh_seg_node, 'brain_seg', shiva_wmh_quantification, 'roi_nii')
                wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', shiva_wmh_quantification, 'in_nii')
                shiva_wmh_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-shivaParcWMH_volume.csv")
            
            if 'McDonald' in self.location_method:
                if self.seg_method == 'LSTAI':
                    # already have parcellations, but not used, so just need to copy
                    copy_parcellated_csv = Node(CopyFileCommandLine(), name="copy_parcellated_csv")
                    wmh_workflow.connect(lst_ai, 'parcellated_volume', copy_parcellated_csv, 'input_file')
                    copy_parcellated_csv.inputs.output_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-McDonald_volume.csv")

                    copy_parcellated_wmh = Node(CopyFileCommandLine(), name="copy_parcellated_wmh")
                    wmh_workflow.connect(lst_ai, 'parcellated_wmh_mask', copy_parcellated_wmh, 'input_file')
                    copy_parcellated_wmh.inputs.output_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-WMH_McDonaldSeg.nii.gz")
                else:
                    lst_ai_outputdir = os.path.join(self.output_path, 'lst_ai')
                    os.makedirs(lst_ai_outputdir, exist_ok=True)
                    lst_ai_annote_node = Node(LSTAI(), name='lst_ai_annote')
                    wmh_workflow.connect(flair_in_t1w_node, 'flair_in_t1w', lst_ai_annote_node, 'flair_img')
                    wmh_workflow.connect(t1w_stripped_node, 't1w_stripped', lst_ai_annote_node, 't1w_img')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', lst_ai_annote_node, 'existing_seg')
                    lst_ai_annote_node.inputs.output_dir = lst_ai_outputdir
                    lst_ai_annote_node.inputs.temp_dir = lst_ai_outputdir
                    lst_ai_annote_node.inputs.img_stripped = True
                    lst_ai_annote_node.inputs.annotate_only = True

                    # and then copy the results
                    copy_parcellated_csv = Node(CopyFileCommandLine(), name="copy_parcellated_csv")
                    wmh_workflow.connect(lst_ai_annote_node, 'parcellated_volume', copy_parcellated_csv, 'input_file')
                    copy_parcellated_csv.inputs.output_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-McDonald_volume.csv")

                    copy_parcellated_wmh = Node(CopyFileCommandLine(), name="copy_parcellated_wmh")
                    wmh_workflow.connect(lst_ai_annote_node, 'parcellated_wmh_mask', copy_parcellated_wmh, 'input_file')
                    copy_parcellated_wmh.inputs.output_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-WMH_McDonaldSeg.nii.gz")

        elif t1w_file == '' and flair_file != '':
            # quantification in FLAIR space
            fwmh_quantification = Node(CalculateVolume(), name='fwmh_quantification')
            wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_flair', fwmh_quantification, 'in_nii')
            fwmh_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_label-WMH_desc-{self.seg_method}{thr_string}_volume.csv")

            if 'Fazekas' in self.location_method:
                fazekas_output = Node(IdentityInterface(fields=["total_wmh", "pwmh", "dwmh"]), name="fazekas_output_flair")
                fazekas_classification_flair = Node(FazekasClassification(), name='fazekas_classification_flair')

                # only use SynthSeg to generate lateral ventricle mask
                synthseg_flair_node = Node(SynthSeg(), name='synthseg_flair')
                wmh_workflow.connect(inputnode, 'flair', synthseg_flair_node, 'image')
                synthseg_flair_node.inputs.out = os.path.join(self.subject.bids_dir, 'derivatives', 'anat_seg', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_synthseg.nii.gz')
                synthseg_flair_node.inputs.vol = os.path.join(self.subject.bids_dir, 'derivatives', 'anat_seg', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'synthseg', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-synthseg_volume.csv')
                synthseg_flair_node.inputs.robust = True
                synthseg_flair_node.inputs.parc = True
                synthseg_flair_node.inputs.keepgeom = True

                # Extract lateral ventricle mask from synthseg output
                extract_ventmask_flair = Node(ExtractRegion(), name='extract_ventmask_flair')
                wmh_workflow.connect(synthseg_flair_node, 'out', extract_ventmask_flair, 'in_nii')
                extract_ventmask_flair.inputs.roi_list = [4, 43]
                extract_ventmask_flair.inputs.binarize = True
                extract_ventmask_flair.inputs.output_nii = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-LateralVentricle_desc-SynthSeg_mask.nii.gz")

                # Calculate distancemap
                distancemap_flair = Node(DistanceMap(), name='distancemap_flair')
                wmh_workflow.connect(extract_ventmask_flair, 'out_nii', distancemap_flair, 'in_file')
                distancemap_flair.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_desc-SynthSeg_LateralVentricleDistancemap.nii.gz")
                # Calculate 3mm and 10mm mask
                # 3mm mask
                mask_3mm_vent_mask_flair = Node(FSLMathsUnderThr(), name='mask_3mm_vent_mask_flair')
                wmh_workflow.connect(distancemap_flair, 'out_file', mask_3mm_vent_mask_flair, 'in_file')
                mask_3mm_vent_mask_flair.inputs.threshold = 3.0
                mask_3mm_vent_mask_flair.inputs.binarize = True
                mask_3mm_vent_mask_flair.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-LateralVentricle3mm_desc-SynthSeg_mask.nii.gz")
                # 10mm mask
                mask_10mm_vent_mask_flair = Node(FSLMathsUnderThr(), name='mask_10mm_vent_mask_flair')
                wmh_workflow.connect(distancemap_flair, 'out_file', mask_10mm_vent_mask_flair, 'in_file')
                mask_10mm_vent_mask_flair.inputs.threshold = 10.0
                mask_10mm_vent_mask_flair.inputs.binarize = True
                mask_10mm_vent_mask_flair.inputs.out_file = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-LateralVentricle10mm_desc-SynthSeg_mask.nii.gz")

                wmh_workflow.connect(extract_ventmask_flair, 'out_nii', fazekas_classification_flair, 'vent_mask')
                wmh_workflow.connect(mask_10mm_vent_mask_flair, 'out_file', fazekas_classification_flair, 'perivent_mask_10mm')
                wmh_workflow.connect(mask_3mm_vent_mask_flair, 'out_file', fazekas_classification_flair, 'perivent_mask_3mm')
                fazekas_classification_flair.inputs.output_dir = self.output_path
                fazekas_classification_flair.inputs.pwmh_mask_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-PWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"
                fazekas_classification_flair.inputs.dwmh_mask_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-FLAIR_label-DWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"

                # Fazekas classification results
                wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_flair', fazekas_classification_flair, 'wmh_img')
                wmh_workflow.connect(fazekas_classification_flair, 'pwmh_mask', fazekas_output, 'pwmh')
                wmh_workflow.connect(fazekas_classification_flair, 'dwmh_mask', fazekas_output, 'dwmh')
        
        # ------------------------
        # Part 3: Normalize to MNI
        # ------------------------
        if self.normalize_to_mni:
            if t1w_file != '': # If 3D-T1w available, will normalize to MNI space
                wmh_to_mni_transform_node = MapNode(MRIConvertApplyWarp(), name='wmh_to_mni_transform_node', iterfield=['input_image', 'output_image', 'interp'])

                if 'Fazekas' in self.location_method:
                    files_to_register_node = Node(Merge(4), name='files_to_register_node')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', files_to_register_node, 'in1')
                    wmh_workflow.connect(wmh_mask_node, 'wmh_probmap_t1w', files_to_register_node, 'in2')
                    wmh_workflow.connect(fazekas_classification, 'pwmh_mask', files_to_register_node, 'in3')
                    wmh_workflow.connect(fazekas_classification, 'dwmh_mask', files_to_register_node, 'in4')

                    wmh_workflow.connect(files_to_register_node, 'out', wmh_to_mni_transform_node, 'input_image')
                    wmh_to_mni_transform_node.inputs.output_image = [
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-WMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_desc-{self.seg_method}_probmap.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-PWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-DWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                    ]
                    wmh_to_mni_transform_node.inputs.interp = ['nearest', 'interpolate', 'nearest', 'nearest']
                else:
                    # only total WMH mask and probmap to MNI space
                    files_to_register_node = Node(Merge(2), name='files_to_register_node')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', files_to_register_node, 'in1')
                    wmh_workflow.connect(wmh_mask_node, 'wmh_probmap_t1w', files_to_register_node, 'in2')

                    wmh_workflow.connect(files_to_register_node, 'out', wmh_to_mni_transform_node, 'input_image')
                    wmh_to_mni_transform_node.inputs.output_image = [
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-WMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_desc-{self.seg_method}_probmap.nii.gz"),
                    ]
                    wmh_to_mni_transform_node.inputs.interp = ['nearest', 'interpolate']

                # will use the T1w to MNI warp to transform WMH to MNI space (2-step)
                target_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz')
                
                if not os.path.exists(target_warp):
                    print(f"[WMH Pipeline] No existing T1w to MNI warp file found: {target_warp}. Will run Synthmorph registration to get the warp (1mm resolution).")
                    print(f"[WMH Pipeline] If you want a different resolution, please run a separate T1 registration pipeline first.")
                    t1w_to_mni_registration = Node(SynthmorphNonlinear(), name='t1w_to_mni_registration')
                    wmh_workflow.connect(inputnode, 't1w', t1w_to_mni_registration, 't1')
                    t1w_to_mni_registration.inputs.mni_template = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'standard', 'MNI152', 'MNI152_T1_1mm_brain.nii.gz')
                    t1w_to_mni_registration.inputs.t1_mni_out = os.path.join(os.path.dirname(target_warp), rename_bids_file(t1w_file, {'space': 'MNI152NLin6ASym', 'desc':'brain'}, 'T1w', '.nii.gz'))
                    t1w_to_mni_registration.inputs.t1_2_mni_warp = target_warp
                    t1w_to_mni_registration.inputs.mni_2_t1_warp = os.path.join(os.path.dirname(target_warp), f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')
                    t1w_to_mni_registration.inputs.register_between_stripped = True

                    wmh_workflow.connect(t1w_to_mni_registration, 't1_2_mni_warp', wmh_to_mni_transform_node, 'warp_image')
                else:
                    print(f"[WMH Pipeline] Found existing T1w to MNI warp file: {target_warp}. Will use it to transform WMH to MNI space.")
                    wmh_to_mni_transform_node.inputs.warp_image = target_warp
                
                if 'JHU' in self.location_method:
                    # transform JHU WM atlas to T1w space
                    jhu_to_t1w_transform = Node(MRIConvertApplyWarp(), name='jhu_to_t1w_transform')
                    target_inverse_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')
                    if not os.path.exists(target_inverse_warp):
                        wmh_workflow.connect(t1w_to_mni_registration, 'mni_2_t1_warp', jhu_to_t1w_transform, 'warp_image')
                    else:
                        jhu_to_t1w_transform.inputs.warp_image = target_inverse_warp
                    jhu_to_t1w_transform.inputs.input_image = get_package_path('data', 'standard', 'JHU', 'JHU-ICBM-labels-1mm.nii.gz')
                    jhu_to_t1w_transform.inputs.output_image = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_desc-JHU_atlas.nii.gz")
                    jhu_to_t1w_transform.inputs.interp = 'nearest'

                    # volume quantification
                    jhu_wmh_quantification = Node(CalculateROIVolume(), name='jhu_wmh_quantification')
                    wmh_workflow.connect(jhu_to_t1w_transform, 'output_image', jhu_wmh_quantification, 'roi_nii')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_t1w', jhu_wmh_quantification, 'in_nii')
                    jhu_wmh_quantification.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-JHU_volume.csv")
            elif t1w_file == '' and flair_file != '': # If only FLAIR available, will normalize to MNI space
                wmh_to_mni_transform_node = MapNode(MRIConvertApplyWarp(), name='wmh_to_mni_transform_node', iterfield=['input_image', 'output_image', 'interp'])

                if 'Fazekas' in self.location_method:
                    files_to_register_node = Node(Merge(4), name='files_to_register_node')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_flair', files_to_register_node, 'in1')
                    wmh_workflow.connect(wmh_mask_node, 'wmh_probmap_flair', files_to_register_node, 'in2')
                    wmh_workflow.connect(fazekas_classification_flair, 'pwmh_mask', files_to_register_node, 'in3')
                    wmh_workflow.connect(fazekas_classification_flair, 'dwmh_mask', files_to_register_node, 'in4')

                    wmh_workflow.connect(files_to_register_node, 'out', wmh_to_mni_transform_node, 'input_image')
                    wmh_to_mni_transform_node.inputs.output_image = [
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-WMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_desc-{self.seg_method}_probmap.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-PWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-DWMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                    ]
                    wmh_to_mni_transform_node.inputs.interp = ['nearest', 'interpolate', 'nearest', 'nearest']
                else:
                    # only total WMH mask and probmap to MNI space
                    files_to_register_node = Node(Merge(2), name='files_to_register_node')
                    wmh_workflow.connect(final_wmh_mask_node, 'final_wmh_mask_flair', files_to_register_node, 'in1')
                    wmh_workflow.connect(wmh_mask_node, 'wmh_probmap_flair', files_to_register_node, 'in2')
                    wmh_workflow.connect(files_to_register_node, 'out', wmh_to_mni_transform_node, 'input_image')
                    wmh_to_mni_transform_node.inputs.output_image = [
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-WMH_desc-{self.seg_method}{thr_string}_mask.nii.gz"),
                        os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_desc-{self.seg_method}_probmap.nii.gz"),
                    ]
                    wmh_to_mni_transform_node.inputs.interp = ['nearest', 'interpolate']

                # will use the FLAIR to MNI warp to transform WMH to MNI space
                target_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-FLAIR_to-MNI152NLin6ASym_warp.nii.gz')

                if not os.path.exists(target_warp):
                    print(f"[WMH Pipeline] No existing FLAIR to MNI warp file found: {target_warp}. Will run Synthmorph registration to get the warp (1mm resolution).")
                    print(f"[WMH Pipeline] If you want a different resolution, please run a separate FLAIR registration pipeline first.")
                    flair_to_mni_registration = Node(SynthmorphNonlinear(), name='flair_to_mni_registration')
                    wmh_workflow.connect(inputnode, 'flair', flair_to_mni_registration, 't1')
                    flair_to_mni_registration.inputs.mni_template = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'standard', 'MNI152', 'MNI152_T1_1mm_brain.nii.gz')
                    flair_to_mni_registration.inputs.t1_mni_out = os.path.join(os.path.dirname(target_warp), rename_bids_file(flair_file, {'space': 'MNI152NLin6ASym', 'desc':'brain'}, 'FLAIR', '.nii.gz'))
                    flair_to_mni_registration.inputs.t1_2_mni_warp = target_warp
                    flair_to_mni_registration.inputs.mni_2_t1_warp = os.path.join(os.path.dirname(target_warp), f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-FLAIR_warp.nii.gz')
                    flair_to_mni_registration.inputs.register_between_stripped = True

                    wmh_workflow.connect(flair_to_mni_registration, 't1_2_mni_warp', wmh_to_mni_transform_node, 'warp_image')
                else:
                    print(f"[WMH Pipeline] Found existing FLAIR to MNI warp file: {target_warp}. Will use it to transform WMH to MNI space.")
                    wmh_to_mni_transform_node.inputs.warp_image = target_warp
        
        # -----------------------
        # Part 4: Shape Features 
        # -----------------------
        if self.shape_features:
            if not self.normalize_to_mni:
                raise ValueError("[WMH Pipeline] Shape feature calculation requires WMH to be normalized to MNI space. Please set normalize_to_mni=True.")
            
            if 'Fazekas' not in self.location_method:
                raise ValueError("[WMH Pipeline] Shape feature calculation requires WMH to be classified into PWMH and DWMH. Please include 'Fazekas' in location_method.")
        
            # now we have wmh_to_mni_transform_node to get the PWMH and DWMH in MNI space (output_image[2] and output_image[3])
            # we need a node to connect to calculate shape features (input: the list of 4 files; output: pwmh_path and dwmh_path)
            def extract_pwmh_dwmh(wmh_files):
                if len(wmh_files) != 4:
                    raise ValueError("Expected 4 files (WMH mask, WMH probmap, PWMH mask, DWMH mask) to extract PWMH and DWMH.")
                return wmh_files[2], wmh_files[3]
            extract_pwmh_dwmh_node = Node(Function(input_names=['wmh_files'],
                                                   output_names=['pwmh_path', 'dwmh_path'],
                                                    function=extract_pwmh_dwmh), name='extract_pwmh_dwmh_node')
            wmh_workflow.connect(wmh_to_mni_transform_node, 'output_image', extract_pwmh_dwmh_node, 'wmh_files')

            shape_features_dir = os.path.join(self.output_path, 'shape_features')
            os.makedirs(shape_features_dir, exist_ok=True)

            # Calculate shape features for PWMH and DWMH separately
            pwmh_shape_features = Node(WMHShape(), name='pwmh_shape_features')
            wmh_workflow.connect(extract_pwmh_dwmh_node, 'pwmh_path', pwmh_shape_features, 'wmh_mask')
            pwmh_shape_features.inputs.threshold = 10
            pwmh_shape_features.inputs.save_plots = False
            pwmh_shape_features.inputs.output_dir = shape_features_dir
            pwmh_shape_features.inputs.wmh_labeled_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-PWMH_desc-{self.seg_method}{thr_string}_label.nii.gz"
            pwmh_shape_features.inputs.shape_csv_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-PWMH_ShapeFeatures.csv"
            pwmh_shape_features.inputs.shape_csv_avg_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-PWMH_ShapeFeaturesAvg.csv"

            dwmh_shape_features = Node(WMHShape(), name='dwmh_shape_features')
            wmh_workflow.connect(extract_pwmh_dwmh_node, 'dwmh_path', dwmh_shape_features, 'wmh_mask')
            dwmh_shape_features.inputs.threshold = 10
            dwmh_shape_features.inputs.save_plots = False
            dwmh_shape_features.inputs.output_dir = shape_features_dir
            dwmh_shape_features.inputs.wmh_labeled_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-MNI152NLin6ASym_label-DWMH_desc-{self.seg_method}{thr_string}_label.nii.gz"
            dwmh_shape_features.inputs.shape_csv_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-DWMH_ShapeFeatures.csv"
            dwmh_shape_features.inputs.shape_csv_avg_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-DWMH_ShapeFeaturesAvg.csv"

        return wmh_workflow

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