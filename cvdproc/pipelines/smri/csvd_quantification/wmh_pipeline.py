import os
import nibabel as nib
import pandas as pd
from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from ....bids_data.rename_bids_file import rename_bids_file

from .wmh.wmh_seg_nipype import LSTSegmentation, LSTAI, WMHSynthSegSingle, PrepareTrueNetData, PrepareTrueNetData2, TrueNetEvaluate, TrueNetPostProcess
from .wmh.wmh_location_nipype import FazekasClassification
from .wmh.wmh_shape_nipype import WMHShape

from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.fsl.preprocess import ApplyXFM

from cvdproc.pipelines.smri.fsl.fsl_anat_nipype import FSLANAT
from cvdproc.pipelines.smri.fsl.distancemap_nipype import DistanceMap
from cvdproc.pipelines.smri.fsl.fslmaths_thr import FSLMathsUnderThr, FSLMathsThr
from cvdproc.pipelines.smri.fsl.make_bianca_mask_nipype import MakeBIANCAMask
from cvdproc.pipelines.smri.freesurfer.synthstrip import SynthStrip
from cvdproc.pipelines.smri.freesurfer.synthseg import SynthSeg

from cvdproc.pipelines.common.extract_region import ExtractRegion
from cvdproc.pipelines.common.calculate_volume import CalculateVolume
from cvdproc.pipelines.common.calculate_roi_volume import CalculateROIVolume
from cvdproc.pipelines.common.image_calc import CombineMasks, RemoveMaskRegion
from cvdproc.pipelines.common.files import CopyFileCommandLine
from cvdproc.pipelines.common.register import SynthmorphNonlinear, MRIConvertApplyWarp
from cvdproc.pipelines.common.register import ModalityRegistration, Tkregister2fs2t1w

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
                 exclude_mask: str = 'lesion_mask',
                 use_which_exclude_mask: str = None,
                 ignore_t1w_in_truenet: bool = False,
                 seg_threshold: float = 0.5,
                 location_method: list = ['Fazekas'],
                 ventmask_method: str = 'SynthSeg',
                 use_bianca_mask: bool = False,
                 normalize_to_mni: bool = False,
                 shape_features: bool = False,
                 extract_from: str = None,
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
            exclude_mask: Folder name for exclude mask. Expected to find <bids_dir>/derivatives/<exclude_mask>/sub-<id>/ses-<id>/*<use_which_mask>*.nii.gz. Defaults to 'lesion_mask'. Must be in the T1w space.
            use_which_exclude_mask: specific string to select exclude mask, e.g. 'desc-RSSI'. If None, exclude mask is not used
            ignore_t1w_in_truenet: [EXPERIMENTAL] When using SynthSR T1w, ignore it in TrueNet (ukbb -> ukbb_flair)
            seg_threshold: threshold for WMH segmentation (not used for WMHSynthSeg method)
            location_method: list of location method, subset of ['Fazekas', 'bullseye', 'shiva', 'McDonald', 'JHU']
            ventmask_method: method to get ventricle mask, one of ['SynthSeg']
            use_bianca_mask: whether to use BIANCA white matter mask to constrain WMH segmentation
            normalize_to_mni: whether to normalize the WMH mask to MNI space
            shape_features: whether to calculate shape features for each WMH cluster (need normalize_to_mni=True and location_method contains 'Fazekas')
            extract_from (str, optional): Folder name to extract results from
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.use_which_t1w = use_which_t1w
        self.use_which_flair = use_which_flair
        self.seg_method = seg_method
        self.exclude_mask = exclude_mask
        self.use_which_exclude_mask = use_which_exclude_mask
        self.ignore_t1w_in_truenet = ignore_t1w_in_truenet
        self.seg_threshold = seg_threshold
        self.location_method = location_method
        self.ventmask_method = ventmask_method
        self.use_bianca_mask = use_bianca_mask
        self.normalize_to_mni = normalize_to_mni
        self.shape_features = shape_features
        self.extract_from = extract_from
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

        if self.use_which_exclude_mask is not None:
            lesion_mask_dir = self.session._find_output(self.exclude_mask)
            exclude_mask = [file for file in os.listdir(lesion_mask_dir) if f'{self.use_which_exclude_mask}' in file]

            if exclude_mask is not None and len(exclude_mask) == 1:
                print(f"[WMH Pipeline] Using exclude mask: {exclude_mask[0]}.")
                exclude_mask = os.path.join(lesion_mask_dir, exclude_mask[0])
            elif exclude_mask is not None and len(exclude_mask) > 1:
                #print(f"Using the first mask found: {exclude_mask[0]}.")
                exclude_mask = os.path.join(lesion_mask_dir, exclude_mask[0])
                print(f"[WMH Pipeline] Using exclude mask: {exclude_mask}.")
            else:
                exclude_mask = None
                print("[WMH Pipeline] No exclude mask found.")
        else:
            exclude_mask = None

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
            synthseg_outfile = os.path.join(self.subject.bids_dir, 'derivatives', 'anat_seg', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', 'synthseg', rename_bids_file(t1w_file, {'space': 'T1w'}, 'synthseg', '.nii.gz'))
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
                wmh_workflow.connect(t1w_stripped_node, 't1w_stripped', prepare_truenet_data, 't1w')
                wmh_workflow.connect(flair_in_t1w_node, 'flair_in_t1w', prepare_truenet_data, 'flair')
                wmh_workflow.connect(t1w_stripped_node, 'brain_mask', prepare_truenet_data, 'brain_mask')
                wmh_workflow.connect(check_synthseg, 'synthseg_output', prepare_truenet_data, 'synthseg_img')
                prepare_truenet_data.inputs.output_dir = truenet_preprocess_dir
                prepare_truenet_data.inputs.prefix = 'truenet_preprocess'

                wmh_workflow.connect(prepare_truenet_data, 'output_dir', evaluate_truenet_data, 'inp_dir')
                evaluate_truenet_data.inputs.model_name = 'ukbb'
                if self.ignore_t1w_in_truenet:
                    prepare_truenet_data.inputs.keep_t1w = '0'
                    evaluate_truenet_data.inputs.model_name = 'ukbb_flair'
                evaluate_truenet_data.inputs.output_dir = truenet_output_dir

            elif t1w_file != '' and flair_file == '':
                prepare_truenet_data = Node(PrepareTrueNetData2(), name='prepare_truenet_data')
                evaluate_truenet_data = Node(TrueNetEvaluate(), name='evaluate_truenet_data')
                # truenet T1w only
                wmh_workflow.connect(inputnode, 't1w', prepare_truenet_data, 't1w')
                prepare_truenet_data.inputs.flair = 'NONE'  # no FLAIR image
                wmh_workflow.connect(t1w_stripped_node, 'brain_mask', prepare_truenet_data, 'brain_mask')
                wmh_workflow.connect(check_synthseg, 'synthseg_output', prepare_truenet_data, 'synthseg_img')
                prepare_truenet_data.inputs.output_dir = truenet_preprocess_dir
                prepare_truenet_data.inputs.prefix = 'truenet_preprocess'

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
            if t1w_file != '':
                wmh_workflow.connect(truenet_postprocess, 'wmh_mask', wmh_mask_node, 'wmh_mask_t1w')
                wmh_workflow.connect(truenet_postprocess, 'wmh_prob_map', wmh_mask_node, 'wmh_probmap_t1w')
            else:
                wmh_workflow.connect(truenet_postprocess, 'wmh_mask', wmh_mask_node, 'wmh_mask_flair')
                wmh_workflow.connect(truenet_postprocess, 'wmh_prob_map', wmh_mask_node, 'wmh_probmap_flair')

        filter1_wmh_mask_node = Node(IdentityInterface(fields=["filter1_wmh_mask_flair", "filter1_wmh_mask_t1w"]), name="filter1_wmh_mask_node")

        if self.use_bianca_mask:
            if t1w_file == '':
                raise ValueError("[WMH Pipeline] If use_bianca_mask is True, T1w image must be provided.")
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

            wmh_workflow.connect(wmh_mask_node, 'wmh_mask_flair', filter1_wmh_mask_node, 'filter1_wmh_mask_flair')
            wmh_workflow.connect(apply_bianca_mask_node, 'out_file', filter1_wmh_mask_node, 'filter1_wmh_mask_t1w')
        else:
            wmh_workflow.connect(wmh_mask_node, 'wmh_mask_flair', filter1_wmh_mask_node, 'filter1_wmh_mask_flair')
            wmh_workflow.connect(wmh_mask_node, 'wmh_mask_t1w', filter1_wmh_mask_node, 'filter1_wmh_mask_t1w')

        final_wmh_mask_node = Node(IdentityInterface(fields=["final_wmh_mask_flair", "final_wmh_mask_t1w"]), name="final_wmh_mask_node")

        if exclude_mask is not None:
            if t1w_file == '':
                raise ValueError("[WMH Pipeline] If exclude_mask is True, T1w image must be provided.")
            
            exclude_mask_from_wmh_mask = Node(RemoveMaskRegion(), name="exclude_mask_from_wmh_mask")
            wmh_workflow.connect(filter1_wmh_mask_node, 'filter1_wmh_mask_t1w', exclude_mask_from_wmh_mask, 'input_image')
            exclude_mask_from_wmh_mask.inputs.mask_image = exclude_mask
            exclude_mask_from_wmh_mask.inputs.output_image = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-WMH_desc-{self.seg_method}{thr_string}MaskExcluded_mask.nii.gz")

            wmh_workflow.connect(filter1_wmh_mask_node, 'filter1_wmh_mask_flair', final_wmh_mask_node, 'final_wmh_mask_flair')
            wmh_workflow.connect(exclude_mask_from_wmh_mask, 'output_image', final_wmh_mask_node, 'final_wmh_mask_t1w')
        else:
            wmh_workflow.connect(filter1_wmh_mask_node, 'filter1_wmh_mask_flair', final_wmh_mask_node, 'final_wmh_mask_flair')
            wmh_workflow.connect(filter1_wmh_mask_node, 'filter1_wmh_mask_t1w', final_wmh_mask_node, 'final_wmh_mask_t1w')

        # --------------------------------
        # Check T1 <-> MNI non-linear warp
        # --------------------------------
        if t1w_file != '':
            t1_to_mni_warp_node = Node(IdentityInterface(fields=["warp_image"]), name="t1_to_mni_warp_node")
            mni_to_t1_warp_node = Node(IdentityInterface(fields=["warp_image"]), name="mni_to_t1_warp_node")

            # will use the T1w to MNI warp to transform WMH to MNI space
            target_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz')
            target_inverse_warp = os.path.join(self.subject.bids_dir, 'derivatives', 'xfm', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}', f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')

            if not os.path.exists(target_warp) or not os.path.exists(target_inverse_warp):
                print(f"[WMH Pipeline] No existing T1w to MNI warp file found: {target_warp}. Will run Synthmorph registration to get the warp (1mm resolution).")
                print(f"[WMH Pipeline] If you want a different resolution, please run a separate T1 registration pipeline first.")
                t1w_to_mni_registration = Node(SynthmorphNonlinear(), name='t1w_to_mni_registration')
                wmh_workflow.connect(inputnode, 't1w', t1w_to_mni_registration, 't1')
                t1w_to_mni_registration.inputs.mni_template = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'standard', 'MNI152', 'MNI152_T1_1mm_brain.nii.gz')
                t1w_to_mni_registration.inputs.t1_mni_out = os.path.join(os.path.dirname(target_warp), rename_bids_file(t1w_file, {'space': 'MNI152NLin6ASym', 'desc':'brain'}, 'T1w', '.nii.gz'))
                t1w_to_mni_registration.inputs.t1_2_mni_warp = target_warp
                t1w_to_mni_registration.inputs.mni_2_t1_warp = os.path.join(os.path.dirname(target_warp), f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz')
                t1w_to_mni_registration.inputs.register_between_stripped = True

                wmh_workflow.connect(t1w_to_mni_registration, 't1_2_mni_warp', t1_to_mni_warp_node, 'warp_image')
                wmh_workflow.connect(t1w_to_mni_registration, 'mni_2_t1_warp', mni_to_t1_warp_node, 'warp_image')
            else:
                print(f"[WMH Pipeline] Found existing T1w to MNI warp file: {target_warp}. Will use it to transform WMH to MNI space.")
                t1_to_mni_warp_node.inputs.warp_image = target_warp
                mni_to_t1_warp_node.inputs.warp_image = target_inverse_warp

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

                    # Merge native-space LV mask with standard-space LV mask
                    mni_lv_mask_to_native_space = Node(MRIConvertApplyWarp(), name='mni_lv_mask_to_native_space')
                    mni_lv_mask_to_native_space.inputs.input_image = get_package_path('data', 'standard', 'MNI152', 'LV_mask_for_Fazekas.nii.gz')
                    mni_lv_mask_to_native_space.inputs.output_image = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-LateralVentricle_desc-Predefined_mask.nii.gz")
                    wmh_workflow.connect(mni_to_t1_warp_node, 'warp_image', mni_lv_mask_to_native_space, 'warp_image')
                    mni_lv_mask_to_native_space.inputs.interp = 'nearest'

                    merge_lv_masks = Node(CombineMasks(), name='merge_lv_masks')
                    wmh_workflow.connect(extract_ventmask, 'out_nii', merge_lv_masks, 'mask1')
                    wmh_workflow.connect(mni_lv_mask_to_native_space, 'output_image', merge_lv_masks, 'mask2')
                    merge_lv_masks.inputs.output_mask = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_label-LateralVentricle_desc-Merged_mask.nii.gz")

                    # Calculate distancemap
                    distancemap = Node(DistanceMap(), name='distancemap')
                    wmh_workflow.connect(merge_lv_masks, 'output_mask', distancemap, 'in_file')
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

                    wmh_workflow.connect(merge_lv_masks, 'output_mask', fazekas_classification, 'vent_mask')
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

                wmh_workflow.connect(t1_to_mni_warp_node, 'warp_image', wmh_to_mni_transform_node, 'warp_image')

                if 'JHU' in self.location_method:
                    # transform JHU WM atlas to T1w space
                    jhu_to_t1w_transform = Node(MRIConvertApplyWarp(), name='jhu_to_t1w_transform')
                    wmh_workflow.connect(mni_to_t1_warp_node, 'warp_image', jhu_to_t1w_transform, 'warp_image')
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
        import os
        import pandas as pd

        os.makedirs(self.output_path, exist_ok=True)

        wmh_output_path = self.extract_from

        if not wmh_output_path or not os.path.exists(wmh_output_path):
            raise FileNotFoundError(f"extract_from does not exist: {wmh_output_path}")

        jhu_label_map = {
            1: "Middle_cerebellar_peduncle",
            2: "Pontine_crossing_tract",
            3: "Genu_of_corpus_callosum",
            4: "Body_of_corpus_callosum",
            5: "Splenium_of_corpus_callosum",
            6: "Fornix",
            7: "Corticospinal_tract_R",
            8: "Corticospinal_tract_L",
            9: "Medial_lemniscus_R",
            10: "Medial_lemniscus_L",
            11: "Inferior_cerebellar_peduncle_R",
            12: "Inferior_cerebellar_peduncle_L",
            13: "Superior_cerebellar_peduncle_R",
            14: "Superior_cerebellar_peduncle_L",
            15: "Cerebral_peduncle_R",
            16: "Cerebral_peduncle_L",
            17: "Anterior_limb_of_internal_capsule_R",
            18: "Anterior_limb_of_internal_capsule_L",
            19: "Posterior_limb_of_internal_capsule_R",
            20: "Posterior_limb_of_internal_capsule_L",
            21: "Retrolenticular_part_of_internal_capsule_R",
            22: "Retrolenticular_part_of_internal_capsule_L",
            23: "Anterior_corona_radiata_R",
            24: "Anterior_corona_radiata_L",
            25: "Superior_corona_radiata_R",
            26: "Superior_corona_radiata_L",
            27: "Posterior_corona_radiata_R",
            28: "Posterior_corona_radiata_L",
            29: "Posterior_thalamic_radiation_R",
            30: "Posterior_thalamic_radiation_L",
            31: "Sagittal_stratum_R",
            32: "Sagittal_stratum_L",
            33: "External_capsule_R",
            34: "External_capsule_L",
            35: "Cingulum_cingulate_gyrus_R",
            36: "Cingulum_cingulate_gyrus_L",
            37: "Cingulum_hippocampus_R",
            38: "Cingulum_hippocampus_L",
            39: "Fornix_stria_terminalis_R",
            40: "Fornix_stria_terminalis_L",
            41: "Superior_longitudinal_fasciculus_R",
            42: "Superior_longitudinal_fasciculus_L",
            43: "Superior_fronto_occipital_fasciculus_R",
            44: "Superior_fronto_occipital_fasciculus_L",
            45: "Inferior_fronto_occipital_fasciculus_R",
            46: "Inferior_fronto_occipital_fasciculus_L",
            47: "Uncinate_fasciculus_R",
            48: "Uncinate_fasciculus_L",
            49: "Tapetum_R",
            50: "Tapetum_L",
        }

        lobarseg_label_map = {
            52: "basal_ganglia",
            251: "corpus_callosum",
            3001: "left_frontal",
            3002: "left_parietal",
            3003: "left_occipital",
            3004: "left_temporal",
            4001: "right_frontal",
            4002: "right_parietal",
            4003: "right_occipital",
            4004: "right_temporal",
        }

        bullseye_label_map = {
            40011: "frontal_lh1",
            40012: "frontal_lh2",
            40013: "frontal_lh3",
            40014: "frontal_lh4",
            40021: "parietal_lh1",
            40022: "parietal_lh2",
            40023: "parietal_lh3",
            40024: "parietal_lh4",
            40031: "occipital_lh1",
            40032: "occipital_lh2",
            40033: "occipital_lh3",
            40034: "occipital_lh4",
            40041: "temporal_lh1",
            40042: "temporal_lh2",
            40043: "temporal_lh3",
            40044: "temporal_lh4",
            30011: "frontal_rh1",
            30012: "frontal_rh2",
            30013: "frontal_rh3",
            30014: "frontal_rh4",
            30021: "parietal_rh1",
            30022: "parietal_rh2",
            30023: "parietal_rh3",
            30024: "parietal_rh4",
            30031: "occipital_rh1",
            30032: "occipital_rh2",
            30033: "occipital_rh3",
            30034: "occipital_rh4",
            30041: "temporal_rh1",
            30042: "temporal_rh2",
            30043: "temporal_rh3",
            30044: "temporal_rh4",
            521: "bg1",
            522: "bg2",
            523: "bg3",
            524: "bg4",
            2551: "parstriangularis",
            2552: "pericalcarine",
            2553: "postcentral",
            2554: "posteriorcingulate",
        }

        shiva_label_map = {
            1: "Left_Shallow_WM",
            2: "Left_Deep_WM",
            3: "Left_PV_WM",
            4: "Left_Cerebellar",
            5: "Right_Shallow_WM",
            6: "Right_Deep_WM",
            7: "Right_PV_WM",
            8: "Right_Cerebellar",
            9: "Brainstem",
        }

        twmh_cols = ["Subject", "Session", "Total_WMH_Volume", "PWMH_Volume", "DWMH_Volume"]
        jhu_cols = ["Subject", "Session"] + list(jhu_label_map.values())
        lobarseg_cols = ["Subject", "Session"] + list(lobarseg_label_map.values())
        bullseye_cols = ["Subject", "Session"] + list(bullseye_label_map.values())
        shiva_cols = ["Subject", "Session"] + list(shiva_label_map.values())

        twmh_rows = []
        jhu_rows = []
        lobarseg_rows = []
        bullseye_rows = []
        shiva_rows = []

        def _safe_read_csv(csv_path):
            if csv_path is None or not os.path.exists(csv_path):
                return None
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"[WARN] Failed to read CSV: {csv_path}. Error: {e}")
                return None

        def _find_first_csv(base_path, include_keywords, exclude_keywords=None):
            exclude_keywords = exclude_keywords or []
            if not os.path.exists(base_path):
                return None

            matched = []
            for f in os.listdir(base_path):
                if not f.endswith("_volume.csv"):
                    continue
                if all(k in f for k in include_keywords) and not any(k in f for k in exclude_keywords):
                    matched.append(os.path.join(base_path, f))

            matched = sorted(matched)
            return matched[0] if matched else None

        def _extract_single_binary_volume(csv_path):
            df = _safe_read_csv(csv_path)
            if df is None or df.empty:
                return None

            if "Label" not in df.columns or "Volume (mm^3)" not in df.columns:
                return None

            df = df.copy()
            df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
            df["Volume (mm^3)"] = pd.to_numeric(df["Volume (mm^3)"], errors="coerce")

            hit = df.loc[df["Label"] == 1, "Volume (mm^3)"]
            if not hit.empty:
                return hit.iloc[0]

            nonzero_hit = df.loc[df["Label"] != 0, "Volume (mm^3)"]
            if not nonzero_hit.empty:
                return nonzero_hit.iloc[0]

            return None

        def _extract_mapped_volumes(csv_path, label_map):
            result = {v: None for v in label_map.values()}
            df = _safe_read_csv(csv_path)
            if df is None or df.empty:
                return result

            if "Label" not in df.columns or "Volume (mm^3)" not in df.columns:
                return result

            df = df.copy()
            df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
            df["Volume (mm^3)"] = pd.to_numeric(df["Volume (mm^3)"], errors="coerce")

            for label_id, col_name in label_map.items():
                hit = df.loc[df["Label"] == label_id, "Volume (mm^3)"]
                if not hit.empty:
                    result[col_name] = hit.iloc[0]

            return result

        def _append_case(subject_id, session_id, base_path):
            total_csv = _find_first_csv(
                base_path,
                include_keywords=["label-WMH"],
                exclude_keywords=["JHU", "lobarseg", "bullseye", "shivaParcWMH"]
            )
            pwmh_csv = _find_first_csv(base_path, include_keywords=["label-PWMH"])
            dwmh_csv = _find_first_csv(base_path, include_keywords=["label-DWMH"])
            jhu_csv = _find_first_csv(base_path, include_keywords=["label-WMH", "desc-JHU"])
            lobarseg_csv = _find_first_csv(base_path, include_keywords=["label-WMH", "desc-lobarseg"])
            bullseye_csv = _find_first_csv(base_path, include_keywords=["label-WMH", "desc-bullseye"])
            shiva_csv = _find_first_csv(base_path, include_keywords=["desc-shivaParcWMH"])

            twmh_rows.append({
                "Subject": subject_id,
                "Session": session_id,
                "Total_WMH_Volume": _extract_single_binary_volume(total_csv),
                "PWMH_Volume": _extract_single_binary_volume(pwmh_csv),
                "DWMH_Volume": _extract_single_binary_volume(dwmh_csv),
            })

            jhu_row = {"Subject": subject_id, "Session": session_id}
            jhu_row.update(_extract_mapped_volumes(jhu_csv, jhu_label_map))
            jhu_rows.append(jhu_row)

            lobarseg_row = {"Subject": subject_id, "Session": session_id}
            lobarseg_row.update(_extract_mapped_volumes(lobarseg_csv, lobarseg_label_map))
            lobarseg_rows.append(lobarseg_row)

            bullseye_row = {"Subject": subject_id, "Session": session_id}
            bullseye_row.update(_extract_mapped_volumes(bullseye_csv, bullseye_label_map))
            bullseye_rows.append(bullseye_row)

            shiva_row = {"Subject": subject_id, "Session": session_id}
            shiva_row.update(_extract_mapped_volumes(shiva_csv, shiva_label_map))
            shiva_rows.append(shiva_row)

        for subject_folder in sorted(os.listdir(wmh_output_path)):
            subject_folder_path = os.path.join(wmh_output_path, subject_folder)
            if not os.path.isdir(subject_folder_path) or not subject_folder.startswith("sub-"):
                continue

            subject_id = subject_folder.replace("sub-", "", 1)
            session_folders = sorted([
                f for f in os.listdir(subject_folder_path)
                if os.path.isdir(os.path.join(subject_folder_path, f)) and f.startswith("ses-")
            ])

            if session_folders:
                for session_folder in session_folders:
                    session_id = session_folder.replace("ses-", "", 1)
                    session_path = os.path.join(subject_folder_path, session_folder)
                    _append_case(subject_id, session_id, session_path)
            else:
                _append_case(subject_id, "N/A", subject_folder_path)

        twmh_df = pd.DataFrame(twmh_rows, columns=twmh_cols)
        jhu_df = pd.DataFrame(jhu_rows, columns=jhu_cols)
        lobarseg_df = pd.DataFrame(lobarseg_rows, columns=lobarseg_cols)
        bullseye_df = pd.DataFrame(bullseye_rows, columns=bullseye_cols)
        shiva_df = pd.DataFrame(shiva_rows, columns=shiva_cols)

        twmh_df.to_excel(os.path.join(self.output_path, "wmh_total_quantification_results.xlsx"), index=False)
        jhu_df.to_excel(os.path.join(self.output_path, "wmh_jhu_quantification_results.xlsx"), index=False)
        lobarseg_df.to_excel(os.path.join(self.output_path, "wmh_lobarseg_quantification_results.xlsx"), index=False)
        bullseye_df.to_excel(os.path.join(self.output_path, "wmh_bullseye_quantification_results.xlsx"), index=False)
        shiva_df.to_excel(os.path.join(self.output_path, "wmh_shiva_quantification_results.xlsx"), index=False)

        print(f"Results extracted successfully from: {wmh_output_path}")
        print(f"Saved results to: {self.output_path}")