import os
import subprocess
import nibabel as nib
import numpy as np
from nipype.interfaces.freesurfer import ReconAll
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from .freesurfer.recon_all_clinical import ReconAllClinical, CopySynthSR, PostProcess
from .freesurfer.synthSR import SynthSR

from ...bids_data.rename_bids_file import rename_bids_file

class FreesurferPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        Freesurfer pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = kwargs.get('use_which_t1w', None)

    def check_data_requirements(self):
        """
        检查数据需求
        :return: bool
        """
        return self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # create a nipype workflow for the Freesurfer pipeline,
        # rather than running it directly
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
        
        fs_output_path = os.path.dirname(self.output_path)
        fs_output_id = os.path.basename(self.output_path)
        os.makedirs(fs_output_path, exist_ok=True)
        #os.environ["SUBJECTS_DIR"] = fs_output_path

        fs_workflow = Workflow(name='fs_workflow')

        inputnode = Node(IdentityInterface(fields=["t1w_file", "fs_output_id", "subjects_dir"]), name="inputnode")
        #outputnode = Node(IdentityInterface(fields=["fs_output_dir"]), name="outputnode")

        inputnode.inputs.t1w_file = t1w_file
        inputnode.inputs.fs_output_id = fs_output_id
        inputnode.inputs.subjects_dir = fs_output_path

        reconall_node = Node(ReconAll(), name="reconall")
        reconall_node.inputs.directive = "all"
        reconall_node.inputs.flags = '-qcache'
        fs_workflow.connect(inputnode, "t1w_file", reconall_node, "T1_files")
        fs_workflow.connect(inputnode, "fs_output_id", reconall_node, "subject_id")
        fs_workflow.connect(inputnode, "subjects_dir", reconall_node, "subjects_dir")

        fs_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows')

        return fs_workflow

class FreesurferClinicalPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        Freesurfer pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = kwargs.get('use_which_t1w', None)

    def check_data_requirements(self):
        """
        检查数据需求
        :return: bool
        """
        return self.session.get_t1w_files() is not None
        
    def create_workflow(self):
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            t1w_lowres_files = [f for f in t1w_files if 'acq-lowres' in f]
            if len(t1w_lowres_files) == 1:
                print("No specific T1w file selected. Using the one with 'acq-lowres'.")
                t1w_file = t1w_lowres_files[0]
            else:
                print("No specific T1w file selected. Using the first one.")
                t1w_files = [t1w_files[0]]
                t1w_file = t1w_files[0]
        
        # 输出目录为self.output_path的上一级目录
        fs_output_path = os.path.dirname(self.output_path)
        fs_output_id = os.path.basename(self.output_path)

        os.makedirs(fs_output_path, exist_ok=True)

        # change Freesurfer default $SUBJECTS_DIR
        os.environ["SUBJECTS_DIR"] = fs_output_path

        fs_clinical_workflow = Workflow(name='fs_clinical_workflow')

        skip_recon = True
        if not skip_recon:
            inputnode = Node(IdentityInterface(fields=["input_scan", "subject_id", "threads", "subject_dir"]), name="inputnode")
            inputnode.inputs.input_scan = t1w_file
            inputnode.inputs.subject_id = fs_output_id
            inputnode.inputs.threads = 8
            inputnode.inputs.subject_dir = fs_output_path

            recon_all_clinical_node = Node(ReconAllClinical(), name="recon_all_clinical")
            fs_clinical_workflow.connect(inputnode, "input_scan", recon_all_clinical_node, "input_scan")
            fs_clinical_workflow.connect(inputnode, "subject_id", recon_all_clinical_node, "subject_id")
            fs_clinical_workflow.connect(inputnode, "threads", recon_all_clinical_node, "threads")
            fs_clinical_workflow.connect(inputnode, "subject_dir", recon_all_clinical_node, "subject_dir")

            copy_synthsr_node = Node(CopySynthSR(), name="copy_synthsr")
            fs_clinical_workflow.connect(recon_all_clinical_node, "synthsr_raw", copy_synthsr_node, "synthsr_raw")
            fs_clinical_workflow.connect(recon_all_clinical_node, "synthsr_norm", copy_synthsr_node, "synthsr_norm")
            fs_clinical_workflow.connect(inputnode, "input_scan", copy_synthsr_node, "t1w")

            postprocess_node = Node(PostProcess(), name="postprocess")
            fs_clinical_workflow.connect(recon_all_clinical_node, "output_dir", postprocess_node, "fs_output_dir")

            outputnode = Node(IdentityInterface(fields=["output_dir", "synthsr_raw", "synthsr_norm"]), name="outputnode")
            fs_clinical_workflow.connect(postprocess_node, "fs_output_dir", outputnode, "output_dir")
            fs_clinical_workflow.connect(copy_synthsr_node, "synthsr_raw", outputnode, "synthsr_raw")
            fs_clinical_workflow.connect(copy_synthsr_node, "synthsr_norm", outputnode, "synthsr_norm")
        else:
            # assume recon-all-clinical has been run
            inputnode = Node(IdentityInterface(fields=["fs_output_dir"]), name="inputnode")
            inputnode.inputs.fs_output_dir = self.output_path

            postprocess_node = Node(PostProcess(), name="postprocess")
            fs_clinical_workflow.connect(inputnode, "fs_output_dir", postprocess_node, "fs_output_dir")

        # set base directory
        fs_clinical_workflow.base_dir = fs_output_path

        return fs_clinical_workflow

class SynthSRPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        SynthSR pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = kwargs.get('use_which_t1w', None)

    def check_data_requirements(self):
        """
        检查数据需求
        :return: bool
        """
        return self.session.get_t1w_files() is not None

    def create_workflow(self):
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
        
        if len(t1w_files) != 1:
            raise FileNotFoundError("SynthSR requires exactly one T1w file.")
        
        t1w_file = t1w_files[0]
                
        synthsr_workflow = Workflow(name='synthsr_workflow')

        inputnode = Node(IdentityInterface(fields=["t1w_file", "output_path"]), name="inputnode")
        inputnode.inputs.t1w_file = t1w_file
        
        synthsr_img_name = os.path.join(os.path.dirname(t1w_file), rename_bids_file(t1w_file, {'desc': 'SynthSRraw'}, 'T1w', '.nii.gz'))
        inputnode.inputs.output_path = synthsr_img_name

        synthsr_node = Node(SynthSR(), name="synthsr")
        synthsr_workflow.connect(inputnode, "t1w_file", synthsr_node, "input")
        synthsr_workflow.connect(inputnode, "output_path", synthsr_node, "output")

        # set base directory
        synthsr_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows')

        return synthsr_workflow