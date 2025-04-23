import os
import pandas as pd
import subprocess
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function

from .fsl.fsl_anat_nipype import FSLANAT

class FSLANATPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        fsl_anat pipeline
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
            # 确保最终只有1个合适的文件
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]

        if t1w_file is None:
            raise FileNotFoundError("No T1w file found in anat directory.")
        
        # create output directory
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        # copy t1w file to output directory
        t1w_file = os.path.abspath(t1w_file)

        fsl_anat_wf = Workflow(name='fsl_anat_wf', base_dir=self.output_path)

        inputnode = Node(IdentityInterface(fields=['input_image', 'output_directory']),
                         name='inputnode')
        
        inputnode.inputs.input_image = t1w_file
        inputnode.inputs.output_directory = os.path.join(self.output_path, 'fsl')

        fsl_anat_node = Node(FSLANAT(), name='fsl_anat_node')
        fsl_anat_wf.connect(inputnode, 'input_image', fsl_anat_node, 'input_image')
        fsl_anat_wf.connect(inputnode, 'output_directory', fsl_anat_node, 'output_directory')

        outputnode = Node(IdentityInterface(fields=['output_directory']), name='outputnode')
        fsl_anat_wf.connect(fsl_anat_node, 'output_directory', outputnode, 'output_directory')

        return fsl_anat_wf