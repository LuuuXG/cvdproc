import os
import pandas as pd
import subprocess
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function

from cvdproc.pipelines.smri.anat_seg.cp_seg.chpseg import ChPSeg
from cvdproc.pipelines.smri.freesurfer.synthseg import SynthSeg, SynthSegPostProcess

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class AnatSegPipeline:
    """
    Generating anatomical segmentation atlas from sMRI
    """
    def __init__(self,
                 subject: object,
                 session: object,
                 output_path: str,
                 use_which_t1w: str = "T1w",
                 methods: list = ["synthseg", "chpseg"],
                 cpu_first: bool = False,
                 extract_from: str = None,
                 **kwargs):
        """
        Generating anatomical segmentation atlas from sMRI

        Args:
            subject: Subject object
            session: Session object
            output_path: Output directory
            use_which_t1w: specific string to select T1w image, e.g. 'acq-highres'. If None, T1w image is not used
            methods: List of methods to use. Options include 'synthseg' and 'chpseg'.
            cpu_first: Whether to use CPU first for SynthSeg (if available).
            extract_from: Path to extract results from
            **kwargs: Additional arguments
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.use_which_t1w = use_which_t1w
        self.methods = methods
        self.cpu_first = cpu_first
        self.extract_from = extract_from
        self.kwargs = kwargs
    
    def check_data_requirements(self):
        return self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # Get the T1w file
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
        print(f"[ANAT SEG] Using T1w file: {t1w_file}")
        print(f"[ANAT SEG] Using method: {self.methods}")

        # Create the workflow
        anatseg_workflow = Workflow(name='anatseg_workflow')
        anatseg_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', f'sub-{self.subject.subject_id}', f'ses-{self.session.session_id}')

        inputnode = Node(IdentityInterface(fields=['t1w_file']),
                         name='inputnode')
        inputnode.inputs.t1w_file = t1w_file

        # outputnode = Node(IdentityInterface(fields=['synthseg_out', 'chpseg_out']),
        #                   name='outputnode')

        if 'synthseg' in self.methods:
            synthseg = Node(SynthSeg(), name='synthseg')
            anatseg_workflow.connect(inputnode, 't1w_file', synthseg, 'image')
            synthseg.inputs.out = os.path.join(self.output_path, 'synthseg', rename_bids_file(t1w_file, {'space': 'T1w'}, 'synthseg', '.nii.gz'))
            synthseg.inputs.vol = os.path.join(self.output_path, 'synthseg', rename_bids_file(t1w_file, {'space': 'T1w', 'desc': "synthseg"}, 'volumes', '.csv'))
            synthseg.inputs.robust = True
            synthseg.inputs.parc = True
            synthseg.inputs.keepgeom = True
            if self.cpu_first:
                synthseg.inputs.cpu = True
            
            synthseg_postprocess = Node(SynthSegPostProcess(), name='synthseg_postprocess')
            anatseg_workflow.connect(synthseg, 'out', synthseg_postprocess, 'synthseg_input')
            synthseg_postprocess.inputs.wm_output = os.path.join(self.output_path, 'synthseg', rename_bids_file(t1w_file, {'space': 'T1w', 'desc': "WM"}, 'mask', '.nii.gz'))

        
        if 'chpseg' in self.methods:
            chpseg = Node(ChPSeg(), name='chpseg')
            anatseg_workflow.connect(inputnode, 't1w_file', chpseg, 'in_t1')
            chpseg.inputs.output_dir = os.path.join(self.output_path, 'chpseg')
        
        return anatseg_workflow

    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        anat_seg_output_path = self.extract_from

        # SynthSeg output extraction: empty dataframe
        synthseg_columns = [
            'Subject', 'Session',
            'total intracranial', 'left cerebral white matter', 'left cerebral cortex', 'left lateral ventricle', 'left inferior lateral ventricle', 'left cerebellum white matter', 'left cerebellum cortex', 'left thalamus', 'left caudate', 'left putamen', 'left pallidum', 
            '3rd ventricle', '4th ventricle', 'brain-stem', 
            'left hippocampus', 'left amygdala', 'csf', 'left accumbens area', 'left ventral DC', 
            'right cerebral white matter', 'right cerebral cortex', 'right lateral ventricle', 'right inferior lateral ventricle', 'right cerebellum white matter', 'right cerebellum cortex', 'right thalamus', 'right caudate', 'right putamen', 'right pallidum', 'right hippocampus', 'right amygdala', 'right accumbens area', 'right ventral DC',
            'ctx-lh-bankssts', 'ctx-lh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-lh-entorhinal', 'ctx-lh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-lh-inferiortemporal', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital', 'ctx-lh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-lh-medialorbitofrontal', 'ctx-lh-middletemporal', 'ctx-lh-parahippocampal', 'ctx-lh-paracentral', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-parstriangularis', 'ctx-lh-pericalcarine', 'ctx-lh-postcentral', 'ctx-lh-posteriorcingulate', 'ctx-lh-precentral', 'ctx-lh-precuneus', 'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal', 'ctx-lh-superiorfrontal', 'ctx-lh-superiorparietal', 'ctx-lh-superiortemporal', 'ctx-lh-supramarginal', 'ctx-lh-frontalpole', 'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 'ctx-lh-insula',
            'ctx-rh-bankssts', 'ctx-rh-caudalanteriorcingulate', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-cuneus', 'ctx-rh-entorhinal', 'ctx-rh-fusiform', 'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital', 'ctx-rh-lateralorbitofrontal', 'ctx-rh-lingual', 'ctx-rh-medialorbitofrontal', 'ctx-rh-middletemporal', 'ctx-rh-parahippocampal', 'ctx-rh-paracentral', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-parstriangularis', 'ctx-rh-pericalcarine', 'ctx-rh-postcentral', 'ctx-rh-posteriorcingulate', 'ctx-rh-precentral', 'ctx-rh-precuneus', 'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal', 'ctx-rh-superiorfrontal', 'ctx-rh-superiorparietal', 'ctx-rh-superiortemporal', 'ctx-rh-supramarginal', 'ctx-rh-frontalpole', 'ctx-rh-temporalpole', 'ctx-rh-transversetemporal', 'ctx-rh-insula'
        ]
        synthseg_df = pd.DataFrame(columns=synthseg_columns)

        # Chpseg output extraction: empty dataframe
        chpseg_columns = [
            'Subject', 'Session',
            'total_volume_mm3', 'volume_right_mm3', 'volume_left_mm3'
        ]
        chpseg_df = pd.DataFrame(columns=chpseg_columns)

        for subject_folder in os.listdir(anat_seg_output_path):
            subject_path = os.path.join(anat_seg_output_path, subject_folder)
            # session level
            for session_folder in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session_folder)

                # subfolder level: synthseg and/or chpseg
                for method_folder in os.listdir(session_path):
                    method_path = os.path.join(session_path, method_folder)

                    if method_folder == 'synthseg':
                        # read volume csv
                        vol_files = [f for f in os.listdir(method_path) if f.endswith('desc-synthseg_volumes.csv')]
                        if len(vol_files) == 1:
                            vol_file_path = os.path.join(method_path, vol_files[0])
                            vol_data = pd.read_csv(vol_file_path)
                            vol_data = vol_data.drop(columns=[vol_data.columns[0]])
                            vol_data.insert(0, 'Session', session_folder)
                            vol_data.insert(0, 'Subject', subject_folder)
                            synthseg_df = pd.concat([synthseg_df, vol_data], ignore_index=True)
                    elif method_folder == 'chpseg':
                        # read volume csv
                        vol_files = [f for f in os.listdir(method_path) if f.endswith('chp_volumes.csv')]
                        if len(vol_files) == 1:
                            vol_file_path = os.path.join(method_path, vol_files[0])
                            vol_data = pd.read_csv(vol_file_path)
                            vol_data = vol_data.drop(columns=[vol_data.columns[0], vol_data.columns[1]])
                            vol_data.insert(0, 'Session', session_folder)
                            vol_data.insert(0, 'Subject', subject_folder)
                            chpseg_df = pd.concat([chpseg_df, vol_data], ignore_index=True)
                
                print(f"[ANAT SEG] Extracted results for subject {subject_folder}, session {session_folder}")
        # Save the dataframes to CSV
        if not synthseg_df.empty:
            synthseg_output_file = os.path.join(self.output_path, 'synthseg_volumes_summary.csv')
            synthseg_df.to_csv(synthseg_output_file, index=False)
            print(f"SynthSeg volumes summary saved to {synthseg_output_file}")
        if not chpseg_df.empty:
            chpseg_output_file = os.path.join(self.output_path, 'chpseg_volumes_summary.csv')
            chpseg_df.to_csv(chpseg_output_file, index=False)
            print(f"ChPSeg volumes summary saved to {chpseg_output_file}")