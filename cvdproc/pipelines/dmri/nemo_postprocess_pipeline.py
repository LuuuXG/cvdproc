import os
import pandas as pd
import json
import nibabel as nib
import numpy as np
import re

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from cvdproc.pipelines.dmri.nemo.nemo_postprocess_nipype import NemoPostprocess

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class NemoPostprocessPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        Nemo postprocessing pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_freesurfer_clinical = kwargs.get('use_freesurfer_clinical', False) # whether to use freesurfer clinical directory. False will use freesurfer

        self.extract_from = kwargs.get('extract_from', None)

    def check_data_requirements(self):
        """
        :return: bool
        """
        return self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        # Search for nemo output directory
        nemo_output_dir = self.session._find_output('nemo')
        if nemo_output_dir is None:
            raise FileNotFoundError("Nemo output directory not found. Please run Nemo pipeline first.")

        fs_output_dir = self.session._find_output('freesurfer_clinical' if self.use_freesurfer_clinical else 'freesurfer')

        fs_parent_dir = os.path.dirname(fs_output_dir)

        # List all session directories under it that start with 'ses-'
        freesurfer_output_dirs = [
            os.path.join(fs_parent_dir, d)
            for d in os.listdir(fs_parent_dir)
            if d.startswith('ses-') and os.path.isdir(os.path.join(fs_parent_dir, d))
        ]
        
        #################
        # Main Workflow #
        #################
        nemo_postprocess_wf = Workflow(name='nemo_postprocess_wf')
        nemo_postprocess_wf.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows')

        # Input node
        input_node = Node(IdentityInterface(fields=['nemo_output_dir', 'nemo_postprocessed_dir', 'freesurfer_output_dirs', 'csv_output']),
                          name='input_node')
        input_node.inputs.nemo_output_dir = nemo_output_dir
        input_node.inputs.nemo_postprocessed_dir = os.path.join(self.output_path, 'nemo_postprocessed')
        input_node.inputs.freesurfer_output_dirs = freesurfer_output_dirs
        input_node.inputs.output_csv_dir = os.path.join(self.output_path, 'weighted_cortical_metrics')

        # Postprocessing node
        postprocess_node = Node(NemoPostprocess(), name='postprocess_node')
        nemo_postprocess_wf.connect([
            (input_node, postprocess_node, [('nemo_output_dir', 'nemo_output_dir'),
                                            ('nemo_postprocessed_dir', 'nemo_postprocessed_dir'),
                                            ('freesurfer_output_dirs', 'freesurfer_output_dirs'),
                                            ('output_csv_dir', 'output_csv_dir')])
        ])

        return nemo_postprocess_wf
    
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)
        nemo_postprocessed_dir = self.extract_from

        if not os.path.exists(nemo_postprocessed_dir):
            raise FileNotFoundError(f"Nemo postprocessed directory {nemo_postprocessed_dir} does not exist.")
        
        print(f"Reading .csv results from {nemo_postprocessed_dir}...")

        merged_dfs = {}  # {chacovol_id: list of DataFrames}

        for subject_folder in os.listdir(nemo_postprocessed_dir):
            if not subject_folder.startswith('sub-'):
                continue

            subject_id = subject_folder.split('-')[1]
            subject_folder_path = os.path.join(nemo_postprocessed_dir, subject_folder)

            for session_folder in os.listdir(subject_folder_path):
                if not session_folder.startswith('ses-'):
                    continue

                session_id = session_folder.split('-')[1]
                nemo_id = f"sub-{subject_id}_ses-{session_id}"
                session_path = os.path.join(subject_folder_path, session_folder)
                nemo_csv_dir = os.path.join(session_path, 'weighted_cortical_metrics')

                if not os.path.exists(nemo_csv_dir):
                    continue

                for file in os.listdir(nemo_csv_dir):
                    if not file.endswith('.csv'):
                        continue

                    match = re.search(r'(nemo_output_.*)_cortical_metrics\\.csv', file)
                    if not match:
                        continue  # not a valid chacovol file

                    chacovol_id = match.group(1)
                    csv_path = os.path.join(nemo_csv_dir, file)

                    try:
                        df = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Failed to read {csv_path}: {e}")
                        continue

                    # Insert subject info as first two columns
                    df.insert(0, 'subject_id', subject_id)
                    df.insert(0, 'nemo_id', nemo_id)

                    if chacovol_id not in merged_dfs:
                        merged_dfs[chacovol_id] = []

                    merged_dfs[chacovol_id].append(df)

        # Save merged CSVs
        output_dir = os.path.join(self.output_path, "merged_csvs")
        os.makedirs(output_dir, exist_ok=True)

        for chacovol_id, dfs in merged_dfs.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(output_dir, f"{chacovol_id}_merged.csv")
            merged_df.to_csv(out_path, index=False)
            print(f"Saved merged results: {out_path}")