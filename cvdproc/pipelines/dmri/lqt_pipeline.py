import os
import pandas as pd
import json
import nibabel as nib
import numpy as np
import re
import glob

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from cvdproc.pipelines.dmri.lqt.lqt_nipype import LQT

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class LQTPipeline:
    """
    Using the Lesion Quantification Toolkit (LQT) to quantify lesion disconnection.
    """
    def __init__(self, 
                 subject, 
                 session, 
                 output_path,
                 seed_mask: str = 'lesion_mask', 
                 use_which_mask: str = 'infarction',
                 alps_roi_disconnection: bool = False,
                 extract_from: str = None,
                 **kwargs):
        """
        LQT Pipeline for lesion disconnection analysis.

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Directory to save outputs.
            seed_mask (str, optional): Name of the seed mask folder in 'derivatives'. For example, if the ROI mask path is 'derivatives/lesion_mask/sub-XXX/ses-XXX/*infarction.nii.gz', then seed_mask='lesion_mask'.
            use_which_mask (str, optional): Keyword to select the desired lesion mask. Default is 'infarction'. For example, if the lesion mask is 'derivatives/lesion_mask/sub-XXX/ses-XXX/*infarction.nii.gz', then use_which_mask='infarction'.
            alps_roi_disconnection (bool, optional): If True, calculate disconnection in the ALPS ROI.
            extract_from (str, optional): If extracting results, please provide it.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.seed_mask = seed_mask
        self.use_which_mask = use_which_mask
        self.alps_roi_disconnection = alps_roi_disconnection

        self.extract_from = extract_from

    def check_data_requirements(self):
        """
        Will always return True, as the LQT pipeline will check the seed mask in MNI space during creation of the workflow.
        """
        return True
    
    def create_workflow(self):
        # Find the lesion file
        seed_mask_dir = os.path.join(self.subject.bids_dir, 'derivatives', self.seed_mask, 'sub-' + self.subject.subject_id, 'ses-' + self.session.session_id)
        # Search for the file containing the self.use_which_mask (if multiple, return the first one)
        lesion_files = [f for f in os.listdir(seed_mask_dir) if self.use_which_mask in f and f.endswith('.nii.gz')]
        if not lesion_files:
            raise FileNotFoundError(f"No lesion file found in {seed_mask_dir} containing '{self.use_which_mask}'")
        lesion_file = os.path.join(seed_mask_dir, lesion_files[0])

        lqt_workflow = Workflow(name='lqt_workflow')
        lqt_workflow.base_dir = os.path.join(self.subject.bids_dir, 'derivatives', 'workflows', 'sub-' + self.subject.subject_id, 'ses-' + self.session.session_id)

        # Input node
        input_node = Node(IdentityInterface(fields=['patient_id', 'lesion_file', 'output_dir', 'parcel_path', 'lqt_script']),
                          name='input_node')
        input_node.inputs.patient_id = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}"
        input_node.inputs.lesion_file = lesion_file
        input_node.inputs.output_dir = self.output_path
        input_node.inputs.parcel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lqt', 'extdata', 'Schaefer_Yeo_Plus_Subcort', '100Parcels7Networks.nii.gz'))
        input_node.inputs.lqt_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'r', 'lqt', 'lqt_single_subject.R'))

        lqt_process_node = Node(LQT(), name='lqt_process')
        lqt_workflow.connect([
            (input_node, lqt_process_node, [('patient_id', 'patient_id'),
                                            ('lesion_file', 'lesion_file'),
                                            ('output_dir', 'output_dir'),
                                            ('parcel_path', 'parcel_path'),
                                            ('lqt_script', 'lqt_script')])
        ])
        lqt_process_node.inputs.dsi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lqt', 'extdata', 'DSI_studio', 'dsi-studio', 'dsi_studio'))

        if self.alps_roi_disconnection:
            from cvdproc.pipelines.dmri.lqt.lqt_alps_disconnection import LQTALPSDisconnection
            alps_node = Node(LQTALPSDisconnection(), name='alps_disconnection')
            lqt_workflow.connect([
                (lqt_process_node, alps_node, [('postprocessed_percent_tdi_file', 'tdi_file')])
            ])
            alps_node.inputs.output_csv = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-ALPSdisconnection_metrics.csv")

        return lqt_workflow
    
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        lqt_output_path = self.extract_from
        if lqt_output_path is None:
            raise ValueError("extract_from is None. Please provide the LQT derivatives directory to extract results from.")

        lqt_output_path = os.path.abspath(lqt_output_path)

        columns = [
            "Subject",
            "Session",
            "L_SCR_mean_disconnection",
            "L_SLF_mean_disconnection",
            "R_SCR_mean_disconnection",
            "R_SLF_mean_disconnection",
            "L_ALPS_mean_disconnection",
            "R_ALPS_mean_disconnection",
        ]

        alps_disconnection_results_df = pd.DataFrame(columns=columns)

        def read_alps_disconnection_csv(csv_path):
            if csv_path is None or not os.path.exists(csv_path):
                return None

            df = pd.read_csv(csv_path)
            if df.empty:
                return None

            required_cols = [
                "L_SCR_mean_disconnection",
                "L_SLF_mean_disconnection",
                "R_SCR_mean_disconnection",
                "R_SLF_mean_disconnection",
                "L_ALPS_mean_disconnection",
                "R_ALPS_mean_disconnection",
            ]

            row = {}
            for col in required_cols:
                row[col] = df[col].values[0] if col in df.columns else np.nan

            return row

        def find_alps_disconnection_csv(session_path, subject_id, session_id):
            expected_name = f"sub-{subject_id}_ses-{session_id}_desc-ALPSdisconnection_metrics.csv"

            candidate_paths = [
                os.path.join(session_path, expected_name),
                os.path.join(session_path, f"sub-{subject_id}_ses-{session_id}", expected_name),
                os.path.join(session_path, f"sub-{subject_id}_ses-{session_id}", "ALPS_Disconnection", expected_name),
                os.path.join(session_path, "ALPS_Disconnection", expected_name),
            ]

            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    return candidate

            recursive_matches = glob.glob(
                os.path.join(session_path, "**", expected_name),
                recursive=True
            )

            if len(recursive_matches) > 0:
                return sorted(recursive_matches)[0]

            generic_matches = glob.glob(
                os.path.join(session_path, "**", "*desc-ALPSdisconnection_metrics.csv"),
                recursive=True
            )

            if len(generic_matches) > 0:
                return sorted(generic_matches)[0]

            return None

        print(f"Reading LQT results from {lqt_output_path}...")

        if not os.path.isdir(lqt_output_path):
            raise FileNotFoundError(f"LQT output directory not found: {lqt_output_path}")

        for subject_folder in sorted(os.listdir(lqt_output_path)):
            if not subject_folder.startswith("sub-"):
                continue

            subject_id = subject_folder.split("-", 1)[1]
            subject_folder_path = os.path.join(lqt_output_path, subject_folder)

            if not os.path.isdir(subject_folder_path):
                continue

            session_folders = [
                f for f in sorted(os.listdir(subject_folder_path))
                if f.startswith("ses-") and os.path.isdir(os.path.join(subject_folder_path, f))
            ]

            if len(session_folders) == 0:
                session_folders = [""]

            for session_folder in session_folders:
                if session_folder:
                    session_id = session_folder.split("-", 1)[1]
                    session_path = os.path.join(subject_folder_path, session_folder)
                else:
                    session_id = "N/A"
                    session_path = subject_folder_path

                if not os.path.isdir(session_path):
                    continue

                alps_disconnection_csv = find_alps_disconnection_csv(
                    session_path=session_path,
                    subject_id=subject_id,
                    session_id=session_id,
                )

                metric_dict = read_alps_disconnection_csv(alps_disconnection_csv)

                if metric_dict is None:
                    continue

                row = {
                    "Subject": f"sub-{subject_id}",
                    "Session": f"ses-{session_id}" if session_id != "N/A" else "N/A",
                }
                row.update(metric_dict)

                alps_disconnection_results_df = pd.concat(
                    [alps_disconnection_results_df, pd.DataFrame([row])],
                    ignore_index=True,
                )

        output_csv = os.path.join(self.output_path, "alps_roi_disconnection_results.csv")
        output_excel = os.path.join(self.output_path, "alps_roi_disconnection_results.xlsx")

        if not alps_disconnection_results_df.empty:
            alps_disconnection_results_df.to_csv(output_csv, index=False)
            alps_disconnection_results_df.to_excel(output_excel, header=True, index=False)
            print(f"ALPS ROI disconnection results saved to {output_csv}")
            print(f"ALPS ROI disconnection results saved to {output_excel}")
        else:
            print("No ALPS ROI disconnection results found.")

        return alps_disconnection_results_df