import os
import subprocess
import nibabel as nib
import numpy as np
from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge

from cvdproc.pipelines.smri.hipsta.hipsta_nipype import Hipsta, HipstaDocker

from ...bids_data.rename_bids_file import rename_bids_file

class HipstaPipeline:
    def __init__(self,
                 subject,
                 session,
                 output_path,
                 use_freesurfer_clinical: bool = False,
                 extract_from: str = None,
                 **kwargs):
        """
        Hippocampal Shape and Thickness Analysis (HIPSTA) pipeline.

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Output directory to save results.
            use_freesurfer_clinical (bool, optional): Whether to use recon-all-clinical.sh outputs. Default is False.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.use_freesurfer_clinical = use_freesurfer_clinical
        self.extract_from = extract_from

    def check_data_requirements(self):
        # currently no specific data requirements
        return True
    
    def create_workflow(self):
        hipsta_workflow = Workflow(name='hipsta_workflow')

        # handle freesurfer outputs
        if self.use_freesurfer_clinical:
            if self.session.freesurfer_clinical_dir is None:
                raise FileNotFoundError("[HIPSTA Pipeline] Freesurfer clinical directory not found, but use_freesurfer_clinical is set to True.")
            
            fs_subjects_dir = os.path.dirname(self.session.freesurfer_clinical_dir)
            fs_subject_id = os.path.basename(self.session.freesurfer_clinical_dir)

        else:
            if self.session.freesurfer_dir is None:
                raise FileNotFoundError("[HIPSTA Pipeline] Freesurfer directory not found.")
            
            fs_subjects_dir = os.path.dirname(self.session.freesurfer_dir)
            fs_subject_id = os.path.basename(self.session.freesurfer_dir)

        lh_hip_subfield_seg = os.path.join(fs_subjects_dir, fs_subject_id, 'mri', 'lh.hippoAmygLabels-T1.v22.mgz')
        rh_hip_subfield_seg = os.path.join(fs_subjects_dir, fs_subject_id, 'mri', 'rh.hippoAmygLabels-T1.v22.mgz')

        if not os.path.exists(lh_hip_subfield_seg) or not os.path.exists(rh_hip_subfield_seg):
            raise FileNotFoundError("[HIPSTA Pipeline] One or both hippocampal subfield segmentations not found.")

        inputnode = Node(IdentityInterface(fields=['hip_subfield_seg']),
                         name='inputnode')
        inputnode.inputs.hip_subfield_seg = [lh_hip_subfield_seg, rh_hip_subfield_seg]

        hipsta_node = MapNode(HipstaDocker(), name='hipsta_node', iterfield=['filename', 'hemi'])
        hipsta_workflow.connect(inputnode, 'hip_subfield_seg', hipsta_node, 'filename')
        hipsta_node.inputs.hemi = ['lh', 'rh']
        hipsta_node.inputs.outputdir = self.output_path
        hipsta_node.inputs.lut = 'freesurfer'

        return hipsta_workflow
    
    def extract_results(self):
        import pandas as pd

        os.makedirs(self.output_path, exist_ok=True)

        hipsta_output_path = self.extract_from
        if hipsta_output_path is None:
            raise ValueError("[HIPSTA Pipeline] 'extract_from' is None.")

        if not os.path.isdir(hipsta_output_path):
            raise FileNotFoundError(f"[HIPSTA Pipeline] extract_from does not exist: {hipsta_output_path}")

        lh_rows = []
        rh_rows = []

        def flatten_thickness_csv(csv_file, hemi, subject_id, session_id):
            df = pd.read_csv(csv_file)

            first_col = df.columns[0]
            df = df.rename(columns={first_col: "x"})

            long_df = df.melt(
                id_vars="x",
                var_name="y",
                value_name="thickness"
            )

            long_df["x_num"] = long_df["x"].str.replace("x", "", regex=False).astype(int)
            long_df["y_num"] = long_df["y"].str.replace("y", "", regex=False).astype(int)

            long_df = long_df.sort_values(["x_num", "y_num"]).reset_index(drop=True)

            row_dict = {
                "Subject": subject_id,
                "Session": session_id,
            }

            for _, row in long_df.iterrows():
                col_name = f"{hemi}_{row['x']}_{row['y']}"
                row_dict[col_name] = row["thickness"]

            return row_dict

        for subject_folder in sorted(os.listdir(hipsta_output_path)):
            if not subject_folder.startswith("sub-"):
                continue

            subject_id = subject_folder.split("-", 1)[1]
            subject_folder_path = os.path.join(hipsta_output_path, subject_folder)

            if not os.path.isdir(subject_folder_path):
                continue

            for session_folder in sorted(os.listdir(subject_folder_path)):
                if not session_folder.startswith("ses-"):
                    continue

                session_id = session_folder.split("-", 1)[1]
                session_path = os.path.join(subject_folder_path, session_folder)

                if not os.path.isdir(session_path):
                    continue

                lh_thickness_csv = os.path.join(session_path, "thickness", "lh.grid-segments-z.csv")
                rh_thickness_csv = os.path.join(session_path, "thickness", "rh.grid-segments-z.csv")

                if os.path.isfile(lh_thickness_csv):
                    lh_rows.append(
                        flatten_thickness_csv(
                            csv_file=lh_thickness_csv,
                            hemi="lh",
                            subject_id=f'sub-{subject_id}',
                            session_id=f'ses-{session_id}',
                        )
                    )

                if os.path.isfile(rh_thickness_csv):
                    rh_rows.append(
                        flatten_thickness_csv(
                            csv_file=rh_thickness_csv,
                            hemi="rh",
                            subject_id=f'sub-{subject_id}',
                            session_id=f'ses-{session_id}',
                        )
                    )

        if len(lh_rows) == 0 and len(rh_rows) == 0:
            raise FileNotFoundError(
                f"[HIPSTA Pipeline] No lh/rh.grid-segments-z.csv files were found under: {hipsta_output_path}"
            )

        if len(lh_rows) > 0:
            lh_df = pd.DataFrame(lh_rows)
            lh_value_cols = sorted(
                [c for c in lh_df.columns if c.startswith("lh_")],
                key=lambda x: (
                    int(x.split("_")[1].replace("x", "")),
                    int(x.split("_")[2].replace("y", ""))
                )
            )
            lh_df = lh_df[["Subject", "Session"] + lh_value_cols]
            lh_outfile = os.path.join(self.output_path, "lh_thickness_summary.xlsx")
            lh_df.to_excel(lh_outfile, index=False)
            print(f"[HIPSTA Pipeline] Saved: {lh_outfile}")

        if len(rh_rows) > 0:
            rh_df = pd.DataFrame(rh_rows)
            rh_value_cols = sorted(
                [c for c in rh_df.columns if c.startswith("rh_")],
                key=lambda x: (
                    int(x.split("_")[1].replace("x", "")),
                    int(x.split("_")[2].replace("y", ""))
                )
            )
            rh_df = rh_df[["Subject", "Session"] + rh_value_cols]
            rh_outfile = os.path.join(self.output_path, "rh_thickness_summary.xlsx")
            rh_df.to_excel(rh_outfile, index=False)
            print(f"[HIPSTA Pipeline] Saved: {rh_outfile}")

