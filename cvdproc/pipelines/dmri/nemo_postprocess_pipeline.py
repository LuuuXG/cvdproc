import os
import pandas as pd
import json
import nibabel as nib
import numpy as np
import re

from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from cvdproc.pipelines.dmri.nemo.nemo_postprocess_nipype import NemoCorticalMetrics, NemoChacovol, NemoChacoconnSC, NemoChacoconn

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class NemoPostprocessPipeline:
    def __init__(self, 
                 subject: object, 
                 session: object, 
                 output_path: str, 
                 use_freesurfer_clinical: bool = False,
                 use_freesurfer_longitudinal: bool = False,
                 cortical_metrics: bool = False,
                 results_to_csv: bool = False,
                 extract_from: str = "",
                 **kwargs):
        """
        NeMo Postprocessing Pipeline

        Postprocesses outputs from the NeMo (Network Measures) toolbox, extracting
        cortical metrics, subcortical volumes (ChacoVol), and structural connectivity
        matrices (ChacoConn) for population-level analysis.

        Args:
            subject: BIDSSubject object
            session: BIDSSession object
            output_path: output directory for the pipeline
            use_freesurfer_clinical: whether to use recon-all-clinical.sh outputs instead of standard recon-all. Defaults to False.
            use_freesurfer_longitudinal: whether to use longitudinal FreeSurfer outputs. Defaults to False.
            cortical_metrics: whether to extract weighted cortical metrics from NeMo outputs. Defaults to False.
            results_to_csv: whether to convert ChacoVol/ChacoConn results to CSV format. Defaults to False.
            extract_from: path to extract results from for population-level summary.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_freesurfer_clinical = use_freesurfer_clinical
        self.use_freesurfer_longitudinal = use_freesurfer_longitudinal
        self.cortical_metrics = cortical_metrics
        self.results_to_csv = results_to_csv

        self.extract_from = extract_from

    def check_data_requirements(self):
        """
        :return: bool
        """
        return True
    
    def create_workflow(self):
        # Search for nemo output directory
        nemo_output_dir = self.session._find_output('nemo')
        if nemo_output_dir is None:
            raise FileNotFoundError("Nemo output directory not found. Please run Nemo pipeline first.")

        if self.cortical_metrics:
            fs_output_dir = self.session._find_output('freesurfer_clinical' if self.use_freesurfer_clinical else 'freesurfer')

            # ---------------------------------------------
            # Determine FreeSurfer session/long directories
            # ---------------------------------------------
            subject_id = f"sub-{self.subject.subject_id}"

            # fs_output_dir could be:
            #   - .../derivatives/freesurfer/sub-XXX
            #   - .../derivatives/freesurfer/sub-XXX/ses-baseline
            #   - .../derivatives/freesurfer/sub-XXX/ses-baseline.long.sub-XXX
            # We want fs_subject_root = .../derivatives/freesurfer/sub-XXX
            fs_subject_root = fs_output_dir
            base_name = os.path.basename(fs_subject_root)

            if base_name.startswith("ses-"):
                fs_subject_root = os.path.dirname(fs_subject_root)

            if not os.path.isdir(fs_subject_root):
                raise FileNotFoundError(f"FreeSurfer subject root not found: {fs_subject_root}")

            # Patterns:
            # - cross-sectional: ses-<tp> with NO '.' anywhere in folder name
            # - longitudinal: ses-<tp>.long.sub-<ID>
            ses_cross_pat = re.compile(r"^ses-[^.]+$")  # crucial: exclude any '.' (thus excluding *.long.*)
            ses_long_pat = re.compile(rf"^ses-[^.]+\.long\.{re.escape(subject_id)}$")

            freesurfer_output_dirs = []
            for d in sorted(os.listdir(fs_subject_root)):
                p = os.path.join(fs_subject_root, d)
                if not os.path.isdir(p):
                    continue

                if self.use_freesurfer_clinical:
                    # clinical is cross-sectional, also exclude *.long.*
                    if ses_cross_pat.match(d):
                        freesurfer_output_dirs.append(p)

                else:
                    if self.use_freesurfer_longitudinal:
                        if ses_long_pat.match(d):
                            freesurfer_output_dirs.append(p)
                    else:
                        # cross-sectional freesurfer: exclude *.long.* explicitly
                        if ses_cross_pat.match(d):
                            freesurfer_output_dirs.append(p)

            if len(freesurfer_output_dirs) == 0:
                mode = "clinical" if self.use_freesurfer_clinical else ("longitudinal" if self.use_freesurfer_longitudinal else "cross-sectional")
                raise FileNotFoundError(
                    f"No FreeSurfer directories found under {fs_subject_root} for mode={mode}. "
                    f"Expected {'ses-* (without .)' if mode != 'longitudinal' else f'ses-*.long.{subject_id}'}"
                )
        else:
            freesurfer_output_dirs = []

        #################
        # Main Workflow #
        #################
        nemo_postprocess_wf = Workflow(name='nemo_postprocess_wf')

        nemo_postprocessed_dir = os.path.join(nemo_output_dir, 'postprocess')
        os.makedirs(nemo_postprocessed_dir, exist_ok=True)

        # Input node
        input_node = Node(IdentityInterface(fields=['nemo_output_dir', 'nemo_postprocessed_dir']),
                          name='input_node')
        input_node.inputs.nemo_output_dir = nemo_output_dir
        input_node.inputs.nemo_postprocessed_dir = os.path.join(nemo_output_dir, 'postprocess')

        bids_prefix = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}"
        nemo_cortical_metrics = Node(NemoCorticalMetrics(), name='nemo_cortical_metrics')
        nemo_postprocess_wf.connect([
            (input_node, nemo_cortical_metrics, [('nemo_output_dir', 'nemo_output_dir'),
                                                  ('nemo_postprocessed_dir', 'nemo_postprocessed_dir')])
        ])
        nemo_cortical_metrics.inputs.bids_prefix = bids_prefix
        nemo_cortical_metrics.inputs.mean_only = True
        if self.cortical_metrics:
            nemo_cortical_metrics.inputs.freesurfer_output_dirs = freesurfer_output_dirs
            nemo_cortical_metrics.inputs.output_csv_dir = os.path.join(nemo_output_dir, 'postprocess', 'weighted_cortical_metrics')
        
        # If need csv chacovol
        if self.results_to_csv:
            nemo_chacovol = Node(NemoChacovol(), name='nemo_chacovol')
            nemo_postprocess_wf.connect([
                (input_node, nemo_chacovol, [('nemo_output_dir', 'nemo_output_dir'),
                                              ('nemo_postprocessed_dir', 'nemo_postprocessed_dir')])
            ])
            nemo_chacovol.inputs.bids_prefix = bids_prefix

            nemo_chacoconnsc = Node(NemoChacoconnSC(), name='nemo_chacoconnsc')
            nemo_postprocess_wf.connect([
                (input_node, nemo_chacoconnsc, [('nemo_output_dir', 'nemo_output_dir'),
                                               ('nemo_postprocessed_dir', 'nemo_postprocessed_dir')])
            ])
            nemo_chacoconnsc.inputs.bids_prefix = bids_prefix

            nemo_chacoconn = Node(NemoChacoconn(), name='nemo_chacoconn')
            nemo_postprocess_wf.connect([
                (input_node, nemo_chacoconn, [('nemo_output_dir', 'nemo_output_dir'),
                                               ('nemo_postprocessed_dir', 'nemo_postprocessed_dir')])
            ])
            nemo_chacoconn.inputs.bids_prefix = bids_prefix

        return nemo_postprocess_wf
    
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)
        nemo_postprocessed_dir = self.extract_from

        if not os.path.exists(nemo_postprocessed_dir):
            raise FileNotFoundError(f"Nemo postprocessed directory {nemo_postprocessed_dir} does not exist.")

        print(f"Reading .csv results from {nemo_postprocessed_dir}...")

        merged_dfs = {}       # {chacovol_id: list of DataFrames} for weighted cortical metrics
        chacovol_merged = {}  # {atlas: list of DataFrames} for chacovol summaries

        # Patterns
        weighted_pat = re.compile(r"(?P<id>.+)_desc-NemoChacovolWeighted_stat-mean_metrics\.csv$")
        weighted_legacy_pat = re.compile(r"(?P<id>nemo_output_.*)_cortical_metrics\.csv$")
        chacovol_pat = re.compile(r".+_atlas-(?P<atlas>[^_]+)_model-[^_]+_desc-NemoChacovol_stat-mean_metrics\.csv$")
        chacovol_legacy_pat = re.compile(r".*chacovol_(?P<atlas>.+?)_mean\.csv$")

        for subject_folder in os.listdir(nemo_postprocessed_dir):
            if not subject_folder.startswith("sub-"):
                continue

            subject_id = subject_folder.split("-", 1)[1]
            subject_folder_path = os.path.join(nemo_postprocessed_dir, subject_folder)

            if not os.path.isdir(subject_folder_path):
                continue

            for session_folder in os.listdir(subject_folder_path):
                if not session_folder.startswith("ses-"):
                    continue

                session_id = session_folder.split("-", 1)[1]
                nemo_id = f"sub-{subject_id}_ses-{session_id}"
                subject_label = f"sub-{subject_id}"
                session_label = f"ses-{session_id}"
                session_path = os.path.join(subject_folder_path, session_folder)

                if not os.path.isdir(session_path):
                    continue

                # ------------------------------------------------------------
                # Part A: weighted cortical metrics (existing logic)
                # ------------------------------------------------------------
                nemo_csv_dir = os.path.join(session_path, "postprocess", "weighted_cortical_metrics")
                if os.path.exists(nemo_csv_dir):
                    for file in os.listdir(nemo_csv_dir):
                        if not file.endswith(".csv"):
                            continue

                        m = weighted_pat.search(file) or weighted_legacy_pat.search(file)
                        if not m:
                            continue

                        chacovol_id = m.group("id")
                        csv_path = os.path.join(nemo_csv_dir, file)

                        try:
                            df = pd.read_csv(csv_path)
                        except Exception as e:
                            print(f"Failed to read {csv_path}: {e}")
                            continue

                        df.insert(0, "Subject", subject_label)
                        df.insert(1, "Session", session_label)
                        df.insert(2, "nemo_id", nemo_id)

                        merged_dfs.setdefault(chacovol_id, []).append(df)

                # ------------------------------------------------------------
                # Part B: chacovol_csv summaries (NEW)
                # ------------------------------------------------------------
                chacovol_csv_dir = os.path.join(session_path, "postprocess", "chacovol_csv")
                if os.path.exists(chacovol_csv_dir):
                    for file in os.listdir(chacovol_csv_dir):
                        if not file.endswith(".csv"):
                            continue

                        m = chacovol_pat.search(file) or chacovol_legacy_pat.search(file)
                        if not m:
                            continue

                        atlas = m.group("atlas")
                        csv_path = os.path.join(chacovol_csv_dir, file)

                        try:
                            df = pd.read_csv(csv_path)
                        except Exception as e:
                            print(f"Failed to read {csv_path}: {e}")
                            continue

                        # Clean typical format: first column "Unnamed: 0" with value "mean"
                        if df.shape[1] > 0 and str(df.columns[0]).startswith("Unnamed"):
                            df = df.drop(columns=[df.columns[0]])

                        # If multiple rows exist, try to keep the "mean" row if present
                        if df.shape[0] > 1:
                            first_col = df.columns[0] if df.shape[1] > 0 else None
                            if first_col is not None and df[first_col].astype(str).str.lower().eq("mean").any():
                                df = df[df[first_col].astype(str).str.lower().eq("mean")].copy()
                            else:
                                df = df.iloc[[0]].copy()

                        df.insert(0, "Subject", subject_label)
                        df.insert(1, "Session", session_label)

                        chacovol_merged.setdefault(atlas, []).append(df)

        # ------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------
        output_dir = self.output_path
        os.makedirs(output_dir, exist_ok=True)

        # 1) Weighted cortical metrics summary (existing output)
        summary_rows = []
        for chacovol_id, df_list in merged_dfs.items():
            for df in df_list:
                dfx = df.copy()
                dfx.insert(3, "chacovol_id", chacovol_id)
                summary_rows.append(dfx)

        if len(summary_rows) == 0:
            print("Warning: No weighted cortical metrics found to summarize.")
        else:
            summary_df = pd.concat(summary_rows, axis=0, ignore_index=True)
            summary_path = os.path.join(output_dir, "nemo_weighted_cortical_metrics_summary.xlsx")
            summary_df.to_excel(summary_path, index=False)
            print(f"Saved weighted cortical metrics summary to: {summary_path}")

        # 2) Chacovol summaries per atlas (NEW outputs)
        if len(chacovol_merged) == 0:
            print("Warning: No chacovol_csv results found to summarize.")
        else:
            for atlas, df_list in chacovol_merged.items():
                out_df = pd.concat(df_list, axis=0, ignore_index=True)

                # Make filename safe (optional, but helps if atlas has spaces)
                safe_atlas = re.sub(r"[^A-Za-z0-9_.-]+", "_", atlas)

                out_path = os.path.join(output_dir, f"nemo_{safe_atlas}_chacovol_summary.xlsx")
                out_df.to_excel(out_path, index=False)
                print(f"Saved chacovol summary for atlas={atlas} to: {out_path}")
