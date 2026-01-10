import os
import shutil
import subprocess
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import nibabel as nib
import numpy as np
from nipype.interfaces.freesurfer import ReconAll
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Function
from .freesurfer.recon_all_clinical import ReconAllClinical, CopySynthSR, PostProcess
from .freesurfer.synthSR import SynthSR
from cvdproc.pipelines.smri.freesurfer.subfieldseg import SegmentSubregions, SegmentHACross, SegmentBS, SegmentThalamic, HypothalamicSubunits
from cvdproc.pipelines.smri.freesurfer.post_freesurfer import FSQC, Stats2CSV

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class FreesurferPipeline:
    def __init__(
        self,
        subject: object,
        session: object,
        output_path: str,
        use_which_t1w: str = "",
        recon_all: bool = True,
        subregion_ha: bool = False,
        subregion_thalamus: bool = False,
        subregion_brainstem: bool = False,
        subregion_hypothalamus: bool = False,
        fsqc: bool = False,
        stats2csv: bool = False,
        extract_from: str = "",
        **kwargs,
    ):
        """
        Freesurfer pipeline

        Args:
            subject (object): Subject object
            session (object): Session object
            output_path (str): Output path
            use_which_t1w (str, optional): Use specific T1w file if multiple are available. Defaults to "".
            recon_all (bool, optional): Whether to run recon-all. Defaults to True.
            subregion_ha (bool, optional): Whether to segment hippocampus and amygdala subregions. Defaults to False.
            subregion_thalamus (bool, optional): Whether to segment thalamus subregions. Defaults to False.
            subregion_brainstem (bool, optional): Whether to segment brainstem subregions. Defaults to False.
            subregion_hypothalamus (bool, optional): Whether to segment hypothalamus subunits. Defaults to False.
            fsqc (bool, optional): Whether to run FSQC. Defaults to False.
            stats2csv (bool, optional): Whether to convert stats to CSV. Defaults to False.
            extract_from (str, optional): Path to extract results from. Defaults to "".
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = use_which_t1w
        self.recon_all = recon_all

        self.subregion_ha = subregion_ha
        self.subregion_thalamus = subregion_thalamus
        self.subregion_brainstem = subregion_brainstem
        self.subregion_hypothalamus = subregion_hypothalamus

        self.fsqc = fsqc
        self.stats2csv = stats2csv

        self.extract_from = extract_from

    def check_data_requirements(self):
        return self.session.get_t1w_files() is not None

    def create_workflow(self):
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise FileNotFoundError(
                    f"No specific T1w file found for {self.use_which_t1w} or more than one found."
                )
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]

        print(f"[Freesurfer Pipeline] Using T1w file: {t1w_file}")

        fs_output_path = os.path.dirname(self.output_path)
        fs_output_id = os.path.basename(self.output_path)
        os.makedirs(fs_output_path, exist_ok=True)

        fs_workflow = Workflow(name="fs_workflow")

        inputnode = Node(
            IdentityInterface(fields=["t1w_file", "fs_output_id", "subjects_dir"]),
            name="inputnode",
        )

        inputnode.inputs.t1w_file = t1w_file
        inputnode.inputs.fs_output_id = fs_output_id
        inputnode.inputs.subjects_dir = fs_output_path

        if self.recon_all:
            reconall_node = Node(ReconAll(), name="reconall")
            reconall_node.inputs.directive = "all"
            reconall_node.inputs.flags = "-qcache -no-isrunning"
            fs_workflow.connect(inputnode, "t1w_file", reconall_node, "T1_files")
            fs_workflow.connect(inputnode, "fs_output_id", reconall_node, "subject_id")
            fs_workflow.connect(inputnode, "subjects_dir", reconall_node, "subjects_dir")
        else:
            print("[Freesurfer Pipeline] Skipping recon-all step, assuming it has been run already.")
            reconall_node = Node(IdentityInterface(fields=["subject_id", "subjects_dir"]), name="reconall")
            fs_workflow.connect(inputnode, "fs_output_id", reconall_node, "subject_id")
            fs_workflow.connect(inputnode, "subjects_dir", reconall_node, "subjects_dir")

        if self.subregion_ha:
            segment_ha_node = Node(SegmentHACross(), name="segment_ha")
            fs_workflow.connect(reconall_node, "subject_id", segment_ha_node, "subject_id")
            fs_workflow.connect(reconall_node, "subjects_dir", segment_ha_node, "subjects_dir")

        if self.subregion_thalamus:
            segment_thalamus_node = Node(SegmentThalamic(), name="segment_thalamus")
            fs_workflow.connect(reconall_node, "subject_id", segment_thalamus_node, "subject_id")
            fs_workflow.connect(reconall_node, "subjects_dir", segment_thalamus_node, "subjects_dir")

        if self.subregion_brainstem:
            segment_brainstem_node = Node(SegmentBS(), name="segment_brainstem")
            fs_workflow.connect(reconall_node, "subject_id", segment_brainstem_node, "subject_id")
            fs_workflow.connect(reconall_node, "subjects_dir", segment_brainstem_node, "subjects_dir")

        if self.subregion_hypothalamus:
            segment_hypothalamus_node = Node(HypothalamicSubunits(), name="segment_hypothalamus")
            fs_workflow.connect(reconall_node, "subject_id", segment_hypothalamus_node, "s")
            fs_workflow.connect(reconall_node, "subjects_dir", segment_hypothalamus_node, "sd")

        if self.fsqc:
            fsqc_output_dir = os.path.join(
                self.subject.bids_dir,
                "derivatives",
                "fsqc",
                f"sub-{self.subject.subject_id}",
                f"ses-{self.session.session_id}",
            )
            fsqc_node = Node(FSQC(), name="fsqc")
            fs_workflow.connect(reconall_node, "subject_id", fsqc_node, "subject_id")
            fs_workflow.connect(reconall_node, "subjects_dir", fsqc_node, "subjects_dir")
            fsqc_node.inputs.fsqc_output_dir = fsqc_output_dir

        if self.stats2csv:
            stats_dir = os.path.join(self.output_path, "stats")
            stats2csv_node = Node(Stats2CSV(), name="stats2csv")
            stats2csv_node.inputs.output_dir = stats_dir
            fs_workflow.connect(reconall_node, "subject_id", stats2csv_node, "subject_id")
            fs_workflow.connect(reconall_node, "subjects_dir", stats2csv_node, "subjects_dir")

        return fs_workflow

    # ---------------------------------------------------------------------
    # Helpers for extract_results (optimized, wide format, single underscore)
    # ---------------------------------------------------------------------
    @staticmethod
    def _safe_name(x: str) -> str:
        x = str(x).strip()
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", x)
        return x

    @staticmethod
    def _infer_hemi_from_filename(fname: str) -> Optional[str]:
        f = fname.lower()
        if f.startswith("lh.") or ".lh." in f:
            return "lh"
        if f.startswith("rh.") or ".rh." in f:
            return "rh"
        return None

    @staticmethod
    def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    @staticmethod
    def _infer_struct_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["StructName", "Structure", "LabelName", "Name", "ROI", "Region"]:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _infer_volume_col(df: pd.DataFrame) -> Optional[str]:
        candidates = ["Volume_mm3", "volume_mm3", "Volume", "volume"]
        for c in candidates:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                return c
        for c in df.columns:
            if "vol" in str(c).lower() and pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    @staticmethod
    def _sort_cortical_feature_keys_metric_major(keys: List[str]) -> List[str]:
        """
        Metric-major ordering for cortical features:
          <prefix>_<ROI>_<metric>
        Sort by: metric, then ROI, then prefix (stable tie-breaker).
        Prefix may contain underscores, so parse using rsplit("_", 2).
        """
        def _parse(k: str):
            parts = str(k).rsplit("_", 2)
            if len(parts) < 3:
                return ("", "", k)
            prefix, roi, metric = parts[0], parts[1], parts[2]
            return (metric, roi, prefix)

        return sorted(keys, key=_parse)

    def _wide_from_roi_table(
        self,
        df: pd.DataFrame,
        prefix: str,
        struct_col: Optional[str],
        drop_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Convert an ROI table to a single-row wide dict.
        Columns: <prefix>_<ROI>_<metric>
        Keeps all numeric metrics.
        """
        if df is None or df.empty:
            return {}

        df = df.copy()

        if drop_cols:
            for c in drop_cols:
                if c in df.columns:
                    df = df.drop(columns=[c])

        if struct_col is None:
            struct_col = self._infer_struct_col(df)
        if struct_col is None or struct_col not in df.columns:
            return {}

        df[struct_col] = df[struct_col].astype(str).map(self._safe_name)

        exclude = {struct_col, "Index", "SegId"}
        value_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        out: Dict[str, float] = {}
        for _, r in df.iterrows():
            roi = r[struct_col]
            for m in value_cols:
                key = f"{prefix}_{roi}_{self._safe_name(m)}"
                out[key] = r[m]

        return out

    def _wide_volume_only(
        self,
        df: pd.DataFrame,
        prefix: str,
        fname: str,
    ) -> Dict[str, float]:
        """
        Convert aseg / wmparc / subseg tables to a single-row wide dict using only volume.
        Columns: <prefix>_<StructName>_Volume_mm3
        """
        if df is None or df.empty:
            return {}

        df = df.copy()

        for c in ["Index", "SegId"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        struct_col = self._infer_struct_col(df)
        vol_col = self._infer_volume_col(df)
        if struct_col is None or vol_col is None:
            return {}

        df[struct_col] = df[struct_col].astype(str).map(self._safe_name)
        vol = pd.to_numeric(df[vol_col], errors="coerce")

        out: Dict[str, float] = {}
        for name, v in zip(df[struct_col], vol):
            if pd.isna(v):
                continue
            out[f"{prefix}_{name}_Volume_mm3"] = float(v)

        hemi = self._infer_hemi_from_filename(fname)
        if hemi is not None:
            out[f"{prefix}_hemi"] = hemi

        return out

    def _wide_brainvol(self, df: pd.DataFrame, prefix: str) -> Dict[str, float]:
        """
        Convert brainvol.csv (long) to wide dict.
        Columns: <prefix>_<name>
        """
        if df is None or df.empty:
            return {}

        if "name" not in df.columns or "value" not in df.columns:
            return {}

        df = df.copy()
        df["name"] = df["name"].astype(str).map(self._safe_name)
        vals = pd.to_numeric(df["value"], errors="coerce")

        out: Dict[str, float] = {}
        for n, v in zip(df["name"], vals):
            if pd.isna(v):
                continue
            out[f"{prefix}_{n}"] = float(v)

        return out

    # ---------------------------------------------------------------------
    # Extract results (streaming CSV writer; no fragmentation warning)
    # Cortical columns: metric-major ordering
    # ---------------------------------------------------------------------
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)
        out_dir = Path(self.output_path)

        freesurfer_output_dir = self.extract_from
        if not freesurfer_output_dir or not os.path.isdir(freesurfer_output_dir):
            raise FileNotFoundError(f"extract_from is not a valid directory: {freesurfer_output_dir}")

        cortical_files = [
            "lh.aparc.csv",
            "rh.aparc.csv",
            "lh.aparc.a2009s.csv",
            "rh.aparc.a2009s.csv",
            "lh.aparc.DKTatlas.csv",
            "rh.aparc.DKTatlas.csv",
            "lh.aparc.pial.csv",
            "rh.aparc.pial.csv",
            "lh.BA_exvivo.csv",
            "rh.BA_exvivo.csv",
            "lh.BA_exvivo.thresh.csv",
            "rh.BA_exvivo.thresh.csv",
            "lh.w-g.pct.csv",
            "rh.w-g.pct.csv",
        ]

        volume_only_files = [
            "aseg.csv",
            "wmparc.csv",
            "amygdalar-nuclei.lh.T1.v22.csv",
            "amygdalar-nuclei.rh.T1.v22.csv",
            "brainstem.v13.csv",
            "hipposubfields.lh.T1.v22.csv",
            "hipposubfields.rh.T1.v22.csv",
            "hypothalamic_subunits_volumes.v1.csv",
            "thalamic-nuclei.lh.v13.T1.csv",
            "thalamic-nuclei.rh.v13.T1.csv",
        ]

        brainvol_file = "brainvol.csv"

        # filekey -> {"fh": file handle, "writer": DictWriter, "fieldnames": list[str], "is_cortical": bool}
        writers: Dict[str, Dict[str, object]] = {}
        # filekey -> set of (subject, session) to ensure one row per subject-session
        seen: Dict[str, set] = {}

        def _open_writer(filekey: str, out_path: Path, fieldnames: List[str], is_cortical: bool) -> csv.DictWriter:
            out_fh = open(out_path, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            writers[filekey] = {
                "fh": out_fh,
                "writer": w,
                "fieldnames": fieldnames,
                "is_cortical": is_cortical,
                "path": out_path,
            }
            seen[filekey] = set()
            return w

        def _close_all():
            for v in writers.values():
                try:
                    v["fh"].close()
                except Exception:
                    pass

        def _rewrite_with_expanded_header(filekey: str, new_fieldnames: List[str]):
            info = writers[filekey]
            out_path: Path = info["path"]

            # Read existing rows
            old_rows: List[Dict[str, object]] = []
            try:
                if out_path.exists():
                    old_df = pd.read_csv(out_path)
                    old_rows = old_df.to_dict(orient="records")
            except Exception:
                old_rows = []

            # Close old handle
            try:
                info["fh"].close()
            except Exception:
                pass

            # Rewrite
            out_fh = open(out_path, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(out_fh, fieldnames=new_fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in old_rows:
                w.writerow(r)

            writers[filekey] = {
                "fh": out_fh,
                "writer": w,
                "fieldnames": new_fieldnames,
                "is_cortical": info["is_cortical"],
                "path": out_path,
            }

        def _ensure_writer_and_write_row(
            filekey: str,
            out_path: Path,
            row: Dict[str, object],
            is_cortical: bool,
        ):
            key_pair = (row.get("subject"), row.get("session"))
            if key_pair[0] is None or key_pair[1] is None:
                return

            if filekey in seen and key_pair in seen[filekey]:
                return

            # Determine desired feature ordering
            feat_keys = [k for k in row.keys() if k not in ("subject", "session")]

            if is_cortical:
                feat_keys = self._sort_cortical_feature_keys_metric_major(feat_keys)
            else:
                feat_keys = sorted(feat_keys)

            fieldnames = ["subject", "session"] + feat_keys

            if filekey not in writers:
                _open_writer(filekey, out_path, fieldnames, is_cortical)
            else:
                existing = writers[filekey]["fieldnames"]
                existing_feats = existing[2:]

                # Expand schema if new columns appear
                union_feats = set(existing_feats).union(set(feat_keys))
                if is_cortical:
                    new_feats = self._sort_cortical_feature_keys_metric_major(list(union_feats))
                else:
                    new_feats = sorted(list(union_feats))

                new_fieldnames = ["subject", "session"] + new_feats
                if new_fieldnames != existing:
                    _rewrite_with_expanded_header(filekey, new_fieldnames)

            writers[filekey]["writer"].writerow(row)
            seen[filekey].add(key_pair)

        try:
            for subject_folder in os.listdir(freesurfer_output_dir):
                subject_path = os.path.join(freesurfer_output_dir, subject_folder)
                if not os.path.isdir(subject_path):
                    continue

                for session_folder in os.listdir(subject_path):
                    session_path = os.path.join(subject_path, session_folder)
                    if not os.path.isdir(session_path):
                        continue

                    subject = str(subject_folder)
                    session = str(session_folder)
                    stats_dir = os.path.join(session_path, "stats")
                    if not os.path.isdir(stats_dir):
                        continue

                    # ---- cortical (metric-major columns) ----
                    for fname in cortical_files:
                        fpath = os.path.join(stats_dir, fname)
                        df = self._read_csv_if_exists(fpath)
                        if df is None:
                            continue

                        prefix = self._safe_name(Path(fname).stem)
                        row: Dict[str, object] = {"subject": subject, "session": session}
                        row.update(
                            self._wide_from_roi_table(
                                df=df,
                                prefix=prefix,
                                struct_col=self._infer_struct_col(df),
                                drop_cols=["Index", "SegId"],
                            )
                        )
                        if len(row) <= 2:
                            continue

                        out_path = out_dir / f"{Path(fname).stem}_summary.csv"
                        _ensure_writer_and_write_row(
                            filekey=fname,
                            out_path=out_path,
                            row=row,
                            is_cortical=True,
                        )

                    # ---- volume-only ----
                    for fname in volume_only_files:
                        fpath = os.path.join(stats_dir, fname)
                        df = self._read_csv_if_exists(fpath)
                        if df is None:
                            continue

                        prefix = self._safe_name(Path(fname).stem)
                        row = {"subject": subject, "session": session}
                        row.update(self._wide_volume_only(df=df, prefix=prefix, fname=fname))
                        if len(row) <= 2:
                            continue

                        out_path = out_dir / f"{Path(fname).stem}_summary.csv"
                        _ensure_writer_and_write_row(
                            filekey=fname,
                            out_path=out_path,
                            row=row,
                            is_cortical=False,
                        )

                    # ---- brainvol ----
                    fpath = os.path.join(stats_dir, brainvol_file)
                    df = self._read_csv_if_exists(fpath)
                    if df is not None:
                        row = {"subject": subject, "session": session}
                        row.update(self._wide_brainvol(df=df, prefix="brainvol"))
                        if len(row) <= 2:
                            continue

                        out_path = out_dir / "brainvol_summary.csv"
                        _ensure_writer_and_write_row(
                            filekey="brainvol",
                            out_path=out_path,
                            row=row,
                            is_cortical=False,
                        )

        finally:
            _close_all()

class FreesurferClinicalPipeline:
    def __init__(self, 
                 subject: object, 
                 session: object, 
                 output_path: str, 
                 use_which_t1w: str = '',
                 **kwargs):
        """
        Freesurfer clinical pipeline

        Args:
            subject: The subject object.
            session: The session object.
            output_path: The output path for the pipeline.
            use_which_t1w (str, optional): Use specific T1w file if multiple are available. Defaults to "".
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = use_which_t1w

    def check_data_requirements(self):
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

        skip_recon = False
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
            copy_synthsr_node.inputs.out_file = os.path.join(self.subject.bids_dir, f"sub-{self.subject.subject_id}", f"sub-{self.session.session_id}", 'anat', 
                                                             rename_bids_file(t1w_file, {"acq": "SynthSR"}, 'T1w', '.nii.gz'))

            postprocess_node = Node(PostProcess(), name="postprocess")
            fs_clinical_workflow.connect(recon_all_clinical_node, "output_dir", postprocess_node, "fs_output_dir")

            outputnode = Node(IdentityInterface(fields=["output_dir", "synthsr_raw"]), name="outputnode")
            fs_clinical_workflow.connect(postprocess_node, "fs_output_dir", outputnode, "output_dir")
            fs_clinical_workflow.connect(copy_synthsr_node, "out_file", outputnode, "synthsr_raw")
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
    def __init__(self, 
                 subject: object, 
                 session: object, 
                 output_path: str, 
                 use_which_t1w: str = '',
                 **kwargs):
        """
        SynthSR pipeline

        Args:
            subject: The subject object containing BIDS information.
            session: The session object containing BIDS information.
            output_path: The output path for the pipeline results.
            use_which_t1w (str, optional): Use specific T1w file if multiple are available. Defaults to "".
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = use_which_t1w

    def check_data_requirements(self):
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