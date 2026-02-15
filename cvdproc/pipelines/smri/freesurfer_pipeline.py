import os
import shutil
import subprocess
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import nibabel as nib
import numpy as np
from nipype.interfaces.freesurfer import ReconAll
from nipype import Node, Workflow, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from .freesurfer.recon_all_clinical import ReconAllClinical, CopySynthSR, PostProcess
from .freesurfer.synthSR import SynthSR
from cvdproc.pipelines.smri.freesurfer.subfieldseg import SegmentSubregions, SegmentHACross, SegmentBS, SegmentThalamic, HypothalamicSubunits
from cvdproc.pipelines.smri.freesurfer.post_freesurfer import FSQC, Stats2CSV
from cvdproc.pipelines.smri.freesurfer.recon_all_longitudinal import FreesurferLongitudinal
from cvdproc.pipelines.smri.freesurfer.results_extractor import FreesurferStatsExtractorMixin

from cvdproc.bids_data.rename_bids_file import rename_bids_file

def build_long_stats2csv_inputs(long_subject_dirs):
    import os

    subject_ids = []
    output_dirs = []

    if long_subject_dirs is None:
        return subject_ids, output_dirs

    for d in long_subject_dirs:
        d = str(d)
        sid = os.path.basename(d)   # ses-XXX.long.sub-YYY
        subject_ids.append(sid)
        output_dirs.append(os.path.join(d, "stats"))

    return subject_ids, output_dirs

class FreesurferPipeline(FreesurferStatsExtractorMixin):
    def __init__(
        self,
        subject: object,
        session: object,
        output_path: str,
        use_which_t1w: str = "",
        recon_all: bool = True,
        longitudinal: bool = False,
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
    # Extract results (streaming CSV writer; no fragmentation warning)
    # Cortical columns: metric-major ordering
    # ---------------------------------------------------------------------
    def extract_results(self):
        import os
        import re

        if not self.extract_from or not os.path.isdir(self.extract_from):
            raise FileNotFoundError(f"extract_from is not a valid directory: {self.extract_from}")

        fs_root = self.extract_from  # should be derivatives/freesurfer

        ses_dir_pat = re.compile(r"^ses-[^/]+$")

        items = []
        for sub in sorted(os.listdir(fs_root)):
            sub_path = os.path.join(fs_root, sub)
            if not os.path.isdir(sub_path):
                continue
            if not sub.startswith("sub-"):
                continue

            for ses_folder in sorted(os.listdir(sub_path)):
                ses_path = os.path.join(sub_path, ses_folder)
                if not os.path.isdir(ses_path):
                    continue
                if not ses_dir_pat.match(ses_folder):
                    continue

                # Exclude any unexpected longitudinal-like folders if they appear
                if ".long." in ses_folder:
                    continue

                session = ses_folder  # e.g., ses-baseline, ses-F1
                stats_dir = os.path.join(ses_path, "stats")
                if not os.path.isdir(stats_dir):
                    continue

                items.append(
                    {"subject": sub, "session": session, "stats_dir": stats_dir}
                )

        # Write merged summary CSVs into self.output_path (no extra subfolder)
        self._extract_merge_stats_dirs(stats_dir_items=items, output_path=self.output_path)

class FreesurferLongitudinalPipeline(FreesurferStatsExtractorMixin):
    def __init__(
        self,
        subject: object,
        output_path: str,
        subregion_ha: bool = False,
        subregion_thalamus: bool = False,
        subregion_brainstem: bool = False,
        subregion_hypothalamus: bool = False,
        stats2csv: bool = False,
        extract_from: str = "",
        **kwargs,
    ):
        self.subject = subject
        self.output_path = os.path.abspath(output_path)

        self.subregion_ha = subregion_ha
        self.subregion_thalamus = subregion_thalamus
        self.subregion_brainstem = subregion_brainstem
        self.subregion_hypothalamus = subregion_hypothalamus

        self.stats2csv = stats2csv
        self.extract_from = extract_from

    def check_data_requirements(self):
        return True

    def create_workflow(self):
        fs_longitudinal_wf = Workflow(name="fs_longitudinal_workflow")

        inputnode = Node(
            IdentityInterface(
                fields=[
                    "bids_dir",
                    "subject_id",
                    "subregion_ha",
                    "subregion_thalamus",
                    "subregion_brainstem",
                    "subregion_hypothalamus",
                ]
            ),
            name="inputnode",
        )
        inputnode.inputs.bids_dir = self.subject.bids_dir
        inputnode.inputs.subject_id = self.subject.subject_id
        inputnode.inputs.subregion_ha = self.subregion_ha
        inputnode.inputs.subregion_thalamus = self.subregion_thalamus
        inputnode.inputs.subregion_brainstem = self.subregion_brainstem
        inputnode.inputs.subregion_hypothalamus = self.subregion_hypothalamus

        recon_all_longitudinal_node = Node(FreesurferLongitudinal(), name="recon_all_longitudinal")
        fs_longitudinal_wf.connect(inputnode, "bids_dir", recon_all_longitudinal_node, "bids_dir")
        fs_longitudinal_wf.connect(inputnode, "subject_id", recon_all_longitudinal_node, "subject_id")
        fs_longitudinal_wf.connect(inputnode, "subregion_ha", recon_all_longitudinal_node, "subregion_ha")
        fs_longitudinal_wf.connect(inputnode, "subregion_thalamus", recon_all_longitudinal_node, "subregion_thalamus")
        fs_longitudinal_wf.connect(inputnode, "subregion_brainstem", recon_all_longitudinal_node, "subregion_brainstem")
        fs_longitudinal_wf.connect(inputnode, "subregion_hypothalamus", recon_all_longitudinal_node, "subregion_hypothalamus")

        if self.stats2csv:
            build_inputs_node = Node(
                Function(
                    input_names=["long_subject_dirs"],
                    output_names=["subject_ids", "output_dirs"],
                    function=build_long_stats2csv_inputs,
                ),
                name="build_long_stats2csv_inputs",
            )

            fs_longitudinal_wf.connect(
                recon_all_longitudinal_node,
                "long_subject_dirs",
                build_inputs_node,
                "long_subject_dirs",
            )

            stats2csv_map = MapNode(
                Stats2CSV(),
                iterfield=["subject_id", "output_dir"],
                name="stats2csv_long_map",
            )

            fs_longitudinal_wf.connect(recon_all_longitudinal_node, "subjects_dir", stats2csv_map, "subjects_dir")
            fs_longitudinal_wf.connect(build_inputs_node, "subject_ids", stats2csv_map, "subject_id")
            fs_longitudinal_wf.connect(build_inputs_node, "output_dirs", stats2csv_map, "output_dir")

        return fs_longitudinal_wf
    
    def extract_results(self):
        import os
        import re

        if not self.extract_from or not os.path.isdir(self.extract_from):
            raise FileNotFoundError(f"extract_from is not a valid directory: {self.extract_from}")

        fs_root = self.extract_from  # should be derivatives/freesurfer

        long_dir_pat = re.compile(r"^ses-[^.]+\.long\.sub-.+$")

        items = []
        for sub in sorted(os.listdir(fs_root)):
            sub_path = os.path.join(fs_root, sub)
            if not os.path.isdir(sub_path):
                continue
            if not sub.startswith("sub-"):
                continue

            for long_folder in sorted(os.listdir(sub_path)):
                long_path = os.path.join(sub_path, long_folder)
                if not os.path.isdir(long_path):
                    continue
                if not long_dir_pat.match(long_folder):
                    continue

                session = long_folder.split(".long.", 1)[0]  # ses-baseline
                stats_dir = os.path.join(long_path, "stats")
                if not os.path.isdir(stats_dir):
                    continue

                items.append(
                    {"subject": sub, "session": session, "stats_dir": stats_dir}
                )

        # Write merged summary CSVs into self.output_path (no extra subfolder)
        self._extract_merge_stats_dirs(stats_dir_items=items, output_path=self.output_path)

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