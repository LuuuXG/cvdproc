import os
import csv
import json
import glob
import numpy as np
import pandas as pd

from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.fsl import FLIRT
from nipype.interfaces.freesurfer import MRIConvert

from cvdproc.bids_data.rename_bids_file import rename_bids_file
from cvdproc.config.paths import get_package_path

from cvdproc.pipelines.perfusion.exploreasl.exploreasl_nipype import (
    ExploreASLCustom,
    ASLtoT1Register,
)
from cvdproc.pipelines.common.register import (
    MRIConvertApplyWarp,
    SynthmorphNonlinear,
    Tkregister2fs2t1w,
)
from cvdproc.pipelines.common.image_calc import CalculateScalarMaps


class ASLPipeline:
    def __init__(
        self,
        subject: object,
        session: object,
        output_path: str,
        use_which_asl: str = None,
        use_which_t1w: str = None,
        preprocess_method: str = "ExploreASL",
        ignore_m0: bool = False,
        calculate_asl_metrics: bool = False,
        skip_preprocess: bool = False,
        extract_from: str = None,
    ):
        """
        Arterial Spin Labeling (ASL) Pipeline

        Processes ASL data using ExploreASL for CBF quantification. Supports
        both single-PLD and multi-PLD data. If a T1w-to-MNI warp is not already
        available, SynthMorph registration is performed. CBF maps are registered
        to T1w and MNI space. ATT maps are generated for multi-PLD data.

        Args:
            subject: BIDSSubject object
            session: BIDSSession object
            output_path: output directory for the pipeline
            use_which_asl: specific string to select ASL image. If None, use the first ASL image found.
            use_which_t1w: specific string to select T1w image. If None, use the first T1w image found.
            preprocess_method: preprocessing method. Currently only 'ExploreASL' is supported (default).
            ignore_m0: if True, skip separate M0 files even when found (M0 scan is not used).
                The M0 nifti/json files are not copied to rawdata, and the ASL JSON
                M0Type is changed from "Separate" to "Absent". Default: False.
            calculate_asl_metrics: if True, compute ROI statistics for CBF (and ATT if multi-PLD)
                in SynthSeg, FS aparc+aseg, ArterialAtlas, and Schirmer VT masks. Default: False.
            skip_preprocess: if True, skip ExploreASL processing and registration. Assumes
                CBF (and optionally ATT) maps already exist in T1w and MNI space under
                output_path. Only ROI extraction is performed. Default: False.
            extract_from: if set, extract_results reads from this path instead of walking
                the subject's own output directory (used for population-level extraction). Default: None.
        """
        self.subject = subject
        self.session = session
        self.output_path = output_path
        self.use_which_asl = use_which_asl
        self.use_which_t1w = use_which_t1w
        self.preprocess_method = preprocess_method
        self.ignore_m0 = ignore_m0
        self.calculate_asl_metrics = calculate_asl_metrics
        self.skip_preprocess = skip_preprocess
        self.extract_from = extract_from

    def check_data_requirements(self):
        return (
            self.session.get_perf_files() is not None
            and self.session.get_t1w_files() is not None
        )

    def _check_has_att(self, asl_file: str) -> bool:
        """
        Determine whether ATT output should be expected.

        Rule:
        - Multi-PLD ASL: PostLabelingDelay is a list with length > 1 -> ATT branch enabled
        - Single-PLD ASL: PostLabelingDelay is a scalar or a list with length == 1 -> no ATT branch
        - Missing/invalid JSON -> assume no ATT branch
        """
        asl_json = asl_file.replace(".nii.gz", ".json").replace(".nii", ".json")

        if not os.path.exists(asl_json):
            print(
                f"[ASL Pipeline] ASL sidecar JSON not found: {asl_json}. "
                "Assume no ATT output."
            )
            return False

        try:
            with open(asl_json, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(
                f"[ASL Pipeline] Failed to read ASL JSON: {asl_json}. "
                f"Assume no ATT output. Error: {e}"
            )
            return False

        pld = meta.get("PostLabelingDelay", None)

        if isinstance(pld, list):
            if len(pld) > 1:
                print(
                    "[ASL Pipeline] Multi-PLD ASL detected from PostLabelingDelay list. "
                    "ATT branch will be created."
                )
                return True

            print(
                "[ASL Pipeline] Single-PLD ASL detected from PostLabelingDelay list. "
                "No ATT branch."
            )
            return False

        if isinstance(pld, (int, float)):
            print(
                "[ASL Pipeline] Single-PLD ASL detected from scalar PostLabelingDelay. "
                "No ATT branch."
            )
            return False

        print(
            "[ASL Pipeline] PostLabelingDelay is missing or unsupported. "
            "Assume no ATT output."
        )
        return False

    def create_workflow(self):
        os.makedirs(self.output_path, exist_ok=True)

        # ===============================
        # Get ASL and T1w files
        # ===============================
        asl_files = self.session.get_perf_files()
        if self.use_which_asl is not None:
            nifti_asl_files = [
                f for f in asl_files if f.endswith(".nii") or f.endswith(".nii.gz")
            ]
            asl_files = [f for f in nifti_asl_files if self.use_which_asl in f]
            if len(asl_files) != 1:
                raise ValueError(
                    f"No specific ASL file found for {self.use_which_asl} "
                    "or more than one found."
                )
            asl_file = asl_files[0]
            print(f"[ASL Pipeline] Using ASL file: {asl_file}")
        else:
            asl_file = asl_files[0]
            print(f"[ASL Pipeline] Using the first available ASL file: {asl_file}")

        t1w_files = self.session.get_t1w_files()
        if self.use_which_t1w is not None:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise ValueError(
                    f"No specific T1w file found for {self.use_which_t1w} "
                    "or more than one found."
                )
            t1w_file = t1w_files[0]
            print(f"[ASL Pipeline] Using T1w file: {t1w_file}")
        else:
            t1w_file = t1w_files[0]
            print(f"[ASL Pipeline] Using the first available T1w file: {t1w_file}")

        # ===============================
        # Determine whether ATT branch is needed
        # ===============================
        has_att = self._check_has_att(asl_file)

        # ===============================
        # Handle M0 file
        # ===============================
        m0_file = None
        m0_index = None
        asl_context_file = (
            asl_file.replace("_asl.nii.gz", "_aslcontext.tsv")
            .replace("_asl.nii", "_aslcontext.tsv")
        )

        # The file should contain one single column.
        # Either have a 'volume_type' header or not.
        # Possible volume types:
        # 'control', 'label', 'm0scan', 'deltam', 'cbf', 'noRF', 'n/a'
        if os.path.exists(asl_context_file):
            with open(asl_context_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0 and lines[0].strip() == "volume_type":
                    volume_types = [line.strip() for line in lines[1:]]
                else:
                    volume_types = [line.strip() for line in lines]

            if "m0scan" in volume_types:
                m0_index = volume_types.index("m0scan")
                print(f"[ASL Pipeline] Found M0 scan in ASL file at index {m0_index}.")
            else:
                possible_m0_file = (
                    asl_file.replace("_asl.nii.gz", "_m0scan.nii.gz")
                    .replace("_asl.nii", "_m0scan.nii")
                )
                if os.path.exists(possible_m0_file):
                    if self.ignore_m0:
                        print(f"[ASL Pipeline] Separate M0 file found but ignore_m0=True, skipping: {possible_m0_file}")
                        m0_file = None
                    else:
                        m0_file = possible_m0_file
                        print(f"[ASL Pipeline] Using separate M0 file: {m0_file}")

        # ===============================
        # Main Workflow
        # ===============================
        asl_wf = Workflow(name="asl_workflow")

        inputnode = Node(IdentityInterface(fields=["asl_file", "t1w_file", "m0_file", "output_path", "fs_subjects_dir", "fs_subject_id"]), name="inputnode")
        inputnode.inputs.asl_file = asl_file if asl_file else None
        inputnode.inputs.t1w_file = t1w_file if t1w_file else None
        inputnode.inputs.m0_file = m0_file if m0_file else None
        inputnode.inputs.output_path = self.output_path
        inputnode.inputs.fs_subjects_dir = os.path.dirname(self.session.freesurfer_dir) if self.session.freesurfer_dir else None
        inputnode.inputs.fs_subject_id = os.path.basename(self.session.freesurfer_dir) if self.session.freesurfer_dir else None

        # ===============================
        # Check T1 <-> MNI non-linear warp
        # ===============================
        if t1w_file != "":
            t1_to_mni_warp_node = Node(IdentityInterface(fields=["warp_image"]), name="t1_to_mni_warp_node")
            mni_to_t1_warp_node = Node(IdentityInterface(fields=["warp_image"]), name="mni_to_t1_warp_node")

            target_warp = os.path.join(
                self.subject.bids_dir,
                "derivatives",
                "xfm",
                f"sub-{self.subject.subject_id}",
                f"ses-{self.session.session_id}",
                f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz",
            )
            target_inverse_warp = os.path.join(
                self.subject.bids_dir,
                "derivatives",
                "xfm",
                f"sub-{self.subject.subject_id}",
                f"ses-{self.session.session_id}",
                f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz",
            )

            if not os.path.exists(target_warp) or not os.path.exists(target_inverse_warp):
                print(
                    f"[ASL Pipeline] No existing T1w to MNI warp file found: {target_warp}. "
                    "Will run Synthmorph registration to get the warp (1mm resolution)."
                )
                print(
                    "[ASL Pipeline] If you want a different resolution, "
                    "please run a separate T1 registration pipeline first."
                )

                t1w_to_mni_registration = Node(SynthmorphNonlinear(), name="t1w_to_mni_registration")
                asl_wf.connect(inputnode, "t1w_file", t1w_to_mni_registration, "t1")
                t1w_to_mni_registration.inputs.mni_template = get_package_path("data", "standard", "MNI152", "MNI152_T1_1mm_brain.nii.gz")
                t1w_to_mni_registration.inputs.t1_mni_out = os.path.join(
                    os.path.dirname(target_warp),
                    rename_bids_file(
                        t1w_file,
                        {"space": "MNI152NLin6ASym", "desc": "brain"},
                        "T1w",
                        ".nii.gz",
                    ),
                )
                t1w_to_mni_registration.inputs.t1_2_mni_warp = target_warp
                t1w_to_mni_registration.inputs.mni_2_t1_warp = target_inverse_warp
                t1w_to_mni_registration.inputs.register_between_stripped = True

                asl_wf.connect(t1w_to_mni_registration, "t1_2_mni_warp", t1_to_mni_warp_node, "warp_image")
                asl_wf.connect(t1w_to_mni_registration, "mni_2_t1_warp", mni_to_t1_warp_node, "warp_image")
            else:
                print(
                    f"[ASL Pipeline] Found existing T1w to MNI warp file: {target_warp}. "
                    "Will use it."
                )
                t1_to_mni_warp_node.inputs.warp_image = target_warp
                mni_to_t1_warp_node.inputs.warp_image = target_inverse_warp

        # ===============================
        # ExploreASL preprocessing (or skip if already done)
        # ===============================
        if not self.skip_preprocess and self.preprocess_method.lower() == "exploreasl":
            t1w_filename_without_ext = (
                os.path.basename(t1w_file)
                .replace(".nii.gz", "")
                .replace(".nii", "")
            )
            asl_filename_without_ext = (
                os.path.basename(asl_file)
                .replace("_asl.nii.gz", "")
                .replace("_asl.nii", "")
            )

            exploreasl_node = Node(ExploreASLCustom(), name="exploreasl")
            exploreasl_node.inputs.bids_root_dir = self.subject.bids_dir
            exploreasl_node.inputs.subject_id = self.subject.subject_id
            exploreasl_node.inputs.session_id = self.session.session_id
            exploreasl_node.inputs.t1w_filter_filename = t1w_filename_without_ext
            exploreasl_node.inputs.asl_filter_filename = asl_filename_without_ext
            exploreasl_node.inputs.script_path = get_package_path("pipelines", "matlab", "exploreasl", "exploreasl_process.m")
            exploreasl_node.inputs.exploreasl_dir = get_package_path("data", "matlab_toolbox", "ExploreASL-develop")
            exploreasl_node.inputs.ignore_m0 = self.ignore_m0
            asl_wf.connect(inputnode, "output_path", exploreasl_node, "output_dir")

            # --------------------------------
            # CBF branch (always run)
            # --------------------------------
            cbf_to_t1w_node = Node(ASLtoT1Register(), name="cbf_to_t1w")
            asl_wf.connect(exploreasl_node, "cbf", cbf_to_t1w_node, "asl_space_img")
            asl_wf.connect(exploreasl_node, "rt1", cbf_to_t1w_node, "asl_space_t1w_img")
            asl_wf.connect(inputnode, "t1w_file", cbf_to_t1w_node, "target_t1w_img")
            cbf_to_t1w_node.inputs.asl_in_t1w_img = os.path.join(
                self.output_path,
                f"sub-{self.subject.subject_id}_{self.session.session_id}_space-T1w_cbf.nii.gz",
            )

            cbf_to_mni_node = Node(MRIConvertApplyWarp(), name="cbf_to_mni")
            asl_wf.connect(cbf_to_t1w_node, "out_file", cbf_to_mni_node, "input_image")
            asl_wf.connect(t1_to_mni_warp_node, "warp_image", cbf_to_mni_node, "warp_image")
            cbf_to_mni_node.inputs.output_image = os.path.join(
                self.output_path,
                f"sub-{self.subject.subject_id}_{self.session.session_id}_space-MNI152NLin6ASym_cbf.nii.gz",
            )

            # --------------------------------
            # ATT branch (optional)
            # --------------------------------
            if has_att:
                att_to_t1w_node = Node(ASLtoT1Register(), name="att_to_t1w")
                asl_wf.connect(exploreasl_node, "att", att_to_t1w_node, "asl_space_img")
                asl_wf.connect(exploreasl_node, "rt1", att_to_t1w_node, "asl_space_t1w_img")
                asl_wf.connect(inputnode, "t1w_file", att_to_t1w_node, "target_t1w_img")
                att_to_t1w_node.inputs.asl_in_t1w_img = os.path.join(
                    self.output_path,
                    f"sub-{self.subject.subject_id}_{self.session.session_id}_space-T1w_att.nii.gz",
                )

                att_to_mni_node = Node(MRIConvertApplyWarp(), name="att_to_mni")
                asl_wf.connect(att_to_t1w_node, "out_file", att_to_mni_node, "input_image")
                asl_wf.connect(t1_to_mni_warp_node, "warp_image", att_to_mni_node, "warp_image")
                att_to_mni_node.inputs.output_image = os.path.join(
                    self.output_path,
                    f"sub-{self.subject.subject_id}_{self.session.session_id}_space-MNI152NLin6ASym_att.nii.gz",
                )

        elif self.skip_preprocess:
            # --- Skip mode: use existing CBF/ATT maps ---
            cbf_t1w_path = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_{self.session.session_id}_space-T1w_cbf.nii.gz")
            if not os.path.exists(cbf_t1w_path):
                raise FileNotFoundError(f"[ASL Pipeline] skip_preprocess=True but CBF T1w map not found: {cbf_t1w_path}")
            cbf_mni_path = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_{self.session.session_id}_space-MNI152NLin6ASym_cbf.nii.gz")
            if not os.path.exists(cbf_mni_path):
                raise FileNotFoundError(f"[ASL Pipeline] skip_preprocess=True but CBF MNI map not found: {cbf_mni_path}")
            print(f"[ASL Pipeline] skip_preprocess=True, using existing CBF maps")

            cbf_to_t1w_node = Node(IdentityInterface(fields=["out_file"]), name="cbf_to_t1w")
            cbf_to_t1w_node.inputs.out_file = cbf_t1w_path
            cbf_to_mni_node = Node(IdentityInterface(fields=["output_image"]), name="cbf_to_mni")
            cbf_to_mni_node.inputs.output_image = cbf_mni_path

            att_t1w_path = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_{self.session.session_id}_space-T1w_att.nii.gz")
            att_mni_path = os.path.join(self.output_path, f"sub-{self.subject.subject_id}_{self.session.session_id}_space-MNI152NLin6ASym_att.nii.gz")
            has_att = os.path.exists(att_t1w_path) and os.path.exists(att_mni_path)
            if has_att:
                print(f"[ASL Pipeline] skip_preprocess=True, using existing ATT maps")
                att_to_t1w_node = Node(IdentityInterface(fields=["out_file"]), name="att_to_t1w")
                att_to_t1w_node.inputs.out_file = att_t1w_path
                att_to_mni_node = Node(IdentityInterface(fields=["output_image"]), name="att_to_mni")
                att_to_mni_node.inputs.output_image = att_mni_path
            else:
                print("[ASL Pipeline] skip_preprocess=True, no ATT maps found (single-PLD).")

        # ===============================
        # ROI metrics extraction
        # ===============================
        if self.calculate_asl_metrics:
            metrics_dir = os.path.join(self.output_path, "asl_metrics_stats")
            os.makedirs(metrics_dir, exist_ok=True)
            xfm_dir = os.path.join(self.subject.bids_dir, "derivatives", "xfm", f"sub-{self.subject.subject_id}", f"ses-{self.session.session_id}")
            os.makedirs(xfm_dir, exist_ok=True)

            def _assemble_scalar_list(cbf, att):
                files = [cbf]
                colnames = ["CBF"]
                if att:
                    files.append(att)
                    colnames.append("ATT")
                return files, colnames

            # --- T1w-space scalar assembly (for SynthSeg, FS aparc+aseg) ---
            assemble_t1w_scalars = Node(
                Function(input_names=["cbf", "att"], output_names=["scalar_files", "scalar_names"], function=_assemble_scalar_list),
                name="assemble_t1w_scalars")
            asl_wf.connect(cbf_to_t1w_node, "out_file", assemble_t1w_scalars, "cbf")
            if has_att:
                asl_wf.connect(att_to_t1w_node, "out_file", assemble_t1w_scalars, "att")
            else:
                assemble_t1w_scalars.inputs.att = None

            # --- MNI-space scalar assembly (for ArterialAtlas, Schirmer VT) ---
            assemble_mni_scalars = Node(
                Function(input_names=["cbf", "att"], output_names=["scalar_files", "scalar_names"], function=_assemble_scalar_list),
                name="assemble_mni_scalars")
            asl_wf.connect(cbf_to_mni_node, "output_image", assemble_mni_scalars, "cbf")
            if has_att:
                asl_wf.connect(att_to_mni_node, "output_image", assemble_mni_scalars, "att")
            else:
                assemble_mni_scalars.inputs.att = None

            # --- 1. SynthSeg mask (T1w space, no warping) ---
            synthseg_files = []
            if self.session.anat_seg_dir is not None:
                synthseg_files = glob.glob(os.path.join(self.session.anat_seg_dir, "synthseg", "*_synthseg.nii.gz"))
            if synthseg_files:
                synthseg_t1w = synthseg_files[0]
                scalar_maps_for_synthseg = Node(CalculateScalarMaps(), name="scalar_maps_for_synthseg")
                asl_wf.connect(assemble_t1w_scalars, "scalar_files", scalar_maps_for_synthseg, "data_files")
                asl_wf.connect(assemble_t1w_scalars, "scalar_names", scalar_maps_for_synthseg, "colnames")
                scalar_maps_for_synthseg.inputs.mask_file = synthseg_t1w
                scalar_maps_for_synthseg.inputs.output_csv = os.path.join(metrics_dir, "scalar_maps_for_synthseg.csv")
                scalar_maps_for_synthseg.inputs.ignore_background = True
                scalar_maps_for_synthseg.inputs.statistic = "mean"

            # --- 2. FS aparc+aseg (FS space -> T1w) ---
            if self.session.freesurfer_dir is not None:
                dk_fs_files = glob.glob(os.path.join(self.session.freesurfer_dir, "mri", "aparc+aseg.mgz"))
                if dk_fs_files:
                    dk_fs = dk_fs_files[0]

                    fs_to_t1w_xfm_node = Node(Tkregister2fs2t1w(), name="fs_to_t1w_xfm")
                    asl_wf.connect(inputnode, "fs_subjects_dir", fs_to_t1w_xfm_node, "fs_subjects_dir")
                    asl_wf.connect(inputnode, "fs_subject_id", fs_to_t1w_xfm_node, "fs_subject_id")
                    fs_to_t1w_xfm_node.inputs.output_matrix = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-fs_to-T1w_xfm.mat")
                    fs_to_t1w_xfm_node.inputs.output_inverse_matrix = os.path.join(xfm_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-fs_xfm.mat")

                    aseg_mgz_to_nifti_node = Node(MRIConvert(), name="aseg_mgz_to_nifti")
                    aseg_mgz_to_nifti_node.inputs.in_file = dk_fs
                    aseg_mgz_to_nifti_node.inputs.out_file = os.path.join(metrics_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-fs_aparcaseg.nii.gz")

                    fs_aparcaseg_to_t1w_node = Node(FLIRT(), name="fs_aparcaseg_to_t1w")
                    asl_wf.connect(aseg_mgz_to_nifti_node, "out_file", fs_aparcaseg_to_t1w_node, "in_file")
                    asl_wf.connect(fs_to_t1w_xfm_node, "output_matrix", fs_aparcaseg_to_t1w_node, "in_matrix_file")
                    asl_wf.connect(inputnode, "t1w_file", fs_aparcaseg_to_t1w_node, "reference")
                    fs_aparcaseg_to_t1w_node.inputs.interp = "nearestneighbour"
                    fs_aparcaseg_to_t1w_node.inputs.apply_xfm = True
                    fs_aparcaseg_to_t1w_node.inputs.out_file = os.path.join(metrics_dir, f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_space-T1w_aparcaseg.nii.gz")

                    scalar_maps_for_aparcaseg = Node(CalculateScalarMaps(), name="scalar_maps_for_aparcaseg")
                    asl_wf.connect(assemble_t1w_scalars, "scalar_files", scalar_maps_for_aparcaseg, "data_files")
                    asl_wf.connect(assemble_t1w_scalars, "scalar_names", scalar_maps_for_aparcaseg, "colnames")
                    asl_wf.connect(fs_aparcaseg_to_t1w_node, "out_file", scalar_maps_for_aparcaseg, "mask_file")
                    scalar_maps_for_aparcaseg.inputs.output_csv = os.path.join(metrics_dir, "scalar_maps_for_aparcaseg.csv")
                    scalar_maps_for_aparcaseg.inputs.ignore_background = True
                    scalar_maps_for_aparcaseg.inputs.statistic = "mean"

            # --- 3. ArterialAtlas (MNI space, direct) ---
            scalar_maps_for_arterial = Node(CalculateScalarMaps(), name="scalar_maps_for_arterial")
            asl_wf.connect(assemble_mni_scalars, "scalar_files", scalar_maps_for_arterial, "data_files")
            asl_wf.connect(assemble_mni_scalars, "scalar_names", scalar_maps_for_arterial, "colnames")
            scalar_maps_for_arterial.inputs.mask_file = get_package_path("data", "atlas", "ArterialAtlas", "ArterialAtlas.nii.gz")
            scalar_maps_for_arterial.inputs.output_csv = os.path.join(metrics_dir, "scalar_maps_for_arterial_atlas.csv")
            scalar_maps_for_arterial.inputs.ignore_background = True
            scalar_maps_for_arterial.inputs.statistic = "mean"

            # --- 4. Schirmer VT (MNI space, direct) ---
            scalar_maps_for_vt = Node(CalculateScalarMaps(), name="scalar_maps_for_vt")
            asl_wf.connect(assemble_mni_scalars, "scalar_files", scalar_maps_for_vt, "data_files")
            asl_wf.connect(assemble_mni_scalars, "scalar_names", scalar_maps_for_vt, "colnames")
            scalar_maps_for_vt.inputs.mask_file = get_package_path("data", "atlas", "Schirmer_VT", "mni_vascular_territories.nii.gz")
            scalar_maps_for_vt.inputs.output_csv = os.path.join(metrics_dir, "scalar_maps_for_vt_atlas.csv")
            scalar_maps_for_vt.inputs.ignore_background = True
            scalar_maps_for_vt.inputs.statistic = "mean"

        return asl_wf

    def extract_results(self):
        import re
        import pandas as pd

        os.makedirs(self.output_path, exist_ok=True)
        asl_output_path = self.extract_from if self.extract_from else self.output_path

        if not asl_output_path or not os.path.exists(asl_output_path):
            raise FileNotFoundError(f"Output path does not exist: {asl_output_path}")

        # --- Parse FreeSurferColorLUT.txt ---
        fs_lut = {}
        lut_path = get_package_path("data", "labelconvert_in", "FreeSurferColorLUT.txt")
        if os.path.exists(lut_path):
            with open(lut_path, "r") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = re.match(r"^(\d+)\s+(.+?)\s+\d+\s+\d+\s+\d+\s+\d+\s*$", line)
                    if m:
                        fs_lut[int(m.group(1))] = m.group(2).strip()

        # --- Parse atlas CSVs ---
        def _load_atlas_lut(csv_path, include_hemisphere=False):
            lut = {}
            if not os.path.exists(csv_path):
                return lut
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row["label"]
                    if include_hemisphere and row.get("hemisphere", "") not in ("None", "", None):
                        label = f"{label}_{row['hemisphere']}"
                    lut[int(row["id"])] = label
            return lut

        arterial_lut = _load_atlas_lut(get_package_path("data", "atlas", "ArterialAtlas", "ArterialAtlas.csv"))
        vt_lut = _load_atlas_lut(get_package_path("data", "atlas", "Schirmer_VT", "mni_vascular_territories.csv"), include_hemisphere=True)

        # --- Helpers ---
        def _safe_read_csv(csv_path):
            if csv_path is None or not os.path.exists(csv_path):
                return None
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"[WARN] Failed to read CSV: {csv_path}. Error: {e}")
                return None

        def _collect_labels(csv_path):
            df = _safe_read_csv(csv_path)
            if df is None or df.empty or "roi_label" not in df.columns:
                return []
            return sorted(df["roi_label"].dropna().astype(int).unique().tolist())

        def _read_metrics(csv_path, lut, region_ids, metrics):
            result = {f"{m}_{lut.get(rid, f'label_{rid}')}": None for rid in region_ids for m in metrics}
            df = _safe_read_csv(csv_path)
            if df is None or df.empty or "roi_label" not in df.columns:
                return result
            for _, row in df.iterrows():
                rid = int(row["roi_label"])
                if rid not in region_ids:
                    continue
                name = lut.get(rid, f"label_{rid}")
                for m in metrics:
                    if m in df.columns:
                        val = row[m]
                        result[f"{m}_{name}"] = val if pd.notna(val) else None
            return result

        # --- Discover all metrics across subjects ---
        all_metrics = set()
        mask_types = {
            "synthseg": "scalar_maps_for_synthseg.csv",
            "aparcaseg": "scalar_maps_for_aparcaseg.csv",
            "arterial_atlas": "scalar_maps_for_arterial_atlas.csv",
            "vt_atlas": "scalar_maps_for_vt_atlas.csv",
        }
        region_sets = {key: set() for key in mask_types}

        subject_sessions = []
        for subj_folder in sorted(os.listdir(asl_output_path)):
            subj_path = os.path.join(asl_output_path, subj_folder)
            if not os.path.isdir(subj_path) or not subj_folder.startswith("sub-"):
                continue
            subj_id = subj_folder.replace("sub-", "", 1)
            ses_folders = sorted([f for f in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, f)) and f.startswith("ses-")])
            if not ses_folders:
                ses_folders = ["N/A"]
            for ses_folder in ses_folders:
                if ses_folder == "N/A":
                    ses_id = "N/A"
                    stats_dir = os.path.join(subj_path, "asl_metrics_stats")
                else:
                    ses_id = ses_folder.replace("ses-", "", 1)
                    stats_dir = os.path.join(subj_path, ses_folder, "asl_metrics_stats")

                # Discover which metrics columns exist
                for key, csv_name in mask_types.items():
                    csv_p = os.path.join(stats_dir, csv_name)
                    df = _safe_read_csv(csv_p)
                    if df is not None and not df.empty:
                        labels = _collect_labels(csv_p)
                        # Remove single-voxel artifact from VT atlas (atlas creator's mistake)
                        if key == "vt_atlas" and 19 in labels:
                            labels.remove(19)
                        region_sets[key].update(labels)
                        metric_cols = [c for c in df.columns if c != "roi_label"]
                        all_metrics.update(metric_cols)

                subject_sessions.append((subj_id, ses_id, stats_dir))

        all_metrics = sorted(all_metrics)

        # --- Build Excel per mask type ---
        mask_lut = {
            "synthseg": fs_lut,
            "aparcaseg": fs_lut,
            "arterial_atlas": arterial_lut,
            "vt_atlas": vt_lut,
        }
        mask_label = {
            "synthseg": "SynthSeg",
            "aparcaseg": "FS_aparcaseg",
            "arterial_atlas": "ArterialAtlas",
            "vt_atlas": "SchirmerVT",
        }

        for key in mask_types:
            regions = sorted(region_sets[key])
            if not regions:
                continue
            lut = mask_lut[key]
            cols = ["Subject", "Session"] + [f"{m}_{lut.get(rid, f'label_{rid}')}" for rid in regions for m in all_metrics]
            rows = []
            for subj_id, ses_id, stats_dir in subject_sessions:
                csv_p = os.path.join(stats_dir, mask_types[key])
                row_base = {"Subject": f"sub-{subj_id}", "Session": f"ses-{ses_id}" if ses_id != "N/A" else "N/A"}
                rd = dict(row_base)
                rd.update(_read_metrics(csv_p, lut, regions, all_metrics))
                rows.append(rd)
            df = pd.DataFrame(rows, columns=cols)
            xlsx_path = os.path.join(self.output_path, f"asl_{mask_label[key]}_summary.xlsx")
            df.to_excel(xlsx_path, index=False)
            print(f"[ASL Pipeline] Saved {mask_label[key]} summary: {xlsx_path}")

        print(f"[ASL Pipeline] Results extracted successfully from: {asl_output_path}")
        print(f"[ASL Pipeline] Total subject-sessions: {len(subject_sessions)}")