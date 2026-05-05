import os
import json

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface

from cvdproc.bids_data.rename_bids_file import rename_bids_file
from cvdproc.config.paths import get_package_path

from cvdproc.pipelines.perfusion.exploreasl.exploreasl_nipype import (
    ExploreASLCustom,
    ASLtoT1Register,
)
from cvdproc.pipelines.common.register import (
    MRIConvertApplyWarp,
    SynthmorphNonlinear,
)


class ASLPipeline:
    def __init__(
        self,
        subject: object,
        session: object,
        output_path: str,
        use_which_asl: str = None,
        use_which_t1w: str = None,
        preprocess_method: str = "ExploreASL",
    ):
        self.subject = subject
        self.session = session
        self.output_path = output_path
        self.use_which_asl = use_which_asl
        self.use_which_t1w = use_which_t1w
        self.preprocess_method = preprocess_method

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
                    m0_file = possible_m0_file
                    print(f"[ASL Pipeline] Using separate M0 file: {m0_file}")

        # ===============================
        # Main Workflow
        # ===============================
        asl_wf = Workflow(name="asl_workflow")

        inputnode = Node(
            IdentityInterface(
                fields=["asl_file", "t1w_file", "m0_file", "output_path"]
            ),
            name="inputnode",
        )
        inputnode.inputs.asl_file = asl_file if asl_file else None
        inputnode.inputs.t1w_file = t1w_file if t1w_file else None
        inputnode.inputs.m0_file = m0_file if m0_file else None
        inputnode.inputs.output_path = self.output_path

        # ===============================
        # Check T1 <-> MNI non-linear warp
        # ===============================
        if t1w_file != "":
            t1_to_mni_warp_node = Node(
                IdentityInterface(fields=["warp_image"]),
                name="t1_to_mni_warp_node",
            )
            mni_to_t1_warp_node = Node(
                IdentityInterface(fields=["warp_image"]),
                name="mni_to_t1_warp_node",
            )

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

            if not os.path.exists(target_warp) or not os.path.exists(
                target_inverse_warp
            ):
                print(
                    f"[ASL Pipeline] No existing T1w to MNI warp file found: {target_warp}. "
                    "Will run Synthmorph registration to get the warp (1mm resolution)."
                )
                print(
                    "[ASL Pipeline] If you want a different resolution, "
                    "please run a separate T1 registration pipeline first."
                )

                t1w_to_mni_registration = Node(
                    SynthmorphNonlinear(),
                    name="t1w_to_mni_registration",
                )
                asl_wf.connect(inputnode, "t1w_file", t1w_to_mni_registration, "t1")
                t1w_to_mni_registration.inputs.mni_template = get_package_path(
                    "data",
                    "standard",
                    "MNI152",
                    "MNI152_T1_1mm_brain.nii.gz",
                )
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

                asl_wf.connect(
                    t1w_to_mni_registration,
                    "t1_2_mni_warp",
                    t1_to_mni_warp_node,
                    "warp_image",
                )
                asl_wf.connect(
                    t1w_to_mni_registration,
                    "mni_2_t1_warp",
                    mni_to_t1_warp_node,
                    "warp_image",
                )
            else:
                print(
                    f"[ASL Pipeline] Found existing T1w to MNI warp file: {target_warp}. "
                    "Will use it."
                )
                t1_to_mni_warp_node.inputs.warp_image = target_warp
                mni_to_t1_warp_node.inputs.warp_image = target_inverse_warp

        # ===============================
        # ExploreASL branch
        # ===============================
        if self.preprocess_method.lower() == "exploreasl":
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
            exploreasl_node.inputs.script_path = get_package_path(
                "pipelines",
                "matlab",
                "exploreasl",
                "exploreasl_process.m",
            )
            exploreasl_node.inputs.exploreasl_dir = get_package_path(
                "data",
                "matlab_toolbox",
                "ExploreASL-develop",
            )
            asl_wf.connect(inputnode, "output_path", exploreasl_node, "output_dir")

            # --------------------------------
            # CBF branch (always run)
            # --------------------------------
            cbf_to_t1w_node = Node(ASLtoT1Register(), name="cbf_to_t1w")
            asl_wf.connect(exploreasl_node, "cbf", cbf_to_t1w_node, "asl_space_img")
            asl_wf.connect(
                exploreasl_node,
                "rt1",
                cbf_to_t1w_node,
                "asl_space_t1w_img",
            )
            asl_wf.connect(inputnode, "t1w_file", cbf_to_t1w_node, "target_t1w_img")
            cbf_to_t1w_node.inputs.asl_in_t1w_img = os.path.join(
                self.output_path,
                f"sub-{self.subject.subject_id}_{self.session.session_id}_space-T1w_cbf.nii.gz",
            )

            cbf_to_mni_node = Node(MRIConvertApplyWarp(), name="cbf_to_mni")
            asl_wf.connect(cbf_to_t1w_node, "out_file", cbf_to_mni_node, "input_image")
            asl_wf.connect(
                t1_to_mni_warp_node,
                "warp_image",
                cbf_to_mni_node,
                "warp_image",
            )
            cbf_to_mni_node.inputs.output_image = os.path.join(
                self.output_path,
                f"sub-{self.subject.subject_id}_{self.session.session_id}_space-MNI152NLin6ASym_cbf.nii.gz",
            )

            # --------------------------------
            # ATT branch (optional)
            # --------------------------------
            if has_att:
                att_to_t1w_node = Node(ASLtoT1Register(), name="att_to_t1w")
                asl_wf.connect(
                    exploreasl_node,
                    "att",
                    att_to_t1w_node,
                    "asl_space_img",
                )
                asl_wf.connect(
                    exploreasl_node,
                    "rt1",
                    att_to_t1w_node,
                    "asl_space_t1w_img",
                )
                asl_wf.connect(
                    inputnode,
                    "t1w_file",
                    att_to_t1w_node,
                    "target_t1w_img",
                )
                att_to_t1w_node.inputs.asl_in_t1w_img = os.path.join(
                    self.output_path,
                    f"sub-{self.subject.subject_id}_{self.session.session_id}_space-T1w_att.nii.gz",
                )

                att_to_mni_node = Node(MRIConvertApplyWarp(), name="att_to_mni")
                asl_wf.connect(
                    att_to_t1w_node,
                    "out_file",
                    att_to_mni_node,
                    "input_image",
                )
                asl_wf.connect(
                    t1_to_mni_warp_node,
                    "warp_image",
                    att_to_mni_node,
                    "warp_image",
                )
                att_to_mni_node.inputs.output_image = os.path.join(
                    self.output_path,
                    f"sub-{self.subject.subject_id}_{self.session.session_id}_space-MNI152NLin6ASym_att.nii.gz",
                )

        return asl_wf