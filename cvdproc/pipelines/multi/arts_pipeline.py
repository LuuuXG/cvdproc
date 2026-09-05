import os
import re
import glob
import subprocess
import nibabel as nib
import numpy as np
import pandas as pd

from nipype.interfaces.freesurfer import ReconAll
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface

from cvdproc.pipelines.multi.arts.arts_nipype import ARTSFast, ARTSQuick, PrepareARTSMNIFA
#from ...bids_data.rename_bids_file import rename_bids_file


class ARTSPipeline:
    def __init__(self, subject, session, output_path, method="v2", extract_from=None, **kwargs):
        """
        ARTS: Arteriolosclerosis Biomarker
        10x faster than original pipeline

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Output directory to save results.
            method (str): ARTS implementation to use. ``v2`` transforms the native
                dtifit FA map through the existing DWI-to-T1w affine and
                T1w-to-MNI152NLin6Asym warp. Legacy ``v1`` is unavailable because
                its ARTS.sif image has been removed.
            extract_from (str): ARTS output directory to extract results from.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.method = str(method).lower()
        if self.method not in {"v1", "v2"}:
            raise ValueError(f"Unsupported ARTS method: {method}. Expected 'v1' or 'v2'.")
        if self.method == "v1":
            raise RuntimeError("ARTS method='v1' requires the legacy ARTS.sif image, which is no longer included in cvdproc. Use method='v2'.")
        self.extract_from = extract_from

    def check_data_requirements(self):
        def get_dir(name):
            attr = getattr(self.session, f"{name}_dir", None)
            if callable(attr):
                return attr()
            if attr is not None:
                return attr
            if hasattr(self.session, "_find_output"):
                return self.session._find_output(name)
            return None

        wmh_dir = get_dir("wmh_quantification")
        dwi_dir = get_dir("dwi_pipeline")
        anat_seg_dir = get_dir("anat_seg")

        common_ready = wmh_dir is not None and os.path.isdir(wmh_dir) and dwi_dir is not None and os.path.isdir(dwi_dir) and anat_seg_dir is not None and os.path.isdir(anat_seg_dir)
        xfm_dir = get_dir("xfm")
        return common_ready and xfm_dir is not None and os.path.isdir(xfm_dir)

    def create_workflow(self):
        participant_tsv = os.path.join(self.subject.bids_dir, "participants.tsv")
        def get_dir(name):
            attr = getattr(self.session, f"{name}_dir", None)
            if callable(attr):
                return attr()
            if attr is not None:
                return attr
            if hasattr(self.session, "_find_output"):
                return self.session._find_output(name)
            return None

        def find_one_file(root_dir, include_keywords, desc, prefer_keywords=None):
            if root_dir is None or not os.path.isdir(root_dir):
                raise FileNotFoundError(f"{desc} directory not found: {root_dir}")

            prefer_keywords = prefer_keywords or []
            files = sorted(glob.glob(os.path.join(root_dir, "**", "*.nii*"), recursive=True))
            matched = [f for f in files if all(k in os.path.basename(f) for k in include_keywords)]

            if len(matched) == 0:
                raise FileNotFoundError(f"No matched {desc} found under {root_dir}. Required keywords: {include_keywords}")

            if len(matched) > 1 and prefer_keywords:
                preferred = [f for f in matched if all(k in os.path.basename(f) for k in prefer_keywords)]
                if len(preferred) == 1:
                    return os.path.abspath(preferred[0])
                if len(preferred) > 1:
                    matched = preferred

            if len(matched) > 1:
                raise RuntimeError(f"Multiple matched files found for {desc} under {root_dir}:\n" + "\n".join(matched))

            return os.path.abspath(matched[0])

        def strip_prefix(value, prefix):
            value = str(value).strip()
            return value[len(prefix):] if value.startswith(prefix) else value

        def convert_age(age_raw):
            age_raw = str(age_raw).strip()
            m = re.search(r"\d+", age_raw)
            if m is None:
                raise ValueError(f"Cannot parse age from participants.tsv: {age_raw}")
            return int(m.group(0))

        def convert_sex(sex_raw):
            """
            ARTS official coding:
                1 = male
                0 = female
            """
            sex_raw = str(sex_raw).strip()
            sex_lower = sex_raw.lower()

            male_values = {"1", "m", "male", "man"}
            female_values = {"0", "f", "female", "woman"}

            if sex_lower in male_values:
                return "1"

            if sex_lower in female_values:
                return "0"

            raise ValueError(f"Cannot convert sex value to ARTS code. Expected M/F, Male/Female, 1/0; got: {sex_raw}")

        xfm_dir = get_dir("xfm")
        wmh_dir = get_dir("wmh_quantification")
        dwi_dir = get_dir("dwi_pipeline")
        anat_seg_dir = get_dir("anat_seg")

        if xfm_dir is None or not os.path.isdir(xfm_dir):
            raise FileNotFoundError(f"xfm directory not found: {xfm_dir}")
        if wmh_dir is None or not os.path.isdir(wmh_dir):
            raise FileNotFoundError(f"wmh_quantification directory not found: {wmh_dir}")
        if dwi_dir is None or not os.path.isdir(dwi_dir):
            raise FileNotFoundError(f"dwi_pipeline directory not found: {dwi_dir}")
        if anat_seg_dir is None or not os.path.isdir(anat_seg_dir):
            raise FileNotFoundError(f"anat_seg directory not found: {anat_seg_dir}")

        synthseg_dir = os.path.join(anat_seg_dir, "synthseg")
        if not os.path.isdir(synthseg_dir):
            synthseg_dir = anat_seg_dir

        synthseg = find_one_file(synthseg_dir, ["space-T1w_synthseg"], "SynthSeg segmentation in T1w space", prefer_keywords=["acq-highres", "space-T1w_synthseg"])

        wmh_mask = find_one_file(wmh_dir, ["space-T1w_label-WMH", "_mask"], "WMH mask in T1w space", prefer_keywords=["desc-truenetThr0p30", "space-T1w_label-WMH", "_mask"])

        if self.method == "v1":
            t1w_brain = find_one_file(xfm_dir, ["space-T1w_desc-brain_T1w"], "T1w brain image in T1w space", prefer_keywords=["acq-highres", "space-T1w_desc-brain_T1w"])
            flair_brain = find_one_file(xfm_dir, ["space-T1w_desc-brain_FLAIR"], "FLAIR brain image in T1w space", prefer_keywords=["acq-highres", "space-T1w_desc-brain_FLAIR"])
            dtifit_dir = os.path.join(dwi_dir, "dtifit")
            if not os.path.isdir(dtifit_dir):
                dtifit_dir = dwi_dir
            fa = find_one_file(dtifit_dir, ["model-tensor_param-fa_dwimap"], "native DTI FA map from dtifit", prefer_keywords=["acq-DSIb4000", "dir-AP", "model-tensor_param-fa_dwimap"])
        else:
            dtifit_dir = os.path.join(dwi_dir, "dtifit")
            if not os.path.isdir(dtifit_dir):
                raise FileNotFoundError(
                    f"dtifit directory not found: {dtifit_dir}. "
                    "ARTS method='v2' requires a native FA map generated by the DWI pipeline with dti_fit enabled."
                )

            fa = find_one_file(
                dtifit_dir,
                ["model-tensor_param-fa_dwimap"],
                "native DTI FA map from dtifit",
            )
            space_match = re.search(r"_space-([^_]+)", os.path.basename(fa))
            if space_match is None:
                raise ValueError(f"Cannot determine DWI space from dtifit FA filename: {fa}")
            dwi_space = space_match.group(1)

            dwi_to_t1w_matches = sorted(
                glob.glob(os.path.join(xfm_dir, f"*_from-{dwi_space}_to-T1w_xfm.mat"))
            )
            if len(dwi_to_t1w_matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one DWI-to-T1w affine for space '{dwi_space}' under {xfm_dir}, "
                    f"but found {len(dwi_to_t1w_matches)}:\n" + "\n".join(dwi_to_t1w_matches)
                )
            dwi_to_t1w_affine = os.path.abspath(dwi_to_t1w_matches[0])

            t1w_to_mni_matches = sorted(
                glob.glob(os.path.join(xfm_dir, "*_from-T1w_to-MNI152NLin6ASym_warp.nii.gz"))
            )
            if len(t1w_to_mni_matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one T1w-to-MNI152NLin6Asym warp under {xfm_dir}, "
                    f"but found {len(t1w_to_mni_matches)}:\n" + "\n".join(t1w_to_mni_matches)
                )
            t1w_to_mni_warp = os.path.abspath(t1w_to_mni_matches[0])

            t1w_reference_matches = sorted(
                f for f in glob.glob(os.path.join(xfm_dir, "**", "*.nii*"), recursive=True)
                if "desc-brain_T1w" in os.path.basename(f)
            )
            if len(t1w_reference_matches) == 1:
                t1w_reference = os.path.abspath(t1w_reference_matches[0])
            elif len(t1w_reference_matches) > 1:
                raise RuntimeError(
                    f"Multiple T1w brain references found under {xfm_dir}:\n"
                    + "\n".join(t1w_reference_matches)
                )
            else:
                t1w_files = self.session.get_t1w_files()
                if not t1w_files:
                    raise FileNotFoundError(
                        "No T1w reference found in the xfm directory or the source BIDS session."
                    )
                t1w_reference = os.path.abspath(t1w_files[0])

        if not os.path.exists(participant_tsv):
            raise FileNotFoundError(f"participants.tsv not found: {participant_tsv}")

        participants = pd.read_csv(participant_tsv, sep="\t", dtype=str).fillna("")

        required_cols = {"participant_id", "session_id", "age", "sex"}
        missing_cols = required_cols - set(participants.columns)
        if missing_cols:
            raise ValueError(f"participants.tsv is missing required columns: {sorted(missing_cols)}")

        subject_id_no_prefix = strip_prefix(self.subject.subject_id, "sub-")
        session_id_no_prefix = strip_prefix(self.session.session_id, "ses-")

        participants["_participant_id_norm"] = participants["participant_id"].apply(lambda x: strip_prefix(x, "sub-"))
        participants["_session_id_norm"] = participants["session_id"].apply(lambda x: strip_prefix(x, "ses-"))

        row_df = participants[(participants["_participant_id_norm"] == subject_id_no_prefix) & (participants["_session_id_norm"] == session_id_no_prefix)]

        if len(row_df) == 0:
            raise ValueError(f"No matching row found in participants.tsv for sub-{subject_id_no_prefix}, ses-{session_id_no_prefix}")

        if len(row_df) > 1:
            raise ValueError(f"Multiple matching rows found in participants.tsv for sub-{subject_id_no_prefix}, ses-{session_id_no_prefix}")

        age = convert_age(row_df.iloc[0]["age"])
        sex = convert_sex(row_df.iloc[0]["sex"])

        arts_wf = Workflow(name="arts_workflow")

        fields = ["subject_id", "age", "sex", "fa", "synthseg", "wmh_mask"]
        fields += ["t1w_brain", "flair_brain", "output_root"] if self.method == "v1" else [
            "dwi_to_t1w_affine", "t1w_reference", "t1w_to_mni_warp", "output_dir"
        ]
        inputnode = Node(IdentityInterface(fields=fields), name="inputnode")

        if self.method == "v1":
            inputnode.inputs.subject_id = self.session.session_id if str(self.session.session_id).startswith("ses-") else f"ses-{self.session.session_id}"
        else:
            inputnode.inputs.subject_id = self.subject.subject_id if str(self.subject.subject_id).startswith("sub-") else f"sub-{self.subject.subject_id}"
        inputnode.inputs.age = age
        inputnode.inputs.sex = sex
        inputnode.inputs.fa = fa
        inputnode.inputs.synthseg = synthseg
        inputnode.inputs.wmh_mask = wmh_mask
        if self.method == "v1":
            inputnode.inputs.t1w_brain = t1w_brain
            inputnode.inputs.flair_brain = flair_brain
            inputnode.inputs.output_root = os.path.dirname(self.output_path)
            arts_node = Node(ARTSFast(), name="arts_v1")
            connections = [("subject_id", "subject_id"), ("age", "age"), ("sex", "sex"), ("t1w_brain", "t1w_brain"), ("flair_brain", "flair_brain"), ("fa", "fa"), ("synthseg", "synthseg"), ("wmh_mask", "wmh_mask"), ("output_root", "output_root")]
        else:
            inputnode.inputs.output_dir = self.output_path
            inputnode.inputs.dwi_to_t1w_affine = dwi_to_t1w_affine
            inputnode.inputs.t1w_reference = t1w_reference
            inputnode.inputs.t1w_to_mni_warp = t1w_to_mni_warp

            subject_label = self.subject.subject_id if str(self.subject.subject_id).startswith("sub-") else f"sub-{self.subject.subject_id}"
            session_label = self.session.session_id if str(self.session.session_id).startswith("ses-") else f"ses-{self.session.session_id}"
            prepare_fa_node = Node(PrepareARTSMNIFA(), name="prepare_arts_mni_fa")
            prepare_fa_node.inputs.output_file = os.path.join(
                self.output_path,
                "input_preparation",
                f"{subject_label}_{session_label}_space-MNI152NLin6ASym_model-tensor_param-fa_dwimap.nii.gz",
            )
            arts_wf.connect(inputnode, "fa", prepare_fa_node, "fa_file")
            arts_wf.connect(inputnode, "dwi_to_t1w_affine", prepare_fa_node, "dwi_to_t1w_affine")
            arts_wf.connect(inputnode, "t1w_reference", prepare_fa_node, "t1w_reference")
            arts_wf.connect(inputnode, "t1w_to_mni_warp", prepare_fa_node, "t1w_to_mni_warp")

            arts_node = Node(ARTSQuick(), name="arts_v2")
            connections = [("subject_id", "subject_id"), ("age", "age"), ("sex", "sex"), ("synthseg", "synthseg"), ("wmh_mask", "wmh_mask"), ("output_dir", "output_dir")]
            arts_wf.connect(prepare_fa_node, "output_file", arts_node, "fa_mni")

        arts_wf.connect([(inputnode, arts_node, connections)])

        return arts_wf

    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        arts_output_dir = self.extract_from
        if arts_output_dir is None:
            arts_output_dir = self.output_path

        if not os.path.isdir(arts_output_dir):
            raise FileNotFoundError(f"ARTS output directory not found: {arts_output_dir}")

        results_columns = ["Subject", "Session", "ARTS_score"]
        results_df = pd.DataFrame(columns=results_columns)

        for subject_folder in sorted(os.listdir(arts_output_dir)):
            subject_path = os.path.join(arts_output_dir, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            for session_folder in sorted(os.listdir(subject_path)):
                session_path = os.path.join(subject_path, session_folder)
                if not os.path.isdir(session_path):
                    continue

                score_file = os.path.join(session_path, "analysis", "score.csv")
                if not os.path.exists(score_file):
                    print(f"[WARNING] ARTS score file not found: {score_file}")
                    continue

                score_df = pd.read_csv(score_file, header=None)
                if score_df.shape[1] < 2 or score_df.shape[0] < 1:
                    print(f"[WARNING] Invalid ARTS score file: {score_file}")
                    continue

                arts_score = score_df.iloc[0, 1]
                temp_df = pd.DataFrame([{"Subject": subject_folder, "Session": session_folder, "ARTS_score": arts_score}])
                results_df = pd.concat([results_df, temp_df], ignore_index=True)

        combined_xlsx = os.path.join(self.output_path, "arts_summary.xlsx")
        results_df.to_excel(combined_xlsx, index=False)
        print(f"[INFO] ARTS score summary saved to: {combined_xlsx}")
