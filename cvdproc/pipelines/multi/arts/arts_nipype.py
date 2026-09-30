import os
import shutil
import subprocess
import tempfile
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
)
from traits.api import Int, Str

from cvdproc.config.paths import get_package_path


arts_fast_bash = get_package_path("pipelines", "bash", "arts", "run_ARTS_fast_synthmorph.sh")
arts_quick_bash = get_package_path("pipelines", "bash", "arts", "run_ARTS_quick.sh")
arts_mni_to_iit_warp = get_package_path("data", "standard", "MNI152", "custom", "from-MNI152NLin6Asym_to-IIT_warp.nii.gz")


class PrepareARTSMNIFAInputSpec(BaseInterfaceInputSpec):
    fa_file = File(exists=True, mandatory=True, desc="Native-space FA map from dtifit.")
    dwi_to_t1w_affine = File(exists=True, mandatory=True, desc="DWI-to-T1w FLIRT affine.")
    t1w_reference = File(exists=True, mandatory=True, desc="T1w reference used by the DWI-to-T1w affine.")
    t1w_to_mni_warp = File(exists=True, mandatory=True, desc="T1w-to-MNI152NLin6Asym warp.")
    output_file = File(mandatory=True, desc="ARTS FA map in MNI152NLin6Asym space.")


class PrepareARTSMNIFAOutputSpec(TraitedSpec):
    output_file = File(desc="ARTS FA map in MNI152NLin6Asym space.")


class PrepareARTSMNIFA(BaseInterface):
    """Transform a dtifit FA map through DWI -> T1w -> MNI space."""

    input_spec = PrepareARTSMNIFAInputSpec
    output_spec = PrepareARTSMNIFAOutputSpec

    def _run_interface(self, runtime):
        output_file = os.path.abspath(self.inputs.output_file)
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="_arts_fa_mni_", dir=output_dir)
        fa_t1w = os.path.join(work_dir, "fa_t1w.nii.gz")

        try:
            subprocess.run(
                [
                    "flirt",
                    "-in", self.inputs.fa_file,
                    "-ref", self.inputs.t1w_reference,
                    "-out", fa_t1w,
                    "-applyxfm",
                    "-init", self.inputs.dwi_to_t1w_affine,
                    "-interp", "trilinear",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "mri_convert",
                    "-at", self.inputs.t1w_to_mni_warp,
                    fa_t1w,
                    output_file,
                ],
                check=True,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_file"] = os.path.abspath(self.inputs.output_file)
        return outputs


class ARTSFastInputSpec(CommandLineInputSpec):
    subject_id = Str(mandatory=True, argstr="%s", position=0, desc="Subject ID used in ARTS output, without the 'sub-' prefix if possible.")
    age = Int(mandatory=True, argstr="%d", position=1, desc="Age used by the ARTS classifier.")
    sex = Str(mandatory=True, argstr="%s", position=2, desc="Sex coding used by the ARTS classifier. Keep consistent with the original ARTS convention.")
    t1w_brain = File(exists=True, mandatory=True, argstr="%s", position=3, desc="Skull-stripped T1w image.")
    flair_brain = File(exists=True, mandatory=True, argstr="%s", position=4, desc="Skull-stripped FLAIR image. It should be aligned with T1w if an identity T1-to-FLAIR matrix is used.")
    fa = File(exists=True, mandatory=True, argstr="%s", position=5, desc="Native FA image.")
    synthseg = File(exists=True, mandatory=True, argstr="%s", position=6, desc="SynthSeg segmentation in T1w space. Labels 2 and 41 are used to extract cerebral white matter.")
    wmh_mask = File(exists=True, mandatory=True, argstr="%s", position=7, desc="Binary WMH mask in T1w space or in a space aligned with the T1w-derived SynthSeg mask.")
    output_root = Directory(mandatory=True, argstr="%s", position=8, desc="Root output directory. The subject-level output will be saved under output_root/subject_id.")


class ARTSFastOutputSpec(TraitedSpec):
    output_dir = Directory(desc="Subject-level ARTS fast output directory.")
    qc_dir = Directory(desc="QC directory.")
    score_csv = File(desc="Final ARTS score CSV.")
    score_batch_csv = File(desc="Copied batch-style score CSV.")
    classifier_input = File(desc="Classifier input text file.")
    wmh_features = File(desc="WMH feature file.")
    fa_features = File(desc="FA ROI feature file.")
    wm_mask = File(desc="Cerebral white matter mask extracted from SynthSeg labels 2 and 41.")
    wmh_mask_bin = File(desc="Binarized WMH mask copied into ARTS-style output.")
    wmh_no_cerebellum = File(desc="WMH mask after applying the cerebral white matter mask.")
    all_fa = File(desc="FA image registered to IITmean_FA space by SynthMorph.")
    all_fa_skeletonised = File(desc="TBSS-skeletonized FA image.")
    iit_fa_qc = File(desc="IITmean_FA copy for QC.")
    all_fa_qc = File(desc="SynthMorph-registered FA copy for QC.")


class ARTSFast(CommandLine):
    input_spec = ARTSFastInputSpec
    output_spec = ARTSFastOutputSpec
    _cmd = f"bash {arts_fast_bash}"

    def _run_interface(self, runtime):
        os.makedirs(os.path.abspath(self.inputs.output_root), exist_ok=True)

        if not os.path.exists(arts_fast_bash):
            raise FileNotFoundError(f"ARTS fast bash script not found: {arts_fast_bash}")

        if not os.access(arts_fast_bash, os.X_OK):
            os.chmod(arts_fast_bash, 0o755)

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()

        subject_id = self.inputs.subject_id
        output_root = os.path.abspath(self.inputs.output_root)
        output_dir = os.path.join(output_root, subject_id)
        qc_dir = os.path.join(output_dir, "QC")

        outputs["output_dir"] = output_dir
        outputs["qc_dir"] = qc_dir

        outputs["score_csv"] = os.path.join(output_dir, "analysis", "score.csv")
        outputs["score_batch_csv"] = os.path.join(output_root, f"score_batch_{subject_id}_fast_synthmorph.csv")
        outputs["classifier_input"] = os.path.join(output_dir, "analysis", "classifier_input.txt")

        outputs["wmh_features"] = os.path.join(output_dir, "WMH_processing", "features.txt")
        outputs["fa_features"] = os.path.join(output_dir, "FA_processing", "features.txt")

        outputs["wm_mask"] = os.path.join(output_dir, "GMWM", "WM_mask.nii.gz")
        outputs["wmh_mask_bin"] = os.path.join(output_dir, "WMH", "WMH_mask.nii.gz")
        outputs["wmh_no_cerebellum"] = os.path.join(output_dir, "WMH_processing", "WMH_no_cerebellum.nii.gz")

        outputs["all_fa"] = os.path.join(output_dir, "FA_processing", "tbss", "stats", "all_FA.nii.gz")
        outputs["all_fa_skeletonised"] = os.path.join(output_dir, "FA_processing", "tbss", "stats", "all_FA_skeletonised.nii.gz")

        outputs["iit_fa_qc"] = os.path.join(qc_dir, "IITmean_FA.nii.gz")
        outputs["all_fa_qc"] = os.path.join(qc_dir, "all_FA_synthmorph_to_IIT.nii.gz")

        return outputs


class ARTSQuickInputSpec(CommandLineInputSpec):
    subject_id = Str(mandatory=True, argstr="%s", position=0, desc="Subject ID used by the ARTS classifier.")
    age = Int(mandatory=True, argstr="%d", position=1, desc="Age used by the ARTS classifier.")
    sex = Str(mandatory=True, argstr="%s", position=2, desc="Sex coding used by the ARTS classifier.")
    synthseg = File(exists=True, mandatory=True, argstr="%s", position=3, desc="SynthSeg segmentation in T1w space.")
    wmh_mask = File(exists=True, mandatory=True, argstr="%s", position=4, desc="Binary WMH mask aligned with the SynthSeg segmentation.")
    fa_mni = File(exists=True, mandatory=True, argstr="%s", position=5, desc="FA image registered to MNI152NLin6Asym space.")
    output_dir = Directory(mandatory=True, argstr="%s", position=6, desc="ARTS output directory for this subject and session.")
    mni_to_iit_warp = File(arts_mni_to_iit_warp, exists=True, mandatory=False, usedefault=True, argstr="%s", position=7, desc="Warp from MNI152NLin6Asym to IIT space.")


class ARTSQuickOutputSpec(TraitedSpec):
    output_dir = Directory(desc="ARTS quick output directory.")
    qc_dir = Directory(desc="QC directory.")
    score_csv = File(desc="Final ARTS score CSV.")
    classifier_input = File(desc="Classifier input text file.")
    wmh_features = File(desc="WMH feature file.")
    fa_features = File(desc="FA ROI feature file.")
    wm_mask = File(desc="Cerebral white matter mask extracted from SynthSeg labels 2 and 41.")
    wmh_mask_bin = File(desc="Binarized WMH mask copied into ARTS-style output.")
    wmh_no_cerebellum = File(desc="WMH mask after applying the cerebral white matter mask.")
    all_fa = File(desc="FA image transformed from MNI152NLin6Asym to IIT space.")
    all_fa_skeletonised = File(desc="TBSS-skeletonized FA image.")
    iit_fa_qc = File(desc="IITmean_FA copy for QC.")
    all_fa_qc = File(desc="IIT-space FA copy for QC.")


class ARTSQuick(CommandLine):
    input_spec = ARTSQuickInputSpec
    output_spec = ARTSQuickOutputSpec
    _cmd = f"bash {arts_quick_bash}"

    def _run_interface(self, runtime):
        os.makedirs(os.path.abspath(self.inputs.output_dir), exist_ok=True)
        if not os.path.exists(arts_quick_bash):
            raise FileNotFoundError(f"ARTS quick bash script not found: {arts_quick_bash}")
        if not os.access(arts_quick_bash, os.X_OK):
            os.chmod(arts_quick_bash, 0o755)
        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        output_dir = os.path.abspath(self.inputs.output_dir)
        qc_dir = os.path.join(output_dir, "QC")
        outputs["output_dir"] = output_dir
        outputs["qc_dir"] = qc_dir
        outputs["score_csv"] = os.path.join(output_dir, "analysis", "score.csv")
        outputs["classifier_input"] = os.path.join(output_dir, "analysis", "classifier_input.txt")
        outputs["wmh_features"] = os.path.join(output_dir, "WMH_processing", "features.txt")
        outputs["fa_features"] = os.path.join(output_dir, "FA_processing", "features.txt")
        outputs["wm_mask"] = os.path.join(output_dir, "GMWM", "WM_mask.nii.gz")
        outputs["wmh_mask_bin"] = os.path.join(output_dir, "WMH", "WMH_mask.nii.gz")
        outputs["wmh_no_cerebellum"] = os.path.join(output_dir, "WMH_processing", "WMH_no_cerebellum.nii.gz")
        outputs["all_fa"] = os.path.join(output_dir, "FA_processing", "tbss", "stats", "all_FA.nii.gz")
        outputs["all_fa_skeletonised"] = os.path.join(output_dir, "FA_processing", "tbss", "stats", "all_FA_skeletonised.nii.gz")
        outputs["iit_fa_qc"] = os.path.join(qc_dir, "IITmean_FA.nii.gz")
        outputs["all_fa_qc"] = os.path.join(qc_dir, "all_FA_to_IIT.nii.gz")
        return outputs

if __name__ == "__main__":
    arts = ARTSFast()

    arts.inputs.subject_id = "AFib0241"
    arts.inputs.age = 75
    arts.inputs.sex = "0"

    arts.inputs.t1w_brain = (
        "/mnt/e/Neuroimage/ARTS/input/sub-AFib0241/"
        "sub-AFib0241_ses-baseline_acq-highres_desc-brain_T1w.nii.gz"
    )

    arts.inputs.flair_brain = (
        "/mnt/e/Neuroimage/ARTS/input/sub-AFib0241/"
        "sub-AFib0241_ses-baseline_acq-highres_space-T1w_desc-brain_FLAIR.nii.gz"
    )

    arts.inputs.fa = (
        "/mnt/e/Neuroimage/ARTS/input/sub-AFib0241/"
        "sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-tensor_param-fa_dwimap.nii.gz"
    )

    arts.inputs.synthseg = (
        "/mnt/e/Neuroimage/ARTS/input/sub-AFib0241/"
        "sub-AFib0241_ses-baseline_acq-highres_space-T1w_synthseg.nii.gz"
    )

    arts.inputs.wmh_mask = (
        "/mnt/e/Neuroimage/ARTS/input/sub-AFib0241/"
        "sub-AFib0241_ses-baseline_space-T1w_label-WMH_desc-truenetThr0p30_mask.nii.gz"
    )

    arts.inputs.output_root = "/mnt/e/Neuroimage/ARTS/output_fast_synthmorph"
    print("ARTSFast command:")
    print(arts.cmdline)

    result = arts.run()

    print("ARTSFast finished.")
    print(f"Output directory: {result.outputs.output_dir}")
    print(f"QC directory: {result.outputs.qc_dir}")
    print(f"Classifier input: {result.outputs.classifier_input}")
    print(f"Score CSV: {result.outputs.score_csv}")
    print(f"Batch score CSV: {result.outputs.score_batch_csv}")

    if os.path.exists(result.outputs.score_csv):
        with open(result.outputs.score_csv, "r", encoding="utf-8") as f:
            print("ARTS score:")
            print(f.read().strip())
