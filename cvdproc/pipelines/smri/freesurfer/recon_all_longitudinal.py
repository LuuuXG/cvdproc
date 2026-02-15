import os
import glob
from nipype.interfaces.base import CommandLineInputSpec, TraitedSpec, CommandLine, File, Directory, traits

from cvdproc.config.paths import get_package_path

recon_all_longitudinal_script = get_package_path(
    "pipelines", "bash", "freesurfer", "freesurfer_reconall_longitudinal_single.sh"
)

class FreesurferLongitudinalInputSpec(CommandLineInputSpec):
    bids_dir = Directory(
        exists=True,
        mandatory=True,
        desc="BIDS root directory, e.g. /mnt/f/BIDS/WCH_SVD_3T_BIDS",
        argstr="%s",
        position=0,
    )
    subject_id = traits.Str(
        mandatory=True,
        desc="Subject ID without 'sub-' prefix, e.g. SSI0188",
        argstr="%s",
        position=1,
    )

    subregion_ha = traits.Bool(
        False,
        usedefault=True,
        mandatory=False,
        desc="Run hippo-amygdala segmentation on longitudinal timepoints only",
    )
    subregion_thalamus = traits.Bool(
        False,
        usedefault=True,
        mandatory=False,
        desc="Run thalamus nuclei segmentation on longitudinal timepoints only",
    )
    subregion_brainstem = traits.Bool(
        False,
        usedefault=True,
        mandatory=False,
        desc="Run brainstem segmentation on longitudinal timepoints only",
    )
    subregion_hypothalamus = traits.Bool(
        False,
        usedefault=True,
        mandatory=False,
        desc="Run hypothalamus segmentation on longitudinal timepoints only",
    )

class FreesurferLongitudinalOutputSpec(TraitedSpec):
    subject_dir = Directory(desc="FreeSurfer subject directory under derivatives/freesurfer/sub-XXX")
    subjects_dir = Directory(desc="SUBJECTS_DIR used by the script")
    base_aseg = File(desc="Base aseg.mgz")
    base_tps = File(desc="base-tps file created by FreeSurfer longitudinal pipeline")

    long_subject_dirs = traits.List(
        Directory,
        desc="List of longitudinal subject directories (e.g., ses-baseline.long.sub-XXX)",
    )

class FreesurferLongitudinal(CommandLine):
    _cmd = f"bash {recon_all_longitudinal_script}"
    input_spec = FreesurferLongitudinalInputSpec
    output_spec = FreesurferLongitudinalOutputSpec

    def _format_flag(self, v: bool) -> str:
        return "1" if bool(v) else "0"

    def _parse_inputs(self, skip=None):
        cmd = super()._parse_inputs(skip=skip)
        cmd += [
            self._format_flag(self.inputs.subregion_ha),
            self._format_flag(self.inputs.subregion_thalamus),
            self._format_flag(self.inputs.subregion_brainstem),
            self._format_flag(self.inputs.subregion_hypothalamus),
        ]
        return cmd

    def _list_outputs(self):
        outputs = self.output_spec().get()

        subject = f"sub-{self.inputs.subject_id}"
        subject_dir = os.path.join(self.inputs.bids_dir, "derivatives", "freesurfer", subject)

        outputs["subject_dir"] = subject_dir
        outputs["subjects_dir"] = subject_dir

        outputs["base_aseg"] = os.path.join(subject_dir, subject, "mri", "aseg.mgz")
        outputs["base_tps"] = os.path.join(subject_dir, subject, "base-tps")

        # Longitudinal directories live directly under SUBJECTS_DIR (subject_dir)
        # e.g. <SUBJECTS_DIR>/ses-baseline.long.sub-SSI0221
        pat = os.path.join(subject_dir, f"ses-*.long.{subject}")
        long_dirs = [p for p in glob.glob(pat) if os.path.isdir(p)]
        long_dirs.sort()

        outputs["long_subject_dirs"] = long_dirs
        return outputs
