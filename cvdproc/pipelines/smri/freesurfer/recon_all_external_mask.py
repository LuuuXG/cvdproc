import os

from nipype.interfaces.base import CommandLine, CommandLineInputSpec, Directory, File, TraitedSpec, traits

from cvdproc.config.paths import get_package_path


recon_all_external_mask_script = get_package_path("pipelines", "bash", "freesurfer", "freesurfer_reconall_external_mask.sh")


class ReconAllExternalMaskInputSpec(CommandLineInputSpec):
    t1w_file = File(exists=True, mandatory=True, desc="Input T1w image", argstr="%s", position=0)
    brain_mask = File(mandatory=True, desc="External brain mask in T1w space; generated with SynthStrip if missing", argstr="%s", position=1)
    subject_id = traits.Str(mandatory=True, desc="FreeSurfer subject ID", argstr="%s", position=2)
    subjects_dir = Directory(mandatory=True, desc="FreeSurfer SUBJECTS_DIR", argstr="%s", position=3)


class ReconAllExternalMaskOutputSpec(TraitedSpec):
    subject_id = traits.Str(desc="FreeSurfer subject ID")
    subjects_dir = Directory(desc="FreeSurfer SUBJECTS_DIR")
    subject_dir = Directory(desc="FreeSurfer subject output directory")
    external_mask_done = File(desc="Marker indicating successful external-mask processing")


class ReconAllExternalMask(CommandLine):
    _cmd = f"bash {recon_all_external_mask_script}"
    input_spec = ReconAllExternalMaskInputSpec
    output_spec = ReconAllExternalMaskOutputSpec
    terminal_output = "allatonce"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subject_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id)
        outputs["subject_id"] = self.inputs.subject_id
        outputs["subjects_dir"] = os.path.abspath(self.inputs.subjects_dir)
        outputs["subject_dir"] = os.path.abspath(subject_dir)
        outputs["external_mask_done"] = os.path.abspath(os.path.join(subject_dir, "scripts", "recon-all.external-mask.done"))
        return outputs
