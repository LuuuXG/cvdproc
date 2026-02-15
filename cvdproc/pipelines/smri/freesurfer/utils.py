import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Bool, Either, Float, List
import traits

from cvdproc.config.paths import get_package_path

class MRIvol2surfInputSpec(CommandLineInputSpec):
    volume = File(argstr='--mov %s', desc='Input volume file', mandatory=True, exists=True)
    regheader = Str(argstr='--regheader %s', desc='Registration header (subject id)', mandatory=True)
    hemi = Str(argstr='--hemi %s', desc='Hemisphere (lh or rh)', mandatory=True)
    output_surf = File(argstr='--o %s', desc='Output surface file', mandatory=True)
    proj_frac = Float(0.5, argstr='--projfrac %f', desc='Projection fraction', mandatory=False)
    target = Str(argstr='--trgsubject %s', desc='Target subject', mandatory=False)
    subjects_dir = Str(desc='Freesurfer subjects directory', mandatory=True)

class MRIvol2surfOutputSpec(TraitedSpec):
    output_surf = File(desc='Output surface file', exists=True)

class MRIvol2surf(CommandLine):
    _cmd = 'mri_vol2surf'
    input_spec = MRIvol2surfInputSpec
    output_spec = MRIvol2surfOutputSpec

    def _run_interface(self, runtime):
        # set per-node environment (safe for parallel execution)
        self.inputs.environ = getattr(self.inputs, "environ", {})
        self.inputs.environ["SUBJECTS_DIR"] = self.inputs.subjects_dir
        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_surf"] = os.path.abspath(self.inputs.output_surf)
        return outputs
    
class MergeRibbonInputSpec(CommandLineInputSpec):
    subjects_dir = Str(argstr='%s', desc='Freesurfer subjects directory', mandatory=True, position=0)
    subject_id = Str(argstr='%s', desc='Freesurfer subject ID', mandatory=True, position=1)
    output_gm_mask = Str(argstr='%s', desc='Output cortical GM mask file', mandatory=True, position=2)

class MergeRibbonOutputSpec(TraitedSpec):
    output_gm_mask = File(desc='Output cortical GM mask file', exists=True)

class MergeRibbon(CommandLine):
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'freesurfer', 'merge_ribbon.sh')
    input_spec = MergeRibbonInputSpec
    output_spec = MergeRibbonOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_gm_mask'] = self.inputs.output_gm_mask
        return outputs
    
# mri_binarize
class MRIBinarizeInputSpec(CommandLineInputSpec):
    input_volume = File(
        exists=True,
        argstr="--i %s",
        desc="Input volume file",
        mandatory=True,
    )

    output_volume = File(
        argstr="--o %s",
        desc="Output volume file",
        mandatory=True,
    )

    match = List( Int(), argstr="--match %s...", desc=( "Match values for binarization (multiple allowed). " "Cannot be used with min/max." ), xor=["min", "max"], )

    min = Float(
        argstr="--min %f",
        desc="Minimum threshold (exclusive with match)",
        xor=["match"],
    )

    max = Float(
        argstr="--max %f",
        desc="Maximum threshold (exclusive with match)",
        xor=["match"],
    )

    args = Str(
        argstr="%s",
        desc="Additional arguments to pass to mri_binarize",
    )


class MRIBinarizeOutputSpec(TraitedSpec):
    output_volume = File(
        exists=True,
        desc="Binarized output volume",
    )


class MRIBinarize(CommandLine):
    _cmd = "mri_binarize"
    input_spec = MRIBinarizeInputSpec
    output_spec = MRIBinarizeOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_volume"] = os.path.abspath(self.inputs.output_volume)
        return outputs