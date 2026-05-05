import os
import numpy as np
import nibabel as nib
from scipy import ndimage

from nipype.interfaces.base import (
    BaseInterface,
    CommandLine,
    CommandLineInputSpec,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
)
from traits.api import Enum, Str, Bool, File
import hipsta

from cvdproc.config.paths import get_package_path

class HipstaInputSpec(BaseInterfaceInputSpec):
    filename = File(exists=True, desc="Filename of a segmentation file.", mandatory=True)
    hemi = Enum("lh", "rh", desc="Hemisphere (lh or rh)", mandatory=True)
    lut = Str(desc="<freesurfer|ashs-penn_abc_3t_t2|ashs-umcutrecht_7t|filename>", mandatory=True)
    outputdir = Directory(desc="Output directory", mandatory=True)

class HipstaOutputSpec(TraitedSpec):
    outputdir = Directory(desc="Output directory", exists=True)

class Hipsta(BaseInterface):
    input_spec = HipstaInputSpec
    output_spec = HipstaOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(self.inputs.outputdir, exist_ok=True)

        hipsta.run_hipsta(
            filename=self.inputs.filename,
            hemi=self.inputs.hemi,
            lut=self.inputs.lut,
            outputdir=self.inputs.outputdir,
            long_filter=True,
            gauss_filter_size=[2,50]
        )

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["outputdir"] = self.inputs.outputdir
        return outputs

class HipstaDockerInputSpec(CommandLineInputSpec):
    filename = File(
        exists=True,
        desc="Filename of a segmentation file.",
        mandatory=True,
        argstr="%s",
        position=0,
    )

    hemi = Enum(
        "lh",
        "rh",
        desc="Hemisphere (lh or rh).",
        mandatory=True,
        argstr="%s",
        position=1,
    )

    lut = Str(
        desc="Lookup table: freesurfer, ashs-penn_abc_3t_t2, ashs-umcutrecht_7t, or a custom LUT file.",
        mandatory=True,
        argstr="%s",
        position=2,
    )

    outputdir = Directory(
        desc="Output directory.",
        mandatory=True,
        argstr="%s",
        position=3,
    )


class HipstaDockerOutputSpec(TraitedSpec):
    outputdir = Directory(desc="Output directory containing Hipsta results.")


class HipstaDocker(CommandLine):
    input_spec = HipstaDockerInputSpec
    output_spec = HipstaDockerOutputSpec
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'hipsta', 'hipsta_docker.sh')

    def _run_interface(self, runtime):
        fs_license = os.environ.get("FS_LICENSE", "").strip()
        if not fs_license:
            raise RuntimeError(
                "Environment variable FS_LICENSE is not set. "
                "Please export FS_LICENSE to a valid FreeSurfer license file before running Hipsta."
            )

        if not os.path.isfile(fs_license):
            raise RuntimeError(
                f"FS_LICENSE is set but does not point to an existing file: {fs_license}"
            )

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["outputdir"] = os.path.abspath(self.inputs.outputdir)
        return outputs


if __name__ == "__main__":
    hipstanode = Hipsta()
    hipstanode.inputs.filename = "/mnt/e/Neuroimage/workdir/hipo_test2/rh.hippoAmygLabels-T1.v22.mgz"
    hipstanode.inputs.hemi = "rh"
    hipstanode.inputs.lut = "freesurfer"
    hipstanode.inputs.outputdir = "/mnt/e/Neuroimage/workdir/hipo_test2"

    hipstanode.run()