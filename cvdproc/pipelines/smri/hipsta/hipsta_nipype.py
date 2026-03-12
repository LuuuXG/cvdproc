import os
import numpy as np
import nibabel as nib
from scipy import ndimage

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
)
from traits.api import Enum, Str, Bool
import hipsta

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


if __name__ == "__main__":
    hipstanode = Hipsta()
    hipstanode.inputs.filename = "/mnt/e/Neuroimage/workdir/lh.hippoAmygLabels-T1.v22.mgz"
    hipstanode.inputs.hemi = "lh"
    hipstanode.inputs.lut = "freesurfer"
    hipstanode.inputs.outputdir = "/mnt/e/Neuroimage/workdir/hipsta"

    hipstanode.run()