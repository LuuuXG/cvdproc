import os
import subprocess
from nipype.interfaces.base import (
    CommandLine, CommandLineInputSpec, BaseInterface, BaseTraitedSpec, BaseInterfaceInputSpec, TraitedSpec,
    File, traits
)

from traits.api import Either, Float

class MRISynthstripInputSpec(CommandLineInputSpec):
    input_file = File(exists=True, desc="Input MRI file", argstr="-i %s")
    output_file = File(desc="Output file path", argstr="-o %s", position=1)
    mask_file = File(desc="Mask file path", argstr="-m %s", position=2)
    distance_transform = traits.Bool(desc="Save distance transform", argstr="-d")
    use_gpu = traits.Bool(desc="Use GPU for processing", argstr="-g")
    border = traits.Int(desc="Border size", argstr="-b %d", default_value=1)
    threads = traits.Int(desc="Number of threads", argstr="-t %d")
    no_csf = traits.Bool(desc="Do not use CSF prior", argstr="--no-csf")
    model = traits.Str(desc="Model file", argstr="--model %s")

class MRISynthstripOutputSpec(TraitedSpec):
    output_file = Either(None, File(exists=True), desc="Output file")
    mask_file = Either(None, File(exists=True), desc="Mask file")
    distance_transform = Either(None, File(exists=True), desc="Distance transform file")

class MRISynthstripCommandLine(CommandLine):
    _cmd = 'mri_synthstrip'
    input_spec = MRISynthstripInputSpec
    output_spec = MRISynthstripOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        # The output file is the destination file
        outputs["output_file"] = os.path.abspath(self.inputs.output_file) if self.inputs.output_file else None
        outputs["mask_file"] = os.path.abspath(self.inputs.mask_file) if self.inputs.mask_file else None
        outputs["distance_transform"] = os.path.abspath(self.inputs.distance_transform) if self.inputs.distance_transform else None

        return outputs

if __name__ == "__main__":
    # Example usage
    synthstrip = MRISynthstripCommandLine()
    synthstrip.inputs.input_file = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/sub-SVD0100_ses-02_acq-tra_FLAIR.nii.gz"
    synthstrip.inputs.output_file = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/FLAIR_brain.nii.gz"
    #synthstrip.inputs.mask_file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/mask.nii.gz"
    synthstrip.inputs.no_csf = False

    result = synthstrip.run()
    print(result.outputs.output_file)