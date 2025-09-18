import os
import nibabel as nib
from nipype.interfaces.base import (
    TraitedSpec, CommandLineInputSpec, CommandLine,
    File, traits, Str, Directory
)

class ChPSegInputSpec(CommandLineInputSpec):
    in_t1 = File(exists=True, mandatory=True, desc='Input T1-weighted image', argstr='%s', position=0)
    output_dir = Str(mandatory=True, desc='Output directory', argstr='%s', position=1)

class ChPSegOutputSpec(TraitedSpec):
    out_chp_mask = File(desc='Output choroid plexus mask (1 = left, 2 = right)')

class ChPSeg(CommandLine):
    """
    A nipype interface to run the ChP segmentation script.
    """
    input_spec = ChPSegInputSpec
    output_spec = ChPSegOutputSpec
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bash', 'chpseg', 'chpseg.sh'))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_chp_mask'] = os.path.join(self.inputs.output_dir, 'T1w', 'T1w_chp_mask.nii.gz')
        return outputs

if __name__ == "__main__":
    # Example usage
    chpseg = ChPSeg()
    chpseg.inputs.in_t1 = '/mnt/e/Neuroimage/chpseg/input/sub-WZMCI001_ses-01_acq-highres_T1w.nii.gz'
    chpseg.inputs.output_dir = '/mnt/e/Neuroimage/chpseg/output'
    chpseg.run()
    print("Choroid plexus mask saved at:", chpseg._list_outputs()['out_chp_mask'])