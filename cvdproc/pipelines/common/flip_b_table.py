import os
import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    TraitedSpec, BaseInterfaceInputSpec, BaseInterface,
    File, traits, Str
)

class FlipBTableInputSpec(BaseInterfaceInputSpec):
    in_bvec = File(exists=True, mandatory=True, desc='Input bvec file to be flipped')
    out_bvec = Str(mandatory=True, desc='Output flipped bvec file path')
    flip_axis = traits.List(traits.Int, mandatory=True, desc='List of axes to flip (0 for x, 1 for y, 2 for z)')

class FlipBTableOutputSpec(TraitedSpec):
    out_bvec = File(desc='Output flipped bvec file path')

class FlipBTable(BaseInterface):
    """
    A nipype interface to flip bvec file along specified axes.
    """
    input_spec = FlipBTableInputSpec
    output_spec = FlipBTableOutputSpec

    def _run_interface(self, runtime):
        # Load the bvec data
        bvecs = np.loadtxt(self.inputs.in_bvec)

        # Flip the specified axes
        for axis in self.inputs.flip_axis:
            if axis in [0, 1, 2]:
                bvecs[axis, :] = -bvecs[axis, :]

        # Save the flipped bvecs
        np.savetxt(self.inputs.out_bvec, bvecs, fmt='%.8f')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_bvec'] = self.inputs.out_bvec
        return outputs

# example usage:
if __name__ == "__main__":
    flip_btable = FlipBTable()
    flip_btable.inputs.in_bvec = '/mnt/f/BIDS/WCH_AF_Project/derivatives/qsiprep/sub-AFib0241/ses-baseline/dwi/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.bvec'
    flip_btable.inputs.out_bvec = '/mnt/f/BIDS/WCH_AF_Project/derivatives/qsiprep/sub-AFib0241/ses-baseline/dwi/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi_flipped.bvec'
    flip_btable.inputs.flip_axis = [1]  # flip y axes
    flip_btable.run()