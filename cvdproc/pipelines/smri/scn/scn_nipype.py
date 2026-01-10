import os
import subprocess
import nibabel as nib
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory
from traits.api import Bool, Int, Str, List
import pandas as pd
import csv

from cvdproc.config.paths import get_package_path

class MINDComputeInputSpec(BaseInterfaceInputSpec):
    surf_dir = Directory(exists=True, desc='Directory containing surface files', mandatory=True)
    features = List(desc='Comma-separated list of features to use (e.g., "CT,SA,Vol")', mandatory=True)
    parcellation = Str(desc='Parcellation scheme to use (e.g., "aparc")', mandatory=True)
    filter_vertices = Bool(False, desc='Whether to filter out non-biologically feasible vertices', mandatory=False)
    resample = Bool(False, desc='Whether to resample vertices for MIND computation', mandatory=False)
    n_samples = Int(4000, desc='Number of samples for MIND computation if resampling', mandatory=False)
    output_csv = Str(desc='Output CSV file for MIND matrix', mandatory=True)

class MINDComputeOutputSpec(TraitedSpec):
    mind_matrix = File(desc='Output MIND matrix file', exists=True)

class MINDCompute(BaseInterface):
    input_spec = MINDComputeInputSpec
    output_spec = MINDComputeOutputSpec

    def _run_interface(self, runtime):
        surf_dir = self.inputs.surf_dir
        features = self.inputs.features
        parcellation = self.inputs.parcellation
        filter_vertices = self.inputs.filter_vertices
        resample = self.inputs.resample
        n_samples = self.inputs.n_samples
        output_csv = self.inputs.output_csv

        # Import the compute_MIND function from the MIND module
        mind_module_path = get_package_path('pipelines', 'external', 'MIND')
        if mind_module_path not in os.sys.path:
            os.sys.path.insert(0, mind_module_path)
        from MIND import compute_MIND
        mind_matrix = compute_MIND(surf_dir=surf_dir, features=features, parcellation=parcellation,
                                   filter_vertices=filter_vertices, resample=resample, n_samples=n_samples)
        
        # make sure output directory exists
        output_dir = os.path.dirname(output_csv)
        os.makedirs(output_dir, exist_ok=True)

        mind_matrix.to_csv(output_csv)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['mind_matrix'] = self.inputs.output_csv
        return outputs

if __name__ == "__main__":
    # Example usage
    surf_dir = '/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline'
    features = ['CT','MC','Vol','SD','SA']
    parcellation = 'aparc'
    output_csv = '/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/scn/sub-SSI0008/ses-baseline/mind_matrix.csv'

    mind_compute = MINDCompute()
    mind_compute.inputs.surf_dir = surf_dir
    mind_compute.inputs.features = features
    mind_compute.inputs.parcellation = parcellation
    mind_compute.inputs.output_csv = output_csv
    mind_compute.run()