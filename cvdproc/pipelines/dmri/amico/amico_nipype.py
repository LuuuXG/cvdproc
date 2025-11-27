import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str
import amico

class AmicoNoddiInputSpec(BaseInterfaceInputSpec):
    dwi = File(exists=True, desc="Path to the input DWI NIfTI file", mandatory=True)
    bval = File(exists=True, desc="Path to the input b-values file", mandatory=True)
    bvec = File(exists=True, desc="Path to the input b-vectors file", mandatory=True)
    mask = File(exists=True, desc="Path to the brain mask NIfTI file", mandatory=True)
    output_dir = Directory(desc="Directory to save AMICO NODDI results", mandatory=True)

    # if provided, rename files
    direction_filename = Str("fit_dir.nii.gz", desc="Filename for the direction map")
    icvf_filename = Str("fit_NDI.nii.gz", desc="Filename for the ICVF map")
    isovf_filename = Str("fit_FWF.nii.gz", desc="Filename for the ISOVF map")
    od_filename = Str("fit_ODI.nii.gz", desc="Filename for the ODI map")
    modulated_icvf_filename = Str("fit_NDI_modulated.nii.gz", desc="Filename for the modulated ICVF map")
    modulated_od_filename = Str("fit_ODI_modulated.nii.gz", desc="Filename for the modulated ODI map")
    config_filename = Str("config.pickle", desc="Filename for the AMICO config file")

class AmicoNoddiOutputSpec(TraitedSpec):
    direction = Str(desc="Directory where AMICO NODDI results are saved")
    icvf = Str(desc="Path to the ICVF map")
    isovf = Str(desc="Path to the ISOVF map")
    od = Str(desc="Path to the ODI map")
    modulated_icvf = Str(desc="Path to the modulated ICVF map")
    modulated_od = Str(desc="Path to the modulated ODI map")
    config = Str(desc="Path to the AMICO config file")

class AmicoNoddi(BaseInterface):
    input_spec = AmicoNoddiInputSpec
    output_spec = AmicoNoddiOutputSpec

    def _run_interface(self, runtime):
        amico.setup()
        ae = amico.Evaluation()

        # put kernels alongside output_dir (or anywhere you like)
        kernels_dir = '/tmp/amico_kernels'
        if not os.path.exists(kernels_dir):
            os.makedirs(kernels_dir)
        ae.set_config("KERNELS_path", kernels_dir)

        dwi = self.inputs.dwi
        dwi_bval = self.inputs.bval
        dwi_bvec = self.inputs.bvec
        dwi_mask = self.inputs.mask
        output_dir = self.inputs.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        amico.util.fsl2scheme(dwi_bval, dwi_bvec, schemeFilename=os.path.join(output_dir, 'dwi.scheme'))

        ae.load_data(dwi, os.path.join(output_dir, 'dwi.scheme'), mask_filename=dwi_mask, b0_thr=0)
        ae.set_model('NODDI')
        ae.set_config("OUTPUT_path", output_dir)
        ae.set_config("doSaveModulatedMaps", True)
        ae.generate_kernels(regenerate=True)
        ae.load_kernels()
        ae.fit()
        ae.save_results()

        noddi_output_dir = output_dir
        direction_output = os.path.join(noddi_output_dir, 'fit_dir.nii.gz')
        icvf_output = os.path.join(noddi_output_dir, 'fit_NDI.nii.gz')
        isovf_output = os.path.join(noddi_output_dir, 'fit_FWF.nii.gz')
        od_output = os.path.join(noddi_output_dir, 'fit_ODI.nii.gz')
        modulated_icvf_output = os.path.join(noddi_output_dir, 'fit_NDI_modulated.nii.gz')
        modulated_od_output = os.path.join(noddi_output_dir, 'fit_ODI_modulated.nii.gz')
        config_output = os.path.join(noddi_output_dir, 'config.pickle')

        # rename files if specified
        if self.inputs.direction_filename != "fit_dir.nii.gz":
            new_direction_path = os.path.join(noddi_output_dir, self.inputs.direction_filename)
            shutil.move(direction_output, new_direction_path)
            direction_output = new_direction_path
        if self.inputs.icvf_filename != "fit_NDI.nii.gz":
            new_icvf_path = os.path.join(noddi_output_dir, self.inputs.icvf_filename)
            shutil.move(icvf_output, new_icvf_path)
            icvf_output = new_icvf_path
        if self.inputs.isovf_filename != "fit_FWF.nii.gz":
            new_isovf_path = os.path.join(noddi_output_dir, self.inputs.isovf_filename)
            shutil.move(isovf_output, new_isovf_path)
            isovf_output = new_isovf_path
        if self.inputs.od_filename != "fit_ODI.nii.gz":
            new_od_path = os.path.join(noddi_output_dir, self.inputs.od_filename)
            shutil.move(od_output, new_od_path)
            od_output = new_od_path
        if self.inputs.modulated_icvf_filename != "fit_NDI_modulated.nii.gz":
            new_modulated_icvf_path = os.path.join(noddi_output_dir, self.inputs.modulated_icvf_filename)
            shutil.move(modulated_icvf_output, new_modulated_icvf_path)
            modulated_icvf_output = new_modulated_icvf_path
        if self.inputs.modulated_od_filename != "fit_ODI_modulated.nii.gz":
            new_modulated_od_path = os.path.join(noddi_output_dir, self.inputs.modulated_od_filename)
            shutil.move(modulated_od_output, new_modulated_od_path)
            modulated_od_output = new_modulated_od_path
        if self.inputs.config_filename != "config.pickle":
            new_config_path = os.path.join(noddi_output_dir, self.inputs.config_filename)
            shutil.move(config_output, new_config_path)
            config_output = new_config_path

        # clear temp
        if os.path.exists(os.path.join(output_dir, 'dwi.scheme')):
            os.remove(os.path.join(output_dir, 'dwi.scheme'))
        if os.path.exists(kernels_dir):
            shutil.rmtree(kernels_dir)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        output_dir = self.inputs.output_dir
        noddi_output_dir = output_dir

        outputs['direction'] = os.path.join(noddi_output_dir, self.inputs.direction_filename)
        outputs['icvf'] = os.path.join(noddi_output_dir, self.inputs.icvf_filename)
        outputs['isovf'] = os.path.join(noddi_output_dir, self.inputs.isovf_filename)
        outputs['od'] = os.path.join(noddi_output_dir, self.inputs.od_filename)
        outputs['modulated_icvf'] = os.path.join(noddi_output_dir, self.inputs.modulated_icvf_filename)
        outputs['modulated_od'] = os.path.join(noddi_output_dir, self.inputs.modulated_od_filename)
        outputs['config'] = os.path.join(noddi_output_dir, self.inputs.config_filename)

        return outputs

if __name__ == "__main__":
    # Example usage
    noddi_node = AmicoNoddi()
    noddi_node.inputs.dwi = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.nii.gz'
    noddi_node.inputs.bval = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.bval'
    noddi_node.inputs.bvec = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.bvec'
    noddi_node.inputs.mask = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-brain_mask.nii.gz'
    noddi_node.inputs.output_dir = '/mnt/f/BIDS/demo_BIDS/derivatives/amico_noddi/sub-TAOHC0261/ses-baseline'
    noddi_node.inputs.direction_filename = "custom_fit_dir.nii.gz"
    noddi_node.inputs.icvf_filename = "custom_fit_NDI.nii.gz"
    noddi_node.inputs.isovf_filename = "custom_fit_FWF.nii.gz"
    noddi_node.inputs.od_filename = "custom_fit_ODI.nii.gz"
    noddi_node.inputs.modulated_icvf_filename = "custom_fit_NDI_modulated.nii.gz"
    noddi_node.inputs.modulated_od_filename = "custom_fit_ODI_modulated.nii.gz"
    noddi_node.inputs.config_filename = "custom_config.pickle"

    result = noddi_node.run()
    print(result.outputs)