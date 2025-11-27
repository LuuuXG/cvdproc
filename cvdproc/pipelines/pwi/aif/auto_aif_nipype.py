import os
import re
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import savemat
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, Directory, File, InputMultiPath
from neuromaps import transforms
from traits.api import Either, Float

from cvdproc.pipelines.external.AIF_selection_auto.AIF_selection_automatic import AutoAifSel
from cvdproc.pipelines.external.LCB_BNI_USA.dsc import estimate_delta_R2s

class AutoAIFFromPWIInputSpec(BaseInterfaceInputSpec):
    #pwi_path = File(exists=True, mandatory=True, desc='Path to the PWI image file')
    conc_path = File(exists=True, mandatory=True, desc='Path to the PWI Concentration file')
    mask_path = File(exists=True, mandatory=True, desc='Path to the mask image file')
    time_echo = Float(mandatory=True, desc='Echo time in seconds')
    #output_conc = File(mandatory=True, desc='Output path for the concentration image')
    output_aif_vec = File(mandatory=True, desc='Output path for the AIF vector file')
    output_aif_roi = File(mandatory=True, desc='Output path for the AIF ROI file')
    baseline_range = Either(
        (list, tuple), None, mandatory=False, desc='Range of baseline values for AIF selection, e.g., [0, 10]')

class AutoAIFFromPWIOutputSpec(TraitedSpec):
    #output_conc = File(exists=True, desc='Path to the output concentration image')
    output_aif_vec = File(exists=True, desc='Path to the output AIF vector file')
    output_aif_roi = File(exists=True, desc='Path to the output AIF ROI file')

class AutoAIFFromPWI(BaseInterface):
    input_spec = AutoAIFFromPWIInputSpec
    output_spec = AutoAIFFromPWIOutputSpec

    def _run_interface(self, runtime):
        #pwi_path = self.inputs.pwi_path
        conc_path = self.inputs.conc_path
        mask_path = self.inputs.mask_path
        time_echo = self.inputs.time_echo
        #output_conc = self.inputs.output_conc
        output_aif_vec = self.inputs.output_aif_vec
        baseline_range = self.inputs.baseline_range

        plot_config = -2
        plot_suffix = ""
        do_Mouridsen2006 = True

        # # Load PWI image and mask
        # pwi_img = nib.load(pwi_path)
        # pwi_data = pwi_img.get_fdata()
        # mask_data = nib.load(mask_path).get_fdata().astype(bool)

        # # delta R2* estimation
        # delta_r2s_data = np.zeros_like(pwi_data, dtype=np.float32)
        # x, y, z, t = pwi_data.shape

        # for i in range(x):
        #     for j in range(y):
        #         for k in range(z):
        #             if mask_data[i, j, k]:
        #                 s = pwi_data[i, j, k, :]
        #                 baseline = np.mean(s[baseline_range[0]:baseline_range[1]+1])
        #                 if baseline > 0 and np.all(s > 0):
        #                     delta_r2s_data[i, j, k, :] = -1.0 / time_echo * np.log(s / baseline)
        #                 else:
        #                     delta_r2s_data[i, j, k, :] = 0.0

        # delta_r2s_img = nib.Nifti1Image(delta_r2s_data, affine=pwi_img.affine, header=pwi_img.header)
        # nib.save(delta_r2s_img, output_conc)
        # print(f"Delta R2* image saved to {output_conc}")

        conc_img = nib.load(conc_path)

        # Calculate AIF using automatic selection
        AIF_vec, AIF_ROI = AutoAifSel(
            #output_conc,
            conc_path,
            mask_path,
            os.path.dirname(conc_path),
            plot_config,
            plot_suffix,
            do_Mouridsen2006
        ).return_vals()

        # Save AIF vector and ROI
        aif_mask_img = nib.Nifti1Image(AIF_ROI.astype(np.uint8), affine=conc_img.affine, header=conc_img.header)
        nib.save(aif_mask_img, self.inputs.output_aif_roi)
        print(f"AIF ROI saved to {self.inputs.output_aif_roi}")

        savemat(output_aif_vec, {'aif_conc': AIF_vec})
        print(f"AIF saved to: {output_aif_vec}")

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        #outputs['output_conc'] = os.path.abspath(self.inputs.output_conc)
        outputs['output_aif_vec'] = os.path.abspath(self.inputs.output_aif_vec)
        outputs['output_aif_roi'] = os.path.abspath(self.inputs.output_aif_roi)
        return outputs

if __name__ == "__main__":
    from nipype import Node, Workflow
    from nipype.interfaces.io import DataSink

    # Example usage
    pwi_node = Node(AutoAIFFromPWI(), name='auto_aif_from_pwi')
    pwi_node.inputs.pwi_path = '/mnt/e/Neuroimage/TestDataSet/dsc_mri/pwi.nii.gz'
    pwi_node.inputs.mask_path = '/mnt/e/Neuroimage/TestDataSet/dsc_mri/brain_mask.nii.gz'
    pwi_node.inputs.time_echo = 0.032  # Example echo time in seconds
    pwi_node.inputs.output_conc = '/mnt/e/Neuroimage/TestDataSet/dsc_mri/pwi_conc.nii.gz'
    pwi_node.inputs.output_aif_vec = '/mnt/e/Neuroimage/TestDataSet/dsc_mri/aif_vector.mat'
    pwi_node.inputs.output_aif_roi = '/mnt/e/Neuroimage/TestDataSet/dsc_mri/aif_mask.nii.gz'
    pwi_node.inputs.baseline_range = [0, 15]  # Example baseline range

    workflow = Workflow(name='auto_aif_workflow')
    workflow.add_nodes([pwi_node])
    workflow.run()