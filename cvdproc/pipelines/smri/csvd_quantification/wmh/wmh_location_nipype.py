import os
import subprocess
import shutil
import nibabel as nib
import time
import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str, List

from scipy.ndimage import label, sum

from cvdproc.utils.python.basic_image_processor import extract_roi_from_image, calculate_volume
from cvdproc.bids_data.rename_bids_file import rename_bids_file

from cvdproc.config.paths import get_package_path

'''
Different methods of WMH location quantification
'''

###########
# Fazekas #
###########
class FazekasClassificationInputSpec(BaseInterfaceInputSpec):
    wmh_img = File(mandatory=True, desc="Path to the WMH mask")
    vent_mask = File(mandatory=True, desc="Path to the ventricle mask")
    perivent_mask_3mm = File(mandatory=True, desc="Path to the periventricular mask file (3mm)")
    perivent_mask_10mm = File(mandatory=True, desc="Path to the periventricular mask file (10mm)")

    output_dir = Directory(mandatory=True, desc="Path to the output directory")
    pwmh_mask_filename = Str(desc="Path to the periventricular or confluent WMH mask file")
    dwmh_mask_filename = Str(desc="Path to the deep WMH mask file")

class FazekasClassificationOutputSpec(TraitedSpec):
    pwmh_mask = File(desc="Path to the periventricular or confluent WMH mask file")
    dwmh_mask = File(desc="Path to the deep WMH mask file")

class FazekasClassification(BaseInterface):
    input_spec = FazekasClassificationInputSpec
    output_spec = FazekasClassificationOutputSpec

    def _run_interface(self, runtime):
        wmh_img = self.inputs.wmh_img
        vent_mask = self.inputs.vent_mask
        perivent_mask_3mm = self.inputs.perivent_mask_3mm
        perivent_mask_10mm = self.inputs.perivent_mask_10mm

        output_dir = self.inputs.output_dir
        pwmh_mask_filename = self.inputs.pwmh_mask_filename
        dwmh_mask_filename = self.inputs.dwmh_mask_filename

        vent_mask_data = nib.load(vent_mask).get_fdata()
        vent_mask_data[np.isnan(vent_mask_data)] = 0
        
        mask_3mm_data = nib.load(perivent_mask_3mm).get_fdata()
        mask_3mm_data[np.isnan(mask_3mm_data)] = 0

        mask_10mm_data = nib.load(perivent_mask_10mm).get_fdata()
        mask_10mm_data[np.isnan(mask_10mm_data)] = 0

        wmh_nii = nib.load(wmh_img)
        wmh_data = wmh_nii.get_fdata()
        wmh_data[np.isnan(wmh_data)] = 0

        mask_3mm_data = np.add(mask_3mm_data, vent_mask_data)
        mask_3mm_data[mask_3mm_data > 1] = 1

        mask_10mm_nii = nib.load(perivent_mask_10mm)
        mask_10mm_data = mask_10mm_nii.get_fdata()
        mask_10mm_data[np.isnan(mask_10mm_data)] = 0

        labeled_wmh, num_features = label(wmh_data)

        result_confluent_WMH = np.zeros_like(wmh_data)
        result_periventricular_WMH = np.zeros_like(wmh_data)
        result_deep_WMH = np.zeros_like(wmh_data)
        result_periventricular_or_confluent_WMH = np.zeros_like(wmh_data)

        for region_num in range(1, num_features + 1):
            region = (labeled_wmh == region_num).astype(np.int32)

            in_3mm = np.any(np.logical_and(region, mask_3mm_data))

            out_10mm = np.any(np.logical_and(region, np.logical_not(mask_10mm_data)))

            # confluent WMH: lesions that are partially within the 3mm mask and partially extend outside the 10mm mask
            if in_3mm and out_10mm:
                result_confluent_WMH = np.logical_or(result_confluent_WMH, region)

            # periventricular WMH: lesions that are partially within the 3mm mask and do not extend outside the 10mm mask
            if in_3mm and not out_10mm:
                result_periventricular_WMH = np.logical_or(result_periventricular_WMH, region)

            # deep WMH: lesions that are not within the 3mm mask
            if not in_3mm:
                result_deep_WMH = np.logical_or(result_deep_WMH, region)

            result_periventricular_or_confluent_WMH = np.logical_or(result_confluent_WMH, result_periventricular_WMH)

        result_deep_WMH_nii = nib.Nifti1Image(result_deep_WMH.astype(np.int32), wmh_nii.affine, wmh_nii.header)
        nib.save(result_deep_WMH_nii, os.path.join(output_dir, dwmh_mask_filename))

        result_periventricular_or_confluent_WMH_nii = nib.Nifti1Image(result_periventricular_or_confluent_WMH.astype(np.int32), wmh_nii.affine, wmh_nii.header)
        nib.save(result_periventricular_or_confluent_WMH_nii, os.path.join(output_dir, pwmh_mask_filename))

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['pwmh_mask'] = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.pwmh_mask_filename))
        outputs['dwmh_mask'] = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.dwmh_mask_filename))
        return outputs

############
# Bullseye #
############

class BullseyesInputSpec(CommandLineInputSpec):
    fs_output_dir = Str(mandatory=True, desc="Path to the FreeSurfer output directory", argstr='-s %s')
    fs_subject_id = Str(mandatory=True, desc="FreeSurfer subject ID", argstr='--subjects %s')
    output_dir = Str(mandatory=True, desc="Path to the output directory", argstr='-o %s')
    work_dir = Str(desc="Path to the working directory", argstr='-w %s')
    threads = Int(desc="Number of threads to use", argstr='-p %d', default_value=1)

class BullseyesOutputSpec(TraitedSpec):
    bullseye_wmparc = File(desc="Path to the Bullseye WMParc file")

class Bullseyes(CommandLine):
    input_spec = BullseyesInputSpec
    output_spec = BullseyesOutputSpec
    _cmd = f'python {os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "bullseye_WMH", "run_bullseye_pipeline.py"))}'
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bullseye_wmparc'] = os.path.abspath(os.path.join(self.inputs.output_dir, 'bullseye_wmparc.nii.gz'))

        return outputs

#############
# Bullseye2 #
#############
# https://github.com/JoanBalado/WMH_Bullseye

class Bullseye2InputSpec(CommandLineInputSpec):
    subject_id = Str(mandatory=True, desc="Subject ID", argstr='%s', position=0)
    source_dir = Str(mandatory=True, desc="Path to the source directory", argstr='%s', position=1)
    subjects_dir = Str(mandatory=True, desc="Path to the FreeSurfer subjects directory", argstr='%s', position=2)
    output_dir = Str(mandatory=True, desc="Path to the output directory", argstr='%s', position=3)

class Bullseye2OutputSpec(TraitedSpec):
    bullseye_wmparc = Str(desc="Path to the Bullseye WMParc file")
    bullseye_wmparc_bis = Str(desc="Path to the Bullseye WMParc bis file")
    lobar_wmparc = Str(desc="Path to the Lobar WMParc file")
    lobar_wmparc_bis = Str(desc="Path to the Lobar WMParc bis file")

    file_list = List(Str, desc="List of output files")

class Bullseye2(CommandLine):
    input_spec = Bullseye2InputSpec
    output_spec = Bullseye2OutputSpec
    _cmd = 'bash ' + get_package_path('pipelines', 'external', 'WMH_Bullseye', 'bullseye_pipeline_custom.sh')
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bullseye_wmparc'] = os.path.abspath(os.path.join(self.inputs.output_dir, f'bullseye_parcellation.nii.gz'))
        outputs['bullseye_wmparc_bis'] = os.path.abspath(os.path.join(self.inputs.output_dir, f'bullseye_parcellation_bis.nii.gz'))
        outputs['lobar_wmparc'] = os.path.abspath(os.path.join(self.inputs.output_dir, f'lobar_aseg_masked.nii.gz'))
        outputs['lobar_wmparc_bis'] = os.path.abspath(os.path.join(self.inputs.output_dir, f'lobar_aseg_masked_bis.nii.gz'))

        outputs['file_list'] = [
            outputs['bullseye_wmparc'],
            outputs['bullseye_wmparc_bis'],
            outputs['lobar_wmparc'],
            outputs['lobar_wmparc_bis']
        ]

        return outputs

if __name__ == "__main__":
    bullseye_node = Node(Bullseye2(), name='bullseye_node')
    bullseye_node.inputs.subject_id = 'ses-baseline'
    bullseye_node.inputs.source_dir = '/mnt/e/codes/cvdproc/cvdproc/pipelines/external/WMH_Bullseye'
    bullseye_node.inputs.subjects_dir = '/mnt/f/BIDS/demo_BIDS/derivatives/freesurfer/sub-AFib0241'
    bullseye_node.inputs.output_dir = '/mnt/f/BIDS/demo_BIDS/derivatives/wmh_quantification/sub-AFib0241/ses-baseline/bullseye'

    res = bullseye_node.run()
    print(res.outputs)