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
from traits.api import Bool, Int, Str

from scipy.ndimage import label, sum

from cvdproc.utils.python.basic_image_processor import extract_roi_from_image, calculate_volume
from cvdproc.bids_data.rename_bids_file import rename_bids_file

'''
Different methods of WMH location quantification
'''

###########
# Fazekas #
###########

class FazekasInputSpec(BaseInterfaceInputSpec):
    wmh_img = File(mandatory=True, desc="Path to the WMH mask")
    fsl_anat_dir = Str(desc="Path to the FSL ANAT output directory")
    t1_to_flair_xfm = File(desc="Path to the T1 to FLAIR transformation matrix")
    flair_img = File(desc="Path to the FLAIR image")
    # use 'fsl_anat' or 'WMHSynthSeg' or 'SynthSeg'
    use_which_ventmask = Str(mandatory=True, desc="Use which ventricle mask")
    wmh_synthseg = File(desc="Path to the WMH synthseg file")
    use_bianca_mask = Bool(mandatory=True, desc="Use the BIANCA mask")
    output_dir = Directory(mandatory=True, desc="Path to the output directory")
    bianca_mask_filename = Str(desc="Name of the output BIANCA mask file")
    vent_mask_filename = Str(mandatory=True, desc="Name of the output ventricle mask file")
    perivent_mask_3mm_filename = Str(mandatory=True, desc="Name of the output periventricular mask file (3mm)")
    perivent_mask_10mm_filename = Str(mandatory=True, desc="Name of the output periventricular mask file (10mm)")

    wmh_mask_filename = Str(desc="Path to the WMH mask file")
    pwmh_mask_filename = Str(desc="Path to the periventricular or confluent WMH mask file")
    dwmh_mask_filename = Str(desc="Path to the deep WMH mask file")
    # Default: TWMH_vol.csv; PWMH_vol.csv; DWMH_vol.csv
    wmh_mask_vol_filename = Str(desc="Path to the WMH mask volume file", default_value='TWMH_vol.csv')
    pwmh_mask_vol_filename = Str(desc="Path to the periventricular or confluent WMH mask volume file", default_value='PWMH_vol.csv')
    dwmh_mask_vol_filename = Str(desc="Path to the deep WMH mask volume file", default_value='DWMH_vol.csv')

class FazekasOutputSpec(TraitedSpec):
    bianca_mask = Str(desc="Path to the BIANCAMask file")
    vent_mask = File(desc="Path to the ventricle mask file")
    perivent_mask_3mm = File(desc="Path to the periventricular mask file (3mm)")
    perivent_mask_10mm = File(desc="Path to the periventricular mask file (10mm)")

    wmh_mask = File(desc="Path to the WMH mask file") # masked WMH
    pwmh_mask = File(desc="Path to the periventricular or confluent WMH mask file")
    dwmh_mask = File(desc="Path to the deep WMH mask file")
    wmh_mask_vol = File(desc="Path to the WMH mask volume file")
    pwmh_mask_vol = File(desc="Path to the periventricular or confluent WMH mask volume file")
    dwmh_mask_vol = File(desc="Path to the deep WMH mask volume file")

class Fazekas(BaseInterface):
    input_spec = FazekasInputSpec
    output_spec = FazekasOutputSpec

    def _fazekas_classification(self, wmh_img, use_bianca_mask, bianca_mask, vent_mask, perivent_mask_3mm, perivent_mask_10mm,
                                masked_wmh_path, result_pwmh_nii_path, result_dwmh_nii_path):
        wmh_nii = nib.load(wmh_img)
        wmh_data = wmh_nii.get_fdata()
        wmh_data[np.isnan(wmh_data)] = 0

        if use_bianca_mask:
            bianca_mask_nii = nib.load(bianca_mask)
            bianca_mask_data = bianca_mask_nii.get_fdata()

            wmh_data = np.multiply(wmh_data, bianca_mask_data) # Apply the BIANCA mask to the WMH image

            result_masked_WMH_nii = nib.Nifti1Image(wmh_data, wmh_nii.affine, wmh_nii.header)
            nib.save(result_masked_WMH_nii, masked_wmh_path)
        else:
            print("No BIANCA mask provided. Using the original WMH image.")
            # masked WMH is the input WMH image
            masked_wmh_path = wmh_img
            

        mask_3mm_nii = nib.load(perivent_mask_3mm)
        mask_3mm_data = mask_3mm_nii.get_fdata()
        mask_3mm_data[np.isnan(mask_3mm_data)] = 0

        vent_mask_nii = nib.load(vent_mask)
        vent_mask_data = vent_mask_nii.get_fdata()
        vent_mask_data[np.isnan(vent_mask_data)] = 0

        # Here we change he 3mm mask: 3mm mask + vent mask
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
        nib.save(result_deep_WMH_nii, result_dwmh_nii_path)

        result_periventricular_or_confluent_WMH_nii = nib.Nifti1Image(result_periventricular_or_confluent_WMH.astype(np.int32), wmh_nii.affine, wmh_nii.header)
        nib.save(result_periventricular_or_confluent_WMH_nii, result_pwmh_nii_path)

        return masked_wmh_path, result_pwmh_nii_path, result_dwmh_nii_path

    def _run_interface(self, runtime):
        if self.inputs.use_which_ventmask == 'fsl_anat':
            # concat roi2flair.mat
            subprocess.run([
                'convert_xfm',
                '-omat', 'T1_roi2FLAIR.mat',
                '-concat', self.inputs.t1_to_flair_xfm, 'T1_roi2orig.mat'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            # Generate the BIANCA mask
            if os.path.exists(os.path.join(self.inputs.fsl_anat_dir, 'T1_biascorr_bianca_mask.nii.gz')):
                print("BIANCA mask already exists. Skipping BIANCA mask generation.")
            else:
                subprocess.run([
                    'make_bianca_mask',
                    'T1_biascorr',
                    'T1_fast_pve_0',
                    'MNI_to_T1_nonlin_field.nii.gz'
                ], cwd=self.inputs.fsl_anat_dir, check=True)

            self._bianca_mask = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.bianca_mask_filename))

            # fsl_anat
            # generate the dist_to_vent_periventricular mask and transform it to the original space
            subprocess.run([
                'distancemap', '-i', 'T1_biascorr_ventmask.nii.gz', '-o', 'dist_to_vent'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            # 10mm
            subprocess.run([
                'fslmaths', 'dist_to_vent', '-uthr', '10', '-bin', 'dist_to_vent_periventricular_10mm'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            # 3mm
            subprocess.run([
                'fslmaths', 'dist_to_vent', '-uthr', '3', '-bin', 'dist_to_vent_periventricular_3mm'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            subprocess.run([
                'flirt',
                '-in', 'T1_biascorr_bianca_mask',
                '-ref', self.inputs.flair_img,
                '-out', os.path.join(self.inputs.output_dir, self.inputs.bianca_mask_filename),
                '-applyxfm',
                '-init', 'T1_roi2FLAIR.mat',
                '-interp', 'nearestneighbour'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            subprocess.run([
                'flirt',
                '-in', 'T1_biascorr_ventmask',
                '-ref', self.inputs.flair_img,
                '-out', os.path.join(self.inputs.output_dir, self.inputs.vent_mask_filename),
                '-applyxfm',
                '-init',  'T1_roi2FLAIR.mat',
                '-interp', 'nearestneighbour'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            subprocess.run([
                'flirt',
                '-in', 'dist_to_vent_periventricular_10mm',
                '-ref', self.inputs.flair_img,
                '-out', os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_10mm_filename),
                '-applyxfm',
                '-init', 'T1_roi2FLAIR.mat',
                '-interp', 'nearestneighbour'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            subprocess.run([
                'flirt',
                '-in', 'dist_to_vent_periventricular_3mm',
                '-ref', self.inputs.flair_img,
                '-out', os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_3mm_filename),
                '-applyxfm',
                '-init', 'T1_roi2FLAIR.mat',
                '-interp', 'nearestneighbour'
            ], cwd=self.inputs.fsl_anat_dir, check=True)

            self._wmh_mask, self._pwmh_mask, self._dwmh_mask = self._fazekas_classification(
                os.path.join(self.inputs.output_dir, self.inputs.wmh_img),
                self.inputs.use_bianca_mask,
                os.path.join(self.inputs.output_dir, self.inputs.bianca_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.vent_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_3mm_filename),
                os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_10mm_filename),
                os.path.join(self.inputs.output_dir, self.inputs.wmh_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.pwmh_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.dwmh_mask_filename)
            )
        elif self.inputs.use_which_ventmask == 'WMHSynthSeg':
            if self.inputs.use_bianca_mask:
                # raise a warning: cannot use the BIANCA mask with the WMH synthseg ventricle mask
                print("Warning: Cannot use the BIANCA mask with the WMH synthseg ventricle mask. Using the original WMH image.")
                self._bianca_mask = ''

            # Use the ventricle mask from WMH synthseg
            vent_mask = extract_roi_from_image(self.inputs.wmh_synthseg, [4, 43], binarize=True, output_path=os.path.join(self.inputs.output_dir, self.inputs.vent_mask_filename))

            # Generate the periventricular masks
            subprocess.run(["distancemap", "-i", vent_mask, "-o", os.path.join(self.inputs.output_dir, "dist_to_vent_wmhsynthseg")], check=True)
            subprocess.run(["fslmaths", os.path.join(self.inputs.output_dir, "dist_to_vent_wmhsynthseg"), "-uthr", "10", "-bin", os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_10mm_filename)], check=True)
            subprocess.run(["fslmaths", os.path.join(self.inputs.output_dir, "dist_to_vent_wmhsynthseg"), "-uthr", "3", "-bin", os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_3mm_filename)], check=True)
            os.remove(os.path.join(self.inputs.output_dir, "dist_to_vent_wmhsynthseg.nii.gz"))

            self._wmh_mask, self._pwmh_mask, self._dwmh_mask = self._fazekas_classification(
                os.path.join(self.inputs.output_dir, self.inputs.wmh_img),
                False,
                None,
                vent_mask,
                os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_3mm_filename),
                os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_10mm_filename),
                os.path.join(self.inputs.output_dir, self.inputs.wmh_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.pwmh_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.dwmh_mask_filename)
            )
        elif self.inputs.use_which_ventmask == 'SynthSeg':
            if self.inputs.use_bianca_mask:
                print("Warning: Cannot use the BIANCA mask with the SynthSeg ventricle mask. Using the original WMH image.")
                self._bianca_mask = ''

            # prepare input for mri_synthseg
            os.makedirs(os.path.join(self.inputs.output_dir, 'input4SynthSeg'), exist_ok=True)
            # copy the flair image to the input directory
            shutil.copy(self.inputs.flair_img, os.path.join(self.inputs.output_dir, 'input4SynthSeg'))

            subprocess.run(['mri_synthseg',
                                '--i', os.path.join(self.inputs.output_dir, 'input4SynthSeg'),
                                '--o', self.inputs.output_dir,
                                '--parc', '--robust',
                                '--vol', os.path.join(self.inputs.output_dir, 'SynthSegVols.csv'),
                                '--qc', os.path.join(self.inputs.output_dir, 'SynthSegQC.csv'),
                                ])
            
            # remove the input directory
            shutil.rmtree(os.path.join(self.inputs.output_dir, 'input4SynthSeg'))
            
            # filename: basename without .nii or .nii.gz
            flair_filename = os.path.basename(self.inputs.flair_img).split(".")[0]

            entities_SynthSeg = {
                    'space': 'FLAIR',
                }
            synthseg_file = os.path.join(self.inputs.output_dir, rename_bids_file(flair_filename, entities_SynthSeg, 'SynthSeg', '.nii.gz'))
            os.rename(os.path.join(self.inputs.output_dir, f'{flair_filename}_SynthSeg.nii.gz'), synthseg_file)

            entities_SynthSeg_reslice = {
                    'space': 'FLAIR',
                    'desc': 'resliced'
                }
            synthseg_file_reslice = os.path.join(self.inputs.output_dir, rename_bids_file(flair_filename, entities_SynthSeg_reslice, 'SynthSeg', '.nii.gz'))

            # reslice synthseg output to have the save resolution as the wmh mask image
            subprocess.run([
                'flirt',
                '-in', synthseg_file,
                '-ref', self.inputs.wmh_img,
                '-out', synthseg_file_reslice,
                '-applyxfm',
                '-usesqform',
                '-interp', 'nearestneighbour'
            ], check=True)

            vent_mask = extract_roi_from_image(synthseg_file_reslice, [4, 43], binarize=True, output_path=os.path.join(self.inputs.output_dir, self.inputs.vent_mask_filename))

            # Generate the periventricular masks
            subprocess.run(["distancemap", "-i", vent_mask, "-o", os.path.join(self.inputs.output_dir, "dist_to_vent_synthseg")], check=True)
            subprocess.run(["fslmaths", os.path.join(self.inputs.output_dir, "dist_to_vent_synthseg"), "-uthr", "10", "-bin", os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_10mm_filename)], check=True)
            subprocess.run(["fslmaths", os.path.join(self.inputs.output_dir, "dist_to_vent_synthseg"), "-uthr", "3", "-bin", os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_3mm_filename)], check=True)
            os.remove(os.path.join(self.inputs.output_dir, "dist_to_vent_synthseg.nii.gz"))

            self._wmh_mask, self._pwmh_mask, self._dwmh_mask = self._fazekas_classification(
                os.path.join(self.inputs.output_dir, self.inputs.wmh_img),
                False,
                None,
                vent_mask,
                os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_3mm_filename),
                os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_10mm_filename),
                os.path.join(self.inputs.output_dir, self.inputs.wmh_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.pwmh_mask_filename),
                os.path.join(self.inputs.output_dir, self.inputs.dwmh_mask_filename)
            )

        else:
            raise ValueError("Invalid option for ventricle mask. Please use 'fsl_anat', 'wmh_synthseg', or 'synthseg'.")

        # outputs
        self._vent_mask = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.vent_mask_filename))
        self._perivent_mask_3mm = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_3mm_filename))
        self._perivent_mask_10mm = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.perivent_mask_10mm_filename))

        # calculate the volume of the masks
        self._wmh_mask_vol = calculate_volume(self._wmh_mask, os.path.join(self.inputs.output_dir, self.inputs.wmh_mask_vol_filename))
        self._pwmh_mask_vol = calculate_volume(self._pwmh_mask, os.path.join(self.inputs.output_dir, self.inputs.pwmh_mask_vol_filename))
        self._dwmh_mask_vol = calculate_volume(self._dwmh_mask, os.path.join(self.inputs.output_dir, self.inputs.dwmh_mask_vol_filename))

        if not self.inputs.use_bianca_mask:
            self._bianca_mask = ''

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bianca_mask'] = self._bianca_mask
        outputs['vent_mask'] = self._vent_mask
        outputs['perivent_mask_3mm'] = self._perivent_mask_3mm
        outputs['perivent_mask_10mm'] = self._perivent_mask_10mm
        outputs['wmh_mask'] = self._wmh_mask
        outputs['pwmh_mask'] = self._pwmh_mask
        outputs['dwmh_mask'] = self._dwmh_mask
        outputs['wmh_mask_vol'] = self._wmh_mask_vol
        outputs['pwmh_mask_vol'] = self._pwmh_mask_vol
        outputs['dwmh_mask_vol'] = self._dwmh_mask_vol

        return outputs

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

if __name__ == "__main__":
    # bullseyes = Bullseyes()
    # bullseyes.inputs.fs_output_dir = '/mnt/f/BIDS/demo_BIDS/derivatives/freesurfer/sub-TAOHC0261'
    # bullseyes.inputs.fs_subject_id = 'ses-baseline'
    # bullseyes.inputs.output_dir = '/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261/ses-baseline/synthseg'
    # bullseyes.inputs.work_dir = '/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261/ses-baseline/synthseg'
    # bullseyes.inputs.threads = 1
    # res = bullseyes.run()
    from cvdproc.pipelines.external.bullseye_WMH.bullseye_pipeline import create_bullseye_pipeline

    bullseye_wmh_workflow = create_bullseye_pipeline(
        scans_dir='/mnt/f/BIDS/demo_BIDS/derivatives/freesurfer/sub-TAOHC0261',
        work_dir='/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261',
        outputdir='/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261',
        subject_ids=None)
    print("Inputs:\n", bullseye_wmh_workflow.inputs.inputnode)

    test_wf = Workflow(name='test_bullseye_wmh')
    inputnode = Node(IdentityInterface(fields=['fs_subject_id']), name='inputnode')
    inputnode.inputs.fs_subject_id = 'ses-baseline'

    test_wf.connect(inputnode, 'fs_subject_id', bullseye_wmh_workflow, 'inputnode.subject_ids')

    test_wf.base_dir = '/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261/ses-baseline/synthseg'
    res = test_wf.run()