from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits
from traits.api import Bool, Str
import os, sys
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import applymask, bounding_box, crop

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "external", "DiffusionTensorImaging")))
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "external", "DiffusionTensorImaging", "pymods"))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymods.freewater_runner import FreewaterRunner

class SingleShellFWInputSpec(BaseInterfaceInputSpec):
    fdwi = File(desc='input diffusion weighted image', mandatory=True)
    fbval = File(desc='input bval file', mandatory=True)
    fbvec = File(desc='input bvec file', mandatory=True)
    mask_file = File(desc='input mask file', mandatory=True)
    working_directory = Directory(desc='working directory', mandatory=True)
    output_directory = Directory(desc='output directory', mandatory=True)
    crop_shells = Bool(True, desc='Whether to crop multi-shell DWI to single-shell using lowest b>0')

class SingleShellFWOutputSpec(TraitedSpec):
    output_fw = File(desc='output freewater image')
    output_fw_fa = File(desc='output freewater FA image')
    output_fw_md = File(desc='output freewater MD image')

class SingleShellFW(BaseInterface):
    input_spec = SingleShellFWInputSpec
    output_spec = SingleShellFWOutputSpec

    def _run_interface(self, runtime):
        fdwi_file = self.inputs.fdwi
        fbval_file = self.inputs.fbval
        fbvec_file = self.inputs.fbvec
        working_dir = self.inputs.working_directory

        os.makedirs(working_dir, exist_ok=True)

        img = nib.load(fdwi_file)
        img_data = img.get_fdata()
        affine = img.affine

        bvals, bvecs = read_bvals_bvecs(fbval_file, fbvec_file)
        bvals = np.array(bvals)
        bvecs = np.array(bvecs)

        if self.inputs.crop_shells:
            unique_bvals = np.unique(bvals)
            nonzero_bvals = unique_bvals[unique_bvals > 0]

            # Only crop if more than one non-zero b-value exists
            if len(nonzero_bvals) > 1:
                min_b = nonzero_bvals.min()
                keep_indices = np.where((bvals == 0) | (bvals == min_b))[0]

                print(f"Cropping DWI to b=0 and b={min_b} (total volumes kept: {len(keep_indices)})")

                cropped_data = img_data[..., keep_indices]
                cropped_bvals = bvals[keep_indices]
                cropped_bvecs = bvecs[keep_indices, :]

                cropped_dwi_file = os.path.join(working_dir, 'cropped_dwi.nii.gz')
                cropped_bval_file = os.path.join(working_dir, 'cropped.bval')
                cropped_bvec_file = os.path.join(working_dir, 'cropped.bvec')

                nib.save(nib.Nifti1Image(cropped_data, affine), cropped_dwi_file)
                np.savetxt(cropped_bval_file, cropped_bvals[np.newaxis], fmt='%.1f')
                np.savetxt(cropped_bvec_file, cropped_bvecs.T, fmt='%.6f')

                fdwi_file = cropped_dwi_file
                fbval_file = cropped_bval_file
                fbvec_file = cropped_bvec_file
            else:
                print("B-values already single-shell (b=0 + one non-zero b); skipping cropping.")

        # Reload processed or original files
        img = nib.load(fdwi_file)
        img_data = img.get_fdata()
        mask = nib.load(self.inputs.mask_file).get_fdata()
        bvals, bvecs = read_bvals_bvecs(fbval_file, fbvec_file)
        gtab = gradient_table(bvals, bvecs)

        # Apply mask (no cropping on image space)
        mask_boolean = mask > -1
        mins, maxs = bounding_box(mask_boolean)
        mask_boolean = crop(mask_boolean, mins, maxs)
        cropped_volume = crop(img_data, mins, maxs)
        data = applymask(cropped_volume, mask_boolean)

        fw_runner = FreewaterRunner(data, gtab)
        fw_runner.LOG = True
        fw_runner.run_model(num_iter=100, dt=0.001)

        mask_boolean = mask > 0
        output_directory = self.inputs.output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        fw_data = fw_runner.get_fw_map() * mask_boolean
        fw_md_data = fw_runner.get_fw_md() * mask_boolean
        fw_fa_data = fw_runner.get_fw_fa() * mask_boolean

        fw_file = os.path.join(output_directory, "freewater.nii.gz")
        fw_md_file = os.path.join(output_directory, "freewater_md.nii.gz")
        fw_fa_file = os.path.join(output_directory, "freewater_fa.nii.gz")

        nib.save(nib.Nifti1Image(fw_data, affine), fw_file)
        nib.save(nib.Nifti1Image(fw_md_data, affine), fw_md_file)
        nib.save(nib.Nifti1Image(fw_fa_data, affine), fw_fa_file)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        output_directory = self.inputs.output_directory
        outputs['output_fw'] = os.path.join(output_directory, 'freewater.nii.gz')
        outputs['output_fw_fa'] = os.path.join(output_directory, 'freewater_fa.nii.gz')
        outputs['output_fw_md'] = os.path.join(output_directory, 'freewater_md.nii.gz')
        return outputs

if __name__ == "__main__":
    single_shell_fw = SingleShellFW()

    fdwi = '/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/eddy_corrected_data.nii.gz'
    fmask = '/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/dwi_b0_brain_mask.nii.gz'
    fbval = '/mnt/f/BIDS/SVD_BIDS/sub-SVD0035/ses-02/dwi/sub-SVD0035_ses-02_acq-DTIb1000_dwi.bval'
    fbvec = '/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/eddy_corrected_data.eddy_rotated_bvecs'
    working_directory = '/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/'
    output_directory = '/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/'

    single_shell_fw.inputs.fdwi = fdwi
    single_shell_fw.inputs.fbval = fbval
    single_shell_fw.inputs.fbvec = fbvec
    single_shell_fw.inputs.mask_file = fmask
    single_shell_fw.inputs.working_directory = working_directory
    single_shell_fw.inputs.output_directory = output_directory
    single_shell_fw.inputs.crop_shells = True  # default is True, but explicit here

    single_shell_fw.run()
