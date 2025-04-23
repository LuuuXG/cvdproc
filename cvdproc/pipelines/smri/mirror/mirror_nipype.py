import os
import subprocess
import nibabel as nib
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory
from traits.api import Bool, Int, Str

class MirrorMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='Input mask file to be mirrored (in T1w space)')
    t1w_file = File(mandatory=True, desc='T1w image file for reference')
    out_dir = Directory(mandatory=True, desc='Output directory for the mirrored mask')
    fsl_anat_output_dir = Directory(desc='Output directory of fsl_anat')
    t1w_to_mni_xfm = File(desc='Transformation matrix from T1w to MNI space')
    mni_to_t1w_xfm = File(desc='Transformation matrix from MNI to T1w space')

    mask_in_mni_filename = Str(desc='Filename for the mask in MNI space')
    flipped_mask_mni_filename = Str(desc='Filename for the flipped mask in MNI space')
    flipped_mask_t1w_filename = Str(desc='Filename for the flipped mask in T1w space')

class MirrorMaskOutputSpec(TraitedSpec):
    mask_in_mni = File(exists=True, desc='Output mask in MNI space')
    flipped_mask_mni = File(exists=True, desc='Flipped mask in MNI space')
    flipped_mask_t1w = File(exists=True, desc='Flipped mask in T1w space')

class MirrorMask(BaseInterface):
    input_spec = MirrorMaskInputSpec
    output_spec = MirrorMaskOutputSpec

    def _run_interface(self, runtime):
        # Get inputs
        in_file = self.inputs.in_file
        t1w_file = self.inputs.t1w_file
        out_dir = self.inputs.out_dir

        if os.path.exists(self.inputs.fsl_anat_output_dir):
            t1w_to_mni_xfm = os.path.join(self.inputs.fsl_anat_output_dir, 'T1_to_MNI_nonlin_field.nii.gz')
            mni_to_t1w_xfm = os.path.join(self.inputs.fsl_anat_output_dir, 'MNI_to_T1_nonlin_field.nii.gz')
        else:
            t1w_to_mni_xfm = self.inputs.t1w_to_mni_xfm
            mni_to_t1w_xfm = self.inputs.mni_to_t1w_xfm

        # Step 1: Load input mask file (in T1w space)
        # in_img = nib.load(in_file)
        # in_data = in_img.get_fdata()

        # Step 2: Register input mask to MNI space using non-linear warp (applywarp)
        mask_in_mni = os.path.join(out_dir, self.inputs.mask_in_mni_filename)
        # use MNI in FSL_DIR as reference
        mni_ref = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_1mm_brain.nii.gz')
        self._applywarp(in_file, mask_in_mni, t1w_to_mni_xfm, ref_image=mni_ref)

        # Step 3: Load the mask in MNI space and perform left-right flip (mirror)
        mask_mni_img = nib.load(mask_in_mni)
        mask_mni_data = mask_mni_img.get_fdata()
        flipped_mask_mni_data = np.flip(mask_mni_data, axis=0)  # Mirror along the x-axis

        # Step 4: Save the flipped mask in MNI space
        flipped_mask_mni_img = nib.Nifti1Image(flipped_mask_mni_data.astype(np.float32), mask_mni_img.affine, mask_mni_img.header)
        flipped_mask_mni_filename = os.path.join(out_dir, self.inputs.flipped_mask_mni_filename)
        nib.save(flipped_mask_mni_img, flipped_mask_mni_filename)

        # Step 5: Transform the mirrored mask back to T1w space using the non-linear warp (applywarp)
        flipped_mask_t1w_filename = os.path.join(out_dir, self.inputs.flipped_mask_t1w_filename)
        self._applywarp(flipped_mask_mni_filename, flipped_mask_t1w_filename, mni_to_t1w_xfm, ref_image=t1w_file)

        return runtime

    def _applywarp(self, input_file, output_file, transform_file, ref_image=None):
        """
        Apply the given non-linear warp to the input mask file and save the result to output_file.
        Uses FSL's applywarp for non-linear transformations.
        """
        if ref_image is None:
            ref_image = self.inputs.t1w_file  # Default reference is T1w image

        applywarp_cmd = [
            'applywarp',
            f'--in={input_file}',
            f'--ref={ref_image}',  # Use the T1w image as the reference
            f'--out={output_file}',
            f'--warp={transform_file}',
            '--interp=nn',  # Nearest neighbor interpolation for binary mask
        ]
        subprocess.run(applywarp_cmd, check=True)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['mask_in_mni'] = os.path.join(self.inputs.out_dir, self.inputs.mask_in_mni_filename)
        outputs['flipped_mask_mni'] = os.path.join(self.inputs.out_dir, self.inputs.flipped_mask_mni_filename)
        outputs['flipped_mask_t1w'] = os.path.join(self.inputs.out_dir, self.inputs.flipped_mask_t1w_filename)
        return outputs
