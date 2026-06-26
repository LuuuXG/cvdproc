import os
import numpy as np
import nibabel as nib

from nibabel.processing import resample_from_to
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    SimpleInterface,
    traits,
)


class FixDSIStudioDWIAndBvecInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    reference_file = File(exists=True, mandatory=True)
    in_bvec = File(exists=True, mandatory=True)
    out_file = File(mandatory=True)
    out_bvec = File(mandatory=True)
    dtype = traits.Str("float32", usedefault=True)


class FixDSIStudioDWIAndBvecOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    out_bvec = File(exists=True)


class FixDSIStudioDWIAndBvec(SimpleInterface):
    input_spec = FixDSIStudioDWIAndBvecInputSpec
    output_spec = FixDSIStudioDWIAndBvecOutputSpec

    @staticmethod
    def _make_iso_reference_grid(ref_img, dsi_img):
        ref_3d = nib.Nifti1Image(
            np.asanyarray(ref_img.dataobj[..., 0]),
            ref_img.affine,
            ref_img.header.copy()
        )

        target_shape = dsi_img.shape[:3]
        target_zooms = np.array(dsi_img.header.get_zooms()[:3], dtype=float)

        ref_affine = ref_img.affine.copy()
        ref_axes = ref_affine[:3, :3]
        axis_norms = np.linalg.norm(ref_axes, axis=0)
        axis_norms[axis_norms == 0] = 1.0
        direction = ref_axes / axis_norms

        target_affine = np.eye(4)
        target_affine[:3, :3] = direction @ np.diag(target_zooms)

        ref_center_vox = (np.array(ref_img.shape[:3], dtype=float) - 1) / 2.0
        ref_center_world = nib.affines.apply_affine(ref_affine, ref_center_vox)

        target_center_vox = (np.array(target_shape, dtype=float) - 1) / 2.0
        target_affine[:3, 3] = ref_center_world - target_affine[:3, :3] @ target_center_vox

        return resample_from_to(ref_3d, (target_shape, target_affine), order=1)

    def _run_interface(self, runtime):
        dsi_img = nib.load(self.inputs.in_file)
        ref_img = nib.load(self.inputs.reference_file)

        if dsi_img.ndim != 4:
            raise ValueError(f"DSI Studio input must be 4D, but got {dsi_img.ndim}D.")
        if ref_img.ndim != 4:
            raise ValueError(f"Reference input must be 4D, but got {ref_img.ndim}D.")
        if dsi_img.shape[3] != ref_img.shape[3]:
            raise ValueError(
                f"Volume mismatch: DSI Studio volumes = {dsi_img.shape[3]}, "
                f"reference volumes = {ref_img.shape[3]}"
            )

        if dsi_img.shape[:3] != ref_img.shape[:3]:
            ref_grid_img = self._make_iso_reference_grid(ref_img, dsi_img)
        else:
            ref_grid_img = nib.Nifti1Image(
                np.asanyarray(ref_img.dataobj[..., 0]),
                ref_img.affine,
                ref_img.header.copy()
            )

        dsi_data = np.asanyarray(dsi_img.dataobj)
        fixed_data = dsi_data[:, ::-1, :, :]

        dtype = np.dtype(self.inputs.dtype)
        fixed_data = fixed_data.astype(dtype, copy=False)

        header = ref_grid_img.header.copy()
        header.set_data_shape(fixed_data.shape)
        header.set_data_dtype(dtype)

        fixed_img = nib.Nifti1Image(
            fixed_data,
            affine=ref_grid_img.affine,
            header=header,
        )

        fixed_img.set_qform(ref_grid_img.affine, code=1)
        fixed_img.set_sform(ref_grid_img.affine, code=1)

        out_file = os.path.abspath(self.inputs.out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        nib.save(fixed_img, out_file)

        bvec = np.loadtxt(self.inputs.in_bvec)

        if bvec.ndim != 2:
            raise ValueError(f"Input bvec must be 2D, but got shape {bvec.shape}.")
        if bvec.shape[0] != 3 and bvec.shape[1] == 3:
            bvec = bvec.T
        if bvec.shape[0] != 3:
            raise ValueError(f"Input bvec must have shape 3 x N or N x 3, but got {bvec.shape}.")
        if bvec.shape[1] != dsi_img.shape[3]:
            raise ValueError(
                f"Number of bvec volumes does not match DWI volumes: "
                f"bvec columns = {bvec.shape[1]}, DWI volumes = {dsi_img.shape[3]}"
            )

        fixed_bvec = np.vstack([
            -bvec[0, :],
            bvec[1, :],
            bvec[2, :],
        ])

        norms = np.linalg.norm(fixed_bvec, axis=0)
        nonzero = norms > 1e-8
        fixed_bvec[:, nonzero] = fixed_bvec[:, nonzero] / norms[nonzero]

        out_bvec = os.path.abspath(self.inputs.out_bvec)
        os.makedirs(os.path.dirname(out_bvec), exist_ok=True)
        np.savetxt(out_bvec, fixed_bvec, fmt="%.10f")

        self._results["out_file"] = out_file
        self._results["out_bvec"] = out_bvec
        return runtime


if __name__ == "__main__":
    from nipype import Node

    subject_id = "TI005"
    session_id = "baseline"
    bids_dir = "/mnt/f/BIDS/Thalamus_glymphatic_BIDS"

    node = Node(FixDSIStudioDWIAndBvec(), name="fix_dsi_dwi_and_bvec")

    node.inputs.in_file = os.path.join(
        bids_dir, "derivatives", "dwi_pipeline", f"sub-{subject_id}", f"ses-{session_id}",
        f"sub-{subject_id}_ses-{session_id}_acq-DSIb4000_dir-AP_space-preprocdwi_desc-preproc_dwi.nii.gz"
    )
    node.inputs.reference_file = os.path.join(
        bids_dir, f"sub-{subject_id}", f"ses-{session_id}", "dwi",
        f"sub-{subject_id}_ses-{session_id}_acq-DSIb4000_dir-AP_dwi.nii.gz"
    )
    node.inputs.in_bvec = os.path.join(
        bids_dir, "derivatives", "dwi_pipeline", f"sub-{subject_id}", f"ses-{session_id}",
        f"sub-{subject_id}_ses-{session_id}_acq-DSIb4000_dir-AP_space-preprocdwi_desc-preproc_dwi.bvec"
    )
    node.inputs.out_file = os.path.join(
        bids_dir, "derivatives", "dwi_pipeline", f"sub-{subject_id}", f"ses-{session_id}",
        f"sub-{subject_id}_ses-{session_id}_acq-DSIb4000_dir-AP_space-preprocdwi_desc-preprocFixed_dwi.nii.gz"
    )
    node.inputs.out_bvec = os.path.join(
        bids_dir, "derivatives", "dwi_pipeline", f"sub-{subject_id}", f"ses-{session_id}",
        f"sub-{subject_id}_ses-{session_id}_acq-DSIb4000_dir-AP_space-preprocdwi_desc-preprocFixed_dwi.bvec"
    )

    node.run()