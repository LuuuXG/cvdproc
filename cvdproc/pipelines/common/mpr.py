from nipype.interfaces.base import (
    File,
    TraitedSpec,
    BaseInterfaceInputSpec,
    BaseInterface,
    isdefined,
)
from traits.api import Str, Either, Int
import os


class minIPInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        desc="Input 3D image file",
        mandatory=True,
    )
    axis = Str(
        "z",
        desc="Axis along which to compute the minimum intensity projection. One of {'x','y','z'}.",
        mandatory=True,
    )
    slice_number = Either(
        Int,
        Str("all"),
        desc=(
            "If 'all', compute a global minIP along the axis and collapse it to size 1. "
            "If an integer N, compute a sliding-window minIP with window size N and step 1: "
            "windows are [0..N-1], [1..N], ..., [L-N..L-1], so the output axis length is L-N+1."
        ),
        mandatory=True,
    )
    out_file = Str(
        "",
        desc=(
            "Output file name for the minimum intensity projection (NIfTI). "
            "If empty, will be derived from the input file name."
        ),
        mandatory=False,
    )


class minIPOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc=(
            "Output minimum intensity projection image. "
            "If slice_number == 'all', collapsed axis size is 1; "
            "otherwise the size along the chosen axis is L-N+1."
        ),
    )


class minIP(BaseInterface):
    input_spec = minIPInputSpec
    output_spec = minIPOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        from nipype.utils.filemanip import split_filename

        in_file = self.inputs.in_file

        # Load image
        img = nib.load(in_file)
        data = img.get_fdata()
        affine = img.affine
        header = img.header.copy()

        if data.ndim != 3:
            raise ValueError("Input image must be 3D (x, y, z).")

        # Map axis string to numpy axis index
        axis_str = self.inputs.axis.lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis_str not in axis_map:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

        axis_idx = axis_map[axis_str]
        axis_len = data.shape[axis_idx]

        slice_number = self.inputs.slice_number

        # ----- mode 1: global minIP -----
        if isinstance(slice_number, str):
            if slice_number != "all":
                raise ValueError("slice_number as string must be 'all'.")
            # Collapse chosen axis to size 1
            proj_data = np.min(data, axis=axis_idx, keepdims=True)

        # ----- mode 2: sliding-window minIP -----
        else:
            N = int(slice_number)
            if N <= 0:
                raise ValueError("slice_number must be a positive integer.")

            if N > axis_len:
                raise ValueError(
                    f"slice_number ({N}) cannot be larger than axis length ({axis_len})."
                )

            # Move chosen axis to last position: (..., L)
            moved = np.moveaxis(data, axis_idx, -1)
            L = moved.shape[-1]

            # Output length along that axis: L_out = L - N + 1
            L_out = L - N + 1
            out_shape = moved.shape[:-1] + (L_out,)
            proj = np.zeros(out_shape, dtype=moved.dtype)

            # Sliding window: [k, k+N), k = 0..L-N
            for k in range(L_out):
                block = moved[..., k : k + N]
                proj[..., k] = np.min(block, axis=-1)

            # Move axis back to original position
            proj_data = np.moveaxis(proj, -1, axis_idx)

        # Generate output file name
        if isdefined(self.inputs.out_file) and self.inputs.out_file != "":
            out_file = os.path.abspath(self.inputs.out_file)
        else:
            _, base, _ = split_filename(in_file)
            out_file = os.path.abspath(base + "_minIP.nii.gz")

        # Save result
        proj_img = nib.Nifti1Image(proj_data, affine, header)
        nib.save(proj_img, out_file)

        self._out_file = out_file
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = getattr(self, "_out_file", None)
        return outputs


if __name__ == "__main__":
    minip = minIP()
    minip.inputs.in_file = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/sub-SVD0003/ses-baseline/swi/sub-SVD0003_ses-baseline_swi.nii.gz"
    minip.inputs.axis = "z"
    # Example: sliding-window minIP with thickness 10
    # If original K = 292, output K should be 292 - 10 + 1 = 283
    minip.inputs.slice_number = 10
    minip.inputs.out_file = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/sub-SVD0003/ses-baseline/swi/output_minip.nii.gz"
    res = minip.run()
