import os
import shutil
import numpy as np
import nibabel as nib

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    TraitedSpec,
    File,
    traits,
)


class ANTsResampleDWIInputSpec(CommandLineInputSpec):
    dimension = traits.Int(4, usedefault=True, position=0, argstr="%d")
    in_file = File(exists=True, mandatory=True, position=1, argstr="%s")
    out_file = File(mandatory=True, position=2, argstr="%s")

    target_spacing = traits.Str("AUTO", usedefault=True, position=3, argstr="%s")

    size_or_spacing = traits.Int(0, usedefault=True, position=4, argstr="%d")
    interpolation = traits.Int(4, usedefault=True, position=5, argstr="%d")
    bspline_order = traits.Int(3, usedefault=True, position=6, argstr="%d")
    pixeltype = traits.Int(6, usedefault=True, position=7, argstr="%d")

    voxel_size = traits.Either(
        traits.Float(),
        traits.Tuple(traits.Float(), traits.Float(), traits.Float()),
        mandatory=True,
    )

    atol = traits.Float(1e-4, usedefault=True)
    force = traits.Bool(False, usedefault=True)


class ANTsResampleDWIOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ANTsResampleDWI(CommandLine):
    input_spec = ANTsResampleDWIInputSpec
    output_spec = ANTsResampleDWIOutputSpec
    _cmd = "ResampleImage"

    def _get_target_zooms(self):
        voxel_size = self.inputs.voxel_size

        if isinstance(voxel_size, (int, float)):
            target_zooms = np.array([float(voxel_size)] * 3, dtype=float)
        else:
            target_zooms = np.array([float(v) for v in voxel_size], dtype=float)

        if target_zooms.shape[0] != 3:
            raise ValueError(f"voxel_size must be a float or a 3-value tuple, but got {voxel_size}.")

        if np.any(target_zooms <= 0):
            raise ValueError(f"voxel_size must be positive, but got {tuple(target_zooms)}.")

        return target_zooms

    def _build_target_spacing(self):
        img = nib.load(self.inputs.in_file)

        if img.ndim != 4:
            raise ValueError(f"Input image must be 4D, but got {img.ndim}D.")

        target_zooms = self._get_target_zooms()
        zooms = img.header.get_zooms()

        if len(zooms) < 4:
            raise ValueError(f"Input image must have 4D zooms, but got {zooms}.")

        dt = float(zooms[3])

        return "x".join([f"{v:g}" for v in target_zooms] + [f"{dt:g}"])

    def _format_arg(self, name, spec, value):
        if name == "target_spacing":
            return spec.argstr % self._build_target_spacing()

        return super()._format_arg(name, spec, value)

    def _run_interface(self, runtime):
        img = nib.load(self.inputs.in_file)

        if img.ndim != 4:
            raise ValueError(f"Input image must be 4D, but got {img.ndim}D.")

        current_zooms = np.array(img.header.get_zooms()[:3], dtype=float)
        target_zooms = self._get_target_zooms()

        out_file = os.path.abspath(self.inputs.out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        if (
            np.allclose(current_zooms, target_zooms, atol=self.inputs.atol)
            and not self.inputs.force
        ):
            if os.path.abspath(self.inputs.in_file) != out_file:
                shutil.copyfile(self.inputs.in_file, out_file)
            return runtime

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs


if __name__ == "__main__":
    from nipype import Node

    node = Node(ANTsResampleDWI(), name="ants_resample_dwi")

    node.inputs.in_file = (
        "/mnt/f/BIDS/Thalamus_glymphatic_BIDS/derivatives/dwi_pipeline/"
        "sub-TI001/ses-6m/"
        "sub-TI001_ses-6m_acq-DSIb4000_dir-AP_space-preprocdwi_desc-preproc_dwi.nii.gz"
    )

    node.inputs.out_file = (
        "/mnt/f/BIDS/Thalamus_glymphatic_BIDS/derivatives/dwi_pipeline/"
        "sub-TI001/ses-6m/"
        "rsub-TI001_ses-6m_acq-DSIb4000_dir-AP_space-preprocdwi_desc-preproc_dwi.nii.gz"
    )

    node.inputs.voxel_size = 2.5
    node.inputs.interpolation = 4
    node.inputs.bspline_order = 3
    node.inputs.pixeltype = 6
    node.inputs.force = False

    print(node.interface.cmdline)

    result = node.run()

    print(result.outputs.out_file)