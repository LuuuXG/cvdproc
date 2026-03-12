import os

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    TraitedSpec,
    File,
    traits,
    isdefined,
)


class PlotTckOnSliceCmdInputSpec(CommandLineInputSpec):
    # Must appear immediately after "python" in the command
    script_path = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="Path to plot_tck_cli.py",
    )

    ref_nii = File(exists=True, mandatory=True, argstr="--ref-nii %s", position=1)
    tck_files = traits.List(File(exists=True), mandatory=True, argstr="--tck %s", position=2, sep=" ")
    out_png = File(mandatory=True, argstr="--out-png %s", position=3)

    # Enum: do NOT set default_value kwarg here (it conflicts in older traits)
    slice_plane = traits.Enum("axial", "sagittal", "coronal", usedefault=True, argstr="--slice-plane %s")
    slice_index = traits.Int(argstr="--slice-index %d")
    slice_opacity = traits.Float(0.60, usedefault=True, argstr="--slice-opacity %f")

    use_fixed_window = traits.Bool(False, usedefault=True, argstr="--use-fixed-window")
    fixed_vmin = traits.Float(0.0, usedefault=True, argstr="--fixed-vmin %f")
    fixed_vmax = traits.Float(0.6, usedefault=True, argstr="--fixed-vmax %f")
    p_low = traits.Float(2.0, usedefault=True, argstr="--p-low %f")
    p_high = traits.Float(99.5, usedefault=True, argstr="--p-high %f")

    max_streamlines_per_tck = traits.Int(4000, usedefault=True, argstr="--max-streamlines-per-tck %d")
    seed = traits.Int(0, usedefault=True, argstr="--seed %d")

    # Tuple defaults are handled in the interface __init__ for maximum compatibility
    streamline_color = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr="--streamline-color %f %f %f")
    streamline_opacity = traits.Float(0.98, usedefault=True, argstr="--streamline-opacity %f")
    streamline_linewidth = traits.Float(2.5, usedefault=True, argstr="--streamline-linewidth %f")

    snapshot_width = traits.Int(1600, usedefault=True, argstr="--snapshot-width %d")
    snapshot_height = traits.Int(1200, usedefault=True, argstr="--snapshot-height %d")
    zoom = traits.Float(1.35, usedefault=True, argstr="--zoom %f")

    force_chroma_key = traits.Bool(False, usedefault=True, argstr="--force-chroma-key")
    chroma_bg = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr="--chroma-bg %f %f %f")
    chroma_tol = traits.Int(10, usedefault=True, argstr="--chroma-tol %d")


class PlotTckOnSliceCmdOutputSpec(TraitedSpec):
    out_png = File(exists=True, desc="Rendered PNG.")


class PlotTckOnSliceCmd(CommandLine):
    input_spec = PlotTckOnSliceCmdInputSpec
    output_spec = PlotTckOnSliceCmdOutputSpec

    _cmd = "xvfb-run -a python"

    def __init__(self, **inputs):
        super().__init__(**inputs)

        # Set robust defaults for tuple traits if user did not provide them
        if not isdefined(self.inputs.streamline_color):
            self.inputs.streamline_color = (0.10, 0.35, 0.85)
        if not isdefined(self.inputs.chroma_bg):
            self.inputs.chroma_bg = (1.0, 0.0, 1.0)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_png"] = os.path.abspath(self.inputs.out_png)
        return outputs


if __name__ == "__main__":
    from nipype import Node

    node = Node(PlotTckOnSliceCmd(), name="plot_or_ot_qc")

    node.inputs.script_path = "/mnt/e/codes/cvdproc/cvdproc/pipelines/dmri/visualization/plot_tck_cli.py"

    node.inputs.ref_nii = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/dtifit/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-tensor_param-fa_dwimap.nii.gz"

    node.inputs.tck_files = [
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OR_streamlines.tck",
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OR_streamlines.tck",
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OT_streamlines.tck",
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OT_streamlines.tck",
    ]

    node.inputs.out_png = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/QC_cmd.png"

    # Optional overrides
    # node.inputs.use_fixed_window = True
    # node.inputs.fixed_vmin = 0.0
    # node.inputs.fixed_vmax = 0.6
    # node.inputs.slice_plane = "axial"
    # node.inputs.slice_opacity = 0.60
    # node.inputs.streamline_color = (0.10, 0.35, 0.85)
    # node.inputs.chroma_bg = (1.0, 0.0, 1.0)

    print(node.interface.cmdline)
    res = node.run()
    print(res.outputs.out_png)