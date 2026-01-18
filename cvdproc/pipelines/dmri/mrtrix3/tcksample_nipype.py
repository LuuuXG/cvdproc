import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
import csv
import numpy as np

class TckSampleInputSpec(CommandLineInputSpec):
    tracks = File(desc="Path to the input tck file", mandatory=True, argstr="%s", position=0)
    image = File(desc="Path to the input image file", mandatory=True, argstr="%s", position=1)
    values = File(desc="Path to the output values file", mandatory=True, argstr="%s", position=2)
    stat_tck = traits.Str(desc="Statistical operation to perform on the sampled values", argstr="-stat_tck %s")
    nointerp = traits.Bool(desc="Do not interpolate the image", argstr="-nointerp")
    precise = traits.Bool(desc="Use precise interpolation", argstr="-precise")
    use_tdi_fraction = traits.Bool(desc="Use TDI fraction", argstr="-use_tdi_fraction")
    args = traits.Str(desc="Additional arguments to pass to tcksample", argstr="%s")

class TckSampleOutputSpec(TraitedSpec):
    values = File(desc="Path to the output values file")

class TckSampleCommand(CommandLine):
    _cmd = "tcksample"
    input_spec = TckSampleInputSpec
    output_spec = TckSampleOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["values"] = os.path.abspath(self.inputs.values)
        return outputs

class CalculateMeanTckSampleInputSpec(BaseInterfaceInputSpec):
    csv_file = File(desc="Path to the input CSV file", mandatory=True, exists=True)
    output_file = File(desc="Path to the output CSV file", mandatory=True)

class CalculateMeanTckSampleOutputSpec(TraitedSpec):
    output_file = File(desc="Path to the output CSV file")

class CalculateMeanTckSample(BaseInterface):
    input_spec = CalculateMeanTckSampleInputSpec
    output_spec = CalculateMeanTckSampleOutputSpec

    def _run_interface(self, runtime):
        # Read second line of the CSV
        with open(self.inputs.csv_file, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError("The input CSV file does not have a second line.")

        # Parse second line as float list
        second_line = lines[1].strip()
        values = [float(x) for x in second_line.split(",")]

        # Compute mean
        mean_value = sum(values) / len(values)

        # Save to output file
        with open(self.inputs.output_file, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["mean"])
            writer.writerow([mean_value])

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_file"] = os.path.abspath(self.inputs.output_file)
        return outputs

class CalculatePointsMeanTckSampleInputSpec(BaseInterfaceInputSpec):
    csv_file = File(
        desc="Input CSV or whitespace-delimited file: rows=streamlines, columns=points",
        mandatory=True,
        exists=True,
    )
    output_file = File(
        desc="Output CSV file (single row: point-wise mean values)",
        mandatory=True,
    )


class CalculatePointsMeanTckSampleOutputSpec(TraitedSpec):
    output_file = File(
        desc="Output CSV file with one row of point-wise means",
        exists=True,
    )


class CalculatePointsMeanTckSample(BaseInterface):
    input_spec = CalculatePointsMeanTckSampleInputSpec
    output_spec = CalculatePointsMeanTckSampleOutputSpec

    def _run_interface(self, runtime):
        in_file = os.path.abspath(self.inputs.csv_file)
        out_file = os.path.abspath(self.inputs.output_file)

        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Load data (auto-detect delimiter)
        try:
            data = np.loadtxt(in_file, delimiter=",")
        except ValueError:
            data = np.loadtxt(in_file)

        if data.ndim != 2:
            raise RuntimeError(
                f"Input file must be 2D (rows=streamlines, cols=points), got shape {data.shape}"
            )

        n_streamlines, n_points = data.shape

        # -------- QC: check consistency --------
        # After loadtxt, inconsistent row lengths would normally already fail.
        # This check is explicit for clarity and future-proofing.
        if not np.all([len(row) == n_points for row in data]):
            raise RuntimeError(
                "Inconsistent number of points across streamlines. "
                "Point-wise averaging is invalid."
            )

        if n_points < 2:
            raise RuntimeError(
                f"Too few points per streamline ({n_points}). Expected >= 2."
            )

        # Column-wise mean
        pointwise_mean = np.nanmean(data, axis=0)

        # Save as ONE ROW
        np.savetxt(
            out_file,
            pointwise_mean.reshape(1, -1),
            delimiter=",",
            fmt="%.6f",
        )

        self._output_file = out_file
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_file"] = getattr(
            self, "_output_file", os.path.abspath(self.inputs.output_file)
        )
        return outputs


if __name__ == "__main__":
    # Example usage
    tcksample = TckSampleCommand()
    tcksample.inputs.tracks = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OR_streamlines.tck"
    tcksample.inputs.image = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/NODDI/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-noddi_param-icvf_dwimap.nii.gz"
    tcksample.inputs.values = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/L_OR_ICVF2.csv"
    tcksample.inputs.args = '-force'

    # Run the command
    result = tcksample.run()  # This will execute the command