import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
import csv

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

if __name__ == "__main__":
    # Example usage
    tcksample = TckSampleCommand()
    tcksample.inputs.tracks = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/tckgen_output/tracked.tck"
    tcksample.inputs.image = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/dti_FA.nii.gz"
    tcksample.inputs.values = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/tckgen_output/FA.txt"
    tcksample.inputs.stat_tck = "mean"

    # Run the command
    result = tcksample.run()  # This will execute the command