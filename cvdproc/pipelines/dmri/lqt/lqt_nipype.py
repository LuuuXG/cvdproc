from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, Str, traits
from nipype.interfaces.base import CommandLine
import os
import subprocess

class LQTInputSpec(BaseInterfaceInputSpec):
    patient_id = Str(desc="Patient ID", mandatory=True)
    lesion_file = File(exists=True, desc="Lesion file", mandatory=True)
    output_dir = Directory(desc="Output directory", mandatory=True)
    parcel_path = File(exists=True, desc="Path to the parcel file", mandatory=True)
    lqt_script = File(exists=True, desc="Path to the LQT R script template", mandatory=True)
    dsi_path = Str(desc="Path to DSI Studio", mandatory=True)

class LQTOutputSpec(TraitedSpec):
    output_dir = Directory(desc="Directory containing LQT analysis results")

class LQT(BaseInterface):
    input_spec = LQTInputSpec
    output_spec = LQTOutputSpec

    def _run_interface(self, runtime):
        # Read the template R script
        with open(self.inputs.lqt_script, 'r', encoding='utf-8') as file:
            script_content = file.read()

        # Replace placeholders with actual input values
        def normalize(p):
            return os.path.abspath(p).replace('\\', '/')

        script_content = script_content.replace('/this/is/for/nipype/patient_id', self.inputs.patient_id)
        script_content = script_content.replace('/this/is/for/nipype/source_lesion_file', normalize(self.inputs.lesion_file))
        script_content = script_content.replace('/this/is/for/nipype/output_dir', normalize(self.inputs.output_dir))
        script_content = script_content.replace('/this/is/for/nipype/parcel_path', normalize(self.inputs.parcel_path))
        script_content = script_content.replace('/this/is/for/nipype/dsi_path', normalize(self.inputs.dsi_path))

        # Define generated R script path
        generated_script_path = os.path.join(self.inputs.output_dir, 'generated_lqt_analysis.R')

        os.makedirs(self.inputs.output_dir, exist_ok=True)

        # Write the modified R script
        with open(generated_script_path, 'w', encoding='utf-8') as file:
            file.write(script_content)

        # Run the R script
        result = subprocess.run(
            ["Rscript", generated_script_path],
            cwd=self.inputs.output_dir,
        )

        if result.returncode != 0:
            raise RuntimeError("Rscript execution failed. Check console output for details.")

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_dir'] = self.inputs.output_dir
        return outputs

if __name__ == "__main__":
    # Example usage
    lqt = LQT()
    lqt.inputs.patient_id = 'Subject3'
    lqt.inputs.lesion_file = 'E:/Neuroimage/bcbtool_test/input2/ExampleLesion1.nii.gz'
    lqt.inputs.output_dir = 'E:/Neuroimage/bcbtool_test/lqt3'
    lqt.inputs.parcel_path = 'E:/R_packages/LQT/extdata/Schaefer_Yeo_Plus_Subcort/100Parcels7Networks.nii.gz'
    lqt.inputs.lqt_script = 'E:/Codes/cvdproc/cvdproc/pipelines/r/lqt/lqt_single_subject.R'
    
    result = lqt.run()
    print("LQT analysis completed. Output directory:", result.outputs.output_dir)