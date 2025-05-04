import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits, TraitedSpec

import logging
logger = logging.getLogger(__name__)

class FilterExistingInputSpec(BaseInterfaceInputSpec):
    input_file_list = traits.List(
        desc="List of input files to filter",
        mandatory=True,
    )

class FilterExistingOutputSpec(TraitedSpec):
    filtered_file_list = traits.List(        
        desc="List of filtered files",
    )
    filtered_filename_list = traits.List(
        desc="List of filtered file basenames",
    )

class FilterExisting(BaseInterface):
    input_spec = FilterExistingInputSpec
    output_spec = FilterExistingOutputSpec

    def _run_interface(self, runtime):
        input_file_list = self.inputs.input_file_list
        filtered_file_list = []
        filtered_filename_list = []

        for file in input_file_list:
            if os.path.exists(file):
                filtered_file_list.append(file)

                # Get the basename of the file
                file_basename = os.path.basename(file)

                if file_basename.endswith(".nii.gz"):
                    filename = os.path.splitext(os.path.splitext(file_basename)[0])[0]
                else:
                    filename = os.path.splitext(file_basename)[0]

                filtered_filename_list.append(filename)

            else:
                logger.warning(f"File {file} does not exist and will be excluded.")

        self._filtered_file_list = filtered_file_list
        self._filtered_filename_list = filtered_filename_list
        return runtime
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["filtered_file_list"] = self._filtered_file_list
        outputs["filtered_filename_list"] = self._filtered_filename_list
        return outputs

if __name__ == "__main__":
    # Example usage
    filter_existing = FilterExisting()
    filter_existing.inputs.input_file_list = [
        "/mnt/f/BIDS/SVD_BIDS/derivatives/wmh_quantification/sub-SVD0040/ses-02/sub-SVD0040_ses-02_acq-tra_space-FLAIR_desc-SynthSeg_VentMask.nii.gz",
        "/mnt/f/BIDS/SVD_BIDS/derivatives/wmh_quantification/sub-SVD0040/ses-02/sub-SVD0040_ses-02_acq-tra_space-FLAIR_desc-SynthSeg_VentMask.nii.gz",
        "/path/to/nonexistent/file2.nii.gz",
        "/path/to/existing/file3.nii.gz"
    ]

    # Run the command
    result = filter_existing.run()  # This will execute the command
    print(result.outputs.filtered_file_list)  # Should only include existing files
    print(result.outputs.filtered_filename_list)  # Should only include existing file basenames