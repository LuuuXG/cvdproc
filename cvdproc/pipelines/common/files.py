import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits, TraitedSpec

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
                print(f"File {file} does not exist and will be excluded.")

        self._filtered_file_list = filtered_file_list
        self._filtered_filename_list = filtered_filename_list
        return runtime
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["filtered_file_list"] = self._filtered_file_list
        outputs["filtered_filename_list"] = self._filtered_filename_list
        return outputs
    

class MergeFilenameInputSpec(BaseInterfaceInputSpec):
    filename_list = traits.List(
        desc="List of filenames to merge",
        mandatory=True
    )
    dirname = traits.Str(
        desc="Directory name to save the merged file",
        mandatory=True
    )
    prefix = traits.Str(
        desc="Prefix to add to the merged filename",
        mandatory=True
    )
    suffix = traits.Str(
        desc="Suffix to add to the merged filename",
        mandatory=True
    )
    extension = traits.Str(
        desc="Extension to add to the merged filename (e.g., .nii.gz)",
        mandatory=True
    )

class MergeFilenameOutputSpec(TraitedSpec):
    merge_file_list = traits.List(
        desc="List of merged filenames",
    )

class MergeFilename(BaseInterface):
    input_spec = MergeFilenameInputSpec
    output_spec = MergeFilenameOutputSpec

    def _run_interface(self, runtime):
        filename_list = self.inputs.filename_list
        dirname = self.inputs.dirname
        prefix = self.inputs.prefix
        suffix = self.inputs.suffix
        extension = self.inputs.extension

        merge_file_list = []
        for file in filename_list:
            merged_filename = os.path.join(dirname, f"{prefix}{file}{suffix}{extension}")
            merge_file_list.append(merged_filename)

        self._merge_file_list = merge_file_list
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["merge_file_list"] = self._merge_file_list
        return outputs