import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits, TraitedSpec

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