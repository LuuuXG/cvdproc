import shutil
import os
#from ...bids_data.rename_bids_file import rename_bids_file
from cvdproc.bids_data.rename_bids_file import rename_bids_file
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Dict
from nipype import Node

class CopyToRawDataInputSpec(BaseInterfaceInputSpec):
    in_file = File(desc='input file to be copied')
    reference_file = Str(desc='reference file')
    output_dir = Directory(desc='output directory')
    entities = Dict(desc='entities')
    suffix = Str(desc='suffix')
    extension = Str(desc='extension')

class CopyToRawDataOutputSpec(TraitedSpec):
    out_file = File(desc='output file')

class CopyToRawData(BaseInterface):
    input_spec = CopyToRawDataInputSpec
    output_spec = CopyToRawDataOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        reference_file = self.inputs.reference_file
        output_dir = self.inputs.output_dir
        suffix = self.inputs.suffix
        extension = self.inputs.extension
        entities = self.inputs.entities

        if os.path.exists(reference_file):
            output_dir = os.path.dirname(reference_file)
            new_filename = rename_bids_file(reference_file, entities, suffix, extension)
            shutil.copy2(in_file, os.path.join(output_dir, new_filename))
        else:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Reference file {reference_file} does not exist. Use the output_dir instead.")
            new_filename = rename_bids_file('', entities, suffix, extension)
            shutil.copy2(in_file, os.path.join(output_dir, new_filename))
        
        self._out_file = os.path.join(output_dir, new_filename)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._out_file

        return outputs

if __name__ == '__main__':
    # Test the interface
    copy_to_rawdata_node = Node(CopyToRawData(), name='copy_to_rawdata')
    copy_to_rawdata_node.inputs.in_file = '/mnt/f/BIDS/demo_BIDS/derivatives/sepia_qsm/sub-YCHC0003/ses-01/Sepia_clearswi.nii.gz'
    copy_to_rawdata_node.inputs.reference_file = ''
    copy_to_rawdata_node.inputs.output_dir = '/mnt/f/BIDS/demo_BIDS/sub-YCHC0003/ses-01/swi'
    copy_to_rawdata_node.inputs.entities = {
        'sub': 'YCHC0003',
        'ses': '01',
    }
    copy_to_rawdata_node.inputs.suffix = 'SWI'
    copy_to_rawdata_node.inputs.extension = '.nii.gz'
    copy_to_rawdata_node.run()