import numpy as np
import nibabel as nib
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, Directory, TraitedSpec
from traits.api import Either, File, List, Bool

class MergeGiftiInputSpec(BaseInterfaceInputSpec):
    mesh_gii_file = File(exists=True, mandatory=True, desc='Input GIFTI mesh file')
    data_gii_file = File(exists=True, mandatory=True, desc='Input GIFTI data files to merge')
    output_gii_file = File(mandatory=True, desc='Output GIFTI file after merging')
    overwrite = Bool(True, usedefault=True, desc='Overwrite existing output file if it exists')

class MergeGiftiOutputSpec(TraitedSpec):
    output_gii_file = File(exists=True, desc='Output GIFTI file after merging')

class MergeGifti(BaseInterface):
    """
    Merges GIFTI mesh and data files into a single GIFTI file.
    This interface takes a GIFTI mesh file and a GIFTI data file, merges them,
    and saves the result as a new GIFTI file.
    Example:
        >>> from cvdproc.pipelines.common.merge_gifti import MergeGifti
        >>> node = MergeGifti()
        >>> node.inputs.mesh_gii_file = 'path/to/mesh.gii'
        >>> node.inputs.data_gii_file = 'path/to/data.gii'
        >>> node.inputs.output_gii_file = 'path/to/output.gii'
        >>> node.run()
    Inputs:
        - mesh_gii_file: Input GIFTI mesh file (e.g., surface mesh)
        - data_gii_file: Input GIFTI data file (e.g., vertex values)
        - output_gii_file: Output GIFTI file after merging
        - overwrite: Whether to overwrite the output file if it exists (default: True)
    Outputs:
        - output_gii_file: Output GIFTI file after merging
    """
    input_spec = MergeGiftiInputSpec
    output_spec = MergeGiftiOutputSpec

    def _run_interface(self, runtime):
        gii_mesh = nib.load(self.inputs.mesh_gii_file)
        gii_data = nib.load(self.inputs.data_gii_file)

        vertices = gii_mesh.darrays[0].data
        faces = gii_mesh.darrays[1].data

        values = gii_data.darrays[0].data

        assert values.shape[0] == vertices.shape[0], "Error: Measure and mesh vertex counts do not match!"

        vertex_values = (values > 0).astype(np.float32)

        new_gii = nib.gifti.GiftiImage()
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            vertices, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
        ))
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
        ))
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            vertex_values, intent=nib.nifti1.intent_codes['NIFTI_INTENT_SHAPE']
        ))

        nib.save(new_gii, self.inputs.output_gii_file)

        print(f"Merged GIFTI files saved to {self.inputs.output_gii_file}")
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_gii_file'] = self.inputs.output_gii_file
        return outputs

if __name__ == "__main__":
    import nibabel as nib
    import numpy as np

    gii_file = r'D:\WYJ\Codes\cvdproc\cvdproc\data\standard\fsaverage\lh.thickness.shape.gii'
    gii_img = nib.load(gii_file)
    gii_data = gii_img.darrays[0].data

    zero_indices = np.where(gii_data < 1e-8)[0]