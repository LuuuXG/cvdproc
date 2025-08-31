import os
from nipype.interfaces.base import TraitedSpec, File, Directory, BaseInterfaceInputSpec, File, TraitedSpec, BaseInterface, Directory, Str

class QQNetInputSpec(BaseInterfaceInputSpec):
    mag_4d_path = File(desc="Path to the 4D magnitude image", exists=True, mandatory=True)
    qsm_path = File(desc="Path to the QSM image", exists=True, mandatory=True)
    mask_path = File(desc="Path to the brain mask image", exists=True, mandatory=True)
    output_dir = Directory(desc="Output directory for the results", mandatory=True)
    prefix = File(desc="Prefix for the output files", mandatory=True)
    # optional parameters
    r2star_path = File(desc="Path to the R2* image", exists=True, mandatory=False)
    s0_path = File(desc="Path to the S0 image", exists=True, mandatory=False)
    header_path = File(desc="Path to the sepia header file", exists=True, mandatory=False)

class QQNetOutputSpec(TraitedSpec):
    chinb_path = Str(desc="Path to the output CHINB image")
    oef_path = Str(desc="Path to the output OEF image")
    r2_path = Str(desc="Path to the output R2 image")
    s0_path = Str(desc="Path to the output S0 image")
    v_path = Str(desc="Path to the output v image")
    y_path = Str(desc="Path to the output Y image")

class QQNet(BaseInterface):
    input_spec = QQNetInputSpec
    output_spec = QQNetOutputSpec

    def _run_interface(self, runtime):
        from cvdproc.pipelines.external.qqnet.qqnet_predict import qqnet_predict
        inputs = self.inputs
        results_dict = qqnet_predict(
            mag_4d_path=inputs.mag_4d_path,
            qsm_path=inputs.qsm_path,
            mask_path=inputs.mask_path,
            output_dir=inputs.output_dir,
            prefix=inputs.prefix,
            r2star_path=getattr(inputs, 'r2star_path', None),
            s0_path=getattr(inputs, 's0_path', None),
            header_path=getattr(inputs, 'header_path', None)
        )
        self._chinb_path = results_dict.get("chinb")
        self._oef_path = results_dict.get("oef")
        self._r2_path = results_dict.get("r2")
        self._s0_path = results_dict.get("s0")
        self._v_path = results_dict.get("v")
        self._y_path = results_dict.get("y")
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["chinb_path"] = self._chinb_path
        outputs["oef_path"] = self._oef_path
        outputs["r2_path"] = self._r2_path 
        outputs["s0_path"] = self._s0_path
        outputs["v_path"] = self._v_path
        outputs["y_path"] = self._y_path

        return outputs
    
if __name__ == "__main__":
    mag_4d_path = '/mnt/f/BIDS/7T_SVD/derivatives/qsm_pipeline/sub-SVD7TNEW02/ses-01/sub-SVD7TNEW02_ses-01_part-mag_desc-smoothed_GRE.nii.gz'
    qsm_path = '/mnt/f/BIDS/7T_SVD/derivatives/qsm_pipeline/sub-SVD7TNEW02/ses-01/QSM_reconstruction/sub-SVD7TNEW02_ses-01_desc-Chisep_Chimap.nii.gz'
    mask_path = '/mnt/f/BIDS/7T_SVD/derivatives/qsm_pipeline/sub-SVD7TNEW02/ses-01/QSM_reconstruction/sub-SVD7TNEW02_ses-01_mask_QSM.nii.gz'
    output_dir = '/mnt/f/BIDS/7T_SVD/derivatives/qsm_pipeline/sub-SVD7TNEW02/ses-01/qqnet_output'
    prefix = 'sub-SVD7TNEW02_ses-01'

    r2star_path = '/mnt/f/BIDS/7T_SVD/derivatives/qsm_pipeline/sub-SVD7TNEW02/ses-01/sepia_output/sub-SVD7TNEW02_ses-01_R2starmap.nii.gz'
    s0_path = '/mnt/f/BIDS/7T_SVD/derivatives/qsm_pipeline/sub-SVD7TNEW02/ses-01/sepia_output/sub-SVD7TNEW02_ses-01_S0map.nii.gz'
    header_path = '/mnt/f/BIDS/7T_SVD/derivatives/qsm_pipeline/sub-SVD7TNEW02/ses-01/sub-SVD7TNEW02_ses-01_desc-sepia_header.mat'

    qqnet = QQNet()
    qqnet.inputs.mag_4d_path = mag_4d_path
    qqnet.inputs.qsm_path = qsm_path
    qqnet.inputs.mask_path = mask_path
    qqnet.inputs.output_dir = output_dir
    qqnet.inputs.prefix = prefix
    qqnet.inputs.r2star_path = r2star_path
    qqnet.inputs.s0_path = s0_path
    qqnet.inputs.header_path = header_path
    res = qqnet.run()
    #print(res.outputs)