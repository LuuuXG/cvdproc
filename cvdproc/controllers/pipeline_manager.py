import os

class PipelineManager:
    def get_pipeline(self, pipeline_name, subject, session=None, output_path=None, **kwargs):
        """
        获取指定 pipeline 的实例
        :param pipeline_name: str, pipeline 名称
        :param subject: BIDSSubject 对象
        :param session: BIDSSession 对象（可选）
        :param output_path: str, 用户指定的输出路径
        :return: pipeline 对象
        """
        # Default output path: <pipeline_name>/sub-<subject_id>/ses-<session_id>
        if subject is not None:
            default_output_path = self._generate_default_output_path(subject, session, pipeline_name)
            output_path = output_path or default_output_path

        matlab_path = kwargs.pop("matlab_path", None)
        spm_path = kwargs.pop("spm_path", None)

        #### CSVD marker ####
        if pipeline_name.lower() == "wmh_quantification":
            from ..pipelines.smri.csvd_quantification.wmh_pipeline import WMHSegmentationPipeline
            return WMHSegmentationPipeline(subject, session, output_path=output_path, matlab_path=matlab_path, spm_path=spm_path, **kwargs)
        elif pipeline_name.lower() == "pvs_quantification":
            from ..pipelines.smri.csvd_quantification.pvs_pipeline import PVSSegmentationPipeline
            return PVSSegmentationPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "cmb_quantification":
            from ..pipelines.smri.csvd_quantification.cmb_pipeline import CMBSegmentationPipeline
            return CMBSegmentationPipeline(subject, session, output_path=output_path, **kwargs)
        #####################

        #### Structural MRI ####
        elif pipeline_name.lower() == "t1_register":
            from ..pipelines.smri.t1_register import T1RegisterPipeline
            # replace 't1_register' in ouput_path to 'xfm'
            output_path = output_path.replace('t1_register', 'xfm')
            return T1RegisterPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "freesurfer":
            from ..pipelines.smri.freesurfer_pipeline import FreesurferPipeline
            return FreesurferPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "freesurfer_clinical":
            from ..pipelines.smri.freesurfer_pipeline import FreesurferClinicalPipeline
            return FreesurferClinicalPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "synthsr":
            from ..pipelines.smri.freesurfer_pipeline import SynthSRPipeline
            return SynthSRPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "cat12":
            from ..pipelines.smri.cat12_pipeline import CAT12Pipeline
            return CAT12Pipeline(subject, session, output_path=output_path, matlab_path=matlab_path, **kwargs)
        elif pipeline_name.lower() == "fsl_anat":
            from ..pipelines.smri.fsl_anat_pipeline import FSLANATPipeline
            return FSLANATPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "chp_seg":
            from ..pipelines.smri.chpseg_pipeline import ChPSegPipeline
            return ChPSegPipeline(subject, session, output_path=output_path, **kwargs)
        ########################

        #### Diffusion MRI ####
        elif pipeline_name.lower() == "dwi_pipeline":
            from ..pipelines.dmri.dwi_pipeline import DWIPipeline
            return DWIPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "nemo_postprocess":
            from ..pipelines.dmri.nemo_postprocess_pipeline import NemoPostprocessPipeline
            return NemoPostprocessPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "lqt_pipeline":
            from ..pipelines.dmri.lqt_pipeline import LQTPipeline
            return LQTPipeline(subject, session, output_path=output_path, **kwargs)
        
        #### Quantiative MRI ####
        elif pipeline_name.lower() == "sepia_qsm":
            from ..pipelines.qmri.sepia_qsm_pipeline import SepiaQSMPipeline
            return SepiaQSMPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "qsm_pipeline":
            from ..pipelines.qmri.qsm_pipeline import QSMPipeline
            return QSMPipeline(subject, session, output_path=output_path, **kwargs)
        
        #### PWI (DSC-MRI) ####
        elif pipeline_name.lower() == "pwi_pipeline":
            from ..pipelines.pwi.pwi_pipeline import PWIPipeline
            return PWIPipeline(subject, session, output_path=output_path, **kwargs)
    
        #### TEST ####
        elif pipeline_name.lower() == "test":
            from ..pipelines.nipype_test.test import TestPipeline
            return TestPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "test_matlab":
            from ..pipelines.nipype_test.test_matlab import TestMatlabPipeline
            return TestMatlabPipeline(subject, session, output_path=output_path, matlab_path=matlab_path, **kwargs)
        #####################

        raise ValueError(f"Unknown pipeline: {pipeline_name}")


    def _generate_default_output_path(self, subject, session, pipeline_name):
        """
        生成默认输出路径
        :param subject: BIDSSubject 对象
        :param session: BIDSSession 对象
        :param pipeline_name: str, pipeline 名称
        :return: str, 默认输出路径
        """
        session_part = f"/ses-{session.session_id}" if session else ""
        return os.path.join(
            subject.bids_dir,
            "derivatives",
            pipeline_name,
            f"sub-{subject.subject_id}",
            session_part
        )
