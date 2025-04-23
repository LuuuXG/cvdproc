import os
from ..pipelines.smri.csvd_quantification.wmh_pipeline import WMHSegmentationPipeline
from ..pipelines.smri.csvd_quantification.pvs_pipeline import PVSSegmentationPipeline
from ..pipelines.smri.csvd_quantification.cmb_pipeline import CMBSegmentationPipeline
from ..pipelines.smri.freesurfer_pipeline import FreesurferPipeline, FreesurferClinicalPipeline
from ..pipelines.smri.cat12_pipeline import CAT12Pipeline
from ..pipelines.smri.fsl_anat_pipeline import FSLANATPipeline
from ..pipelines.dmri.fdt_pipeline import FDTPipeline
from ..pipelines.nipype_test.test import TestPipeline
from ..pipelines.qmri.sepia_qsm_pipeline import SepiaQSMPipeline

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
            return WMHSegmentationPipeline(subject, session, output_path=output_path, matlab_path=matlab_path, spm_path=spm_path, **kwargs)
        elif pipeline_name.lower() == "pvs_quantification":
            return PVSSegmentationPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "cmb_quantification":
            return CMBSegmentationPipeline(subject, session, output_path=output_path, **kwargs)
        #####################

        #### Structural MRI ####
        elif pipeline_name.lower() == "freesurfer":
            return FreesurferPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "freesurfer_clinical":
            return FreesurferClinicalPipeline(subject, session, output_path=output_path, **kwargs)
        elif pipeline_name.lower() == "cat12":
            return CAT12Pipeline(subject, session, output_path=output_path, matlab_path=matlab_path, **kwargs)
        elif pipeline_name.lower() == "fsl_anat":
            return FSLANATPipeline(subject, session, output_path=output_path, **kwargs)
        ########################

        #### Diffusion MRI ####
        elif pipeline_name.lower() == "fdt":
            return FDTPipeline(subject, session, output_path=output_path, **kwargs)
        
        #### Quantiative MRI ####
        elif pipeline_name.lower() == "sepia_qsm":
            return SepiaQSMPipeline(subject, session, output_path=output_path, **kwargs)
    
        #### TEST ####
        elif pipeline_name.lower() == "test":
            return TestPipeline(subject, session, output_path=output_path, **kwargs)
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
