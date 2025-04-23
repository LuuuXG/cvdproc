# as a function
def get_pipeline_workflow(pipeline, subject, session=None, output_path=None, matlab_path=None, spm_path=None, pipeline_config=None):
    from cvdproc.controllers.pipeline_manager import PipelineManager
    manager = PipelineManager()

    pipeline = manager.get_pipeline(
        pipeline,
        subject=subject,
        session=session,
        output_path=output_path,
        matlab_path=matlab_path,
        spm_path=spm_path,
        **pipeline_config
    )

    pipeline_workflow = pipeline.create_workflow()

    return pipeline_workflow