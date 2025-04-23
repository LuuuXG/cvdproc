# as a function
def run_pipeline_workflow(pipeline_workflow):
    """
    Run a given pipeline workflow.
    :param pipeline_workflow: Workflow, the pipeline workflow to be executed
    :return: str, result of running the pipeline workflow
    """
    pipeline_workflow.run()

    return f"Pipeline workflow {pipeline_workflow.name} executed successfully."