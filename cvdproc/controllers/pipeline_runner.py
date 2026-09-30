"""Run independent subject/session workflows sequentially or in separate processes."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

from cvdproc.bids_data.subject import BIDSSubject
from cvdproc.controllers.pipeline_manager import PipelineManager


def run_pipeline_visit(pipeline_name, subject_id, session_id, bids_dir, output_base, pipeline_config, matlab_path=None):
    """Construct the workflow in its worker; never transfer Nipype objects between processes."""
    visit = f'sub-{subject_id} ses-{session_id}'
    print(f'Creating {pipeline_name} workflow for {visit}', flush=True)
    subject = BIDSSubject(subject_id, bids_dir)
    session = next((item for item in subject.get_all_sessions() if item.session_id == session_id), None)
    if session is None:
        raise ValueError(f'No session {session_id} found for subject {subject_id}')
    output_path = os.path.join(output_base, pipeline_name, f'sub-{subject_id}', f'ses-{session_id}')
    if not any(name in output_path for name in ('t1_register', 'lesion_analysis', 'freesurfer_longitudinal', 'nemo_postprocess', 'freesurfer', 'synthsr')):
        os.makedirs(output_path, exist_ok=True)
    original_directory = os.getcwd()
    original_environment = os.environ.copy()
    try:
        pipeline = PipelineManager().get_pipeline(pipeline_name, subject=subject, session=session, output_path=output_path,
                                                  matlab_path=matlab_path, **pipeline_config)
        workflow = pipeline.create_workflow()
        workflow.base_dir = os.path.join(bids_dir, 'derivatives', 'workflows', f'sub-{subject_id}', f'ses-{session_id}')
        os.makedirs(workflow.base_dir, exist_ok=True)
        workflow.config.setdefault('execution', {})['crashdump_dir'] = workflow.base_dir
        print(f'Running {pipeline_name} for {visit}', flush=True)
        workflow.run()
        print(f'Finished {pipeline_name} for {visit}', flush=True)
    finally:
        os.chdir(original_directory)
        os.environ.clear()
        os.environ.update(original_environment)


def run_pipeline_batch(pipeline_name, visits, bids_dir, output_base, pipeline_config, matlab_path=None, n_jobs=1):
    """Bound visit concurrency; report all parallel failures after submitted visits finish."""
    if n_jobs < 1:
        raise ValueError('n_jobs must be a positive integer')
    visits = list(visits)
    if not visits or any(not subject or not session for subject, session in visits):
        raise ValueError('Each pipeline visit must specify both a subject ID and a session ID')
    if n_jobs > 1:
        if len(visits) != len(set(visits)):
            raise ValueError('Parallel execution requires unique subject/session pairs')
        if pipeline_name.lower() == 'freesurfer_longitudinal' and len({subject for subject, _ in visits}) != len(visits):
            raise ValueError('Parallel freesurfer_longitudinal execution requires distinct subjects')
    bids_dir, output_base = os.path.abspath(bids_dir), os.path.abspath(output_base)
    if n_jobs == 1 or len(visits) == 1:
        for subject, session in visits:
            run_pipeline_visit(pipeline_name, subject, session, bids_dir, output_base, pipeline_config, matlab_path)
        return
    failures = []
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(visits)), mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(run_pipeline_visit, pipeline_name, subject, session, bids_dir, output_base, pipeline_config, matlab_path): (subject, session)
                   for subject, session in visits}
        for future in as_completed(futures):
            subject, session = futures[future]
            try:
                future.result()
            except Exception as exc:
                message = f'sub-{subject} ses-{session}: {type(exc).__name__}: {exc}'
                failures.append(message)
                print(f'FAILED {pipeline_name}: {message}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} of {len(visits)} pipeline visits failed:\n' + '\n'.join(failures))
