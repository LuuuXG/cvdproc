"""Exercise real process concurrency with lightweight visit tasks and isolated fixtures."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cvdproc.controllers import pipeline_runner as runner


def simulated_visit(pipeline_name, subject_id, session_id, bids_dir, output_base, pipeline_config, matlab_path=None):
    """A picklable test workload: record execution intervals in the selected test directory."""
    root = Path(pipeline_config['test_dir'])
    prefix = f'{subject_id}_{session_id}'
    started = time.monotonic()
    (root / f'{prefix}.started').touch()
    deadline = started + 20
    while len(list(root.glob('*.started'))) < pipeline_config.get('barrier', 0):
        if time.monotonic() > deadline:
            raise TimeoutError('Expected concurrent test task did not start')
        time.sleep(0.01)
    time.sleep(0.15)
    record = {'subject': subject_id, 'session': session_id, 'pid': os.getpid(), 'start': started, 'end': time.monotonic()}
    (root / f'{prefix}.json').write_text(json.dumps(record), encoding='utf-8')
    if subject_id == pipeline_config.get('fail_subject'):
        raise ValueError('Simulated visit failure')


def write_workflow_marker(visit):
    import json
    import os
    from pathlib import Path
    path = Path.cwd() / 'visit.json'
    path.write_text(json.dumps({'visit': visit, 'pid': os.getpid()}), encoding='utf-8')
    return str(path)


def nipype_visit(pipeline_name, subject_id, session_id, bids_dir, output_base, pipeline_config, matlab_path=None):
    from nipype import Node, Workflow
    from nipype.interfaces.utility import Function
    workflow = Workflow(name='parallel_smoke')
    node = Node(Function(input_names=['visit'], output_names=['path'], function=write_workflow_marker), name='marker')
    node.inputs.visit = f'{subject_id}/{session_id}'
    workflow.add_nodes([node])
    manager = Mock()
    manager.get_pipeline.return_value.create_workflow.return_value = workflow
    # Spawned workers import the real runner; only pipeline selection is substituted.
    with patch.object(runner, 'PipelineManager', return_value=manager):
        runner.run_pipeline_visit(pipeline_name, subject_id, session_id, bids_dir, output_base, pipeline_config, matlab_path)


class PipelineRunnerTest(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('CVDPROC_TEST_WORKDIR', tempfile.gettempdir())).resolve()
        self.directory = Path(tempfile.mkdtemp(prefix='pipeline_runner_', dir=base)).resolve()
        self.directory.relative_to(base)

    def run_batch(self, visits, n_jobs=1, **config):
        config['test_dir'] = str(self.directory)
        with patch.object(runner, 'run_pipeline_visit', simulated_visit):
            runner.run_pipeline_batch('test', visits, self.directory, self.directory, config, n_jobs=n_jobs)

    def test_process_overlap_and_concurrency_limit(self):
        self.run_batch([('001', '01'), ('002', '01'), ('003', '02')], n_jobs=2, barrier=2)
        records = [json.loads(path.read_text()) for path in self.directory.glob('*.json')]
        self.assertEqual(len(records), 3)
        self.assertEqual(len({row['pid'] for row in records}), 2)
        self.assertNotIn(os.getpid(), {row['pid'] for row in records})
        events = sorted([(row['start'], 1) for row in records] + [(row['end'], -1) for row in records])
        active = maximum = 0
        for _, change in events:
            active += change
            maximum = max(maximum, active)
        self.assertEqual(maximum, 2)

    def test_default_execution_is_sequential(self):
        self.run_batch([('001', '01'), ('002', '01')])
        records = [json.loads(path.read_text()) for path in sorted(self.directory.glob('*.json'))]
        self.assertEqual({row['pid'] for row in records}, {os.getpid()})
        self.assertLessEqual(records[0]['end'], records[1]['start'])

    def test_real_nipype_workflows_have_separate_work_directories(self):
        visits = [('001', '01'), ('002', '02')]
        for subject, session in visits:
            (self.directory / f'sub-{subject}' / f'ses-{session}').mkdir(parents=True)
        with patch.object(runner, 'run_pipeline_visit', nipype_visit):
            runner.run_pipeline_batch('test', visits, self.directory, self.directory / 'output', {}, n_jobs=2)
        for subject, session in visits:
            marker = self.directory / 'derivatives' / 'workflows' / f'sub-{subject}' / f'ses-{session}' / 'parallel_smoke' / 'marker' / 'visit.json'
            record = json.loads(marker.read_text())
            self.assertEqual(record['visit'], f'{subject}/{session}')
            self.assertNotEqual(record['pid'], os.getpid())

    def test_parallel_failure_reports_identity_and_finishes_other_visits(self):
        with self.assertRaisesRegex(RuntimeError, 'sub-001 ses-01: ValueError: Simulated visit failure'):
            self.run_batch([('001', '01'), ('002', '01')], n_jobs=2, fail_subject='001')
        self.assertTrue((self.directory / '002_01.json').exists())

    def test_sequential_failure_stops_batch(self):
        with self.assertRaisesRegex(ValueError, 'Simulated visit failure'):
            self.run_batch([('001', '01'), ('002', '01')], fail_subject='001')
        self.assertFalse((self.directory / '002_01.started').exists())

    def test_invalid_parallel_batches_do_not_start_workers(self):
        with patch.object(runner, 'ProcessPoolExecutor') as executor:
            for visits, count in [([('001', '01')], 0), ([('001', '01'), ('001', '01')], 2), ([('001', None)], 2), ([], 2)]:
                with self.subTest(visits=visits, count=count), self.assertRaises(ValueError):
                    runner.run_pipeline_batch('test', visits, self.directory, self.directory, {}, n_jobs=count)
            with self.assertRaisesRegex(ValueError, 'distinct subjects'):
                runner.run_pipeline_batch('freesurfer_longitudinal', [('001', '01'), ('001', '02')], self.directory, self.directory, {}, n_jobs=2)
            executor.assert_not_called()

    def test_single_visit_avoids_process_pool(self):
        with patch.object(runner, 'ProcessPoolExecutor') as executor:
            self.run_batch([('001', '01')], n_jobs=4)
            executor.assert_not_called()

    def test_workflow_uses_visit_paths_and_restores_process_state(self):
        (self.directory / 'sub-001' / 'ses-01').mkdir(parents=True)
        workflow = SimpleNamespace(config={}, base_dir=None)
        original_directory = os.getcwd()
        original_value = os.environ.get('CVDPROC_TEST_WORKER_STATE')

        def create_workflow():
            os.environ['CVDPROC_TEST_WORKER_STATE'] = 'changed'
            return workflow

        def execute():
            os.chdir(workflow.base_dir)
            (Path.cwd() / 'completed.txt').write_text('completed', encoding='utf-8')

        workflow.run = execute
        manager = Mock()
        manager.get_pipeline.return_value.create_workflow.side_effect = create_workflow
        with patch.object(runner, 'PipelineManager', return_value=manager):
            runner.run_pipeline_visit('test', '001', '01', str(self.directory), str(self.directory / 'output'), {})
        expected = self.directory / 'derivatives' / 'workflows' / 'sub-001' / 'ses-01'
        self.assertEqual(workflow.base_dir, str(expected))
        self.assertEqual(workflow.config['execution']['crashdump_dir'], str(expected))
        self.assertTrue((expected / 'completed.txt').exists())
        self.assertEqual(os.getcwd(), original_directory)
        self.assertEqual(os.environ.get('CVDPROC_TEST_WORKER_STATE'), original_value)


if __name__ == '__main__':
    unittest.main()
