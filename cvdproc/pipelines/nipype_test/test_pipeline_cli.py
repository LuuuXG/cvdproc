"""Validate pipeline visit arguments and forwarding of visit-level concurrency."""
import contextlib
import importlib
import io
import unittest
from unittest.mock import patch

cli = importlib.import_module('cvdproc.main')


class PipelineArgumentsTest(unittest.TestCase):
    def arguments(self):
        return ['cvdproc', '--run_pipeline', '--config_file', 'config.yml', '--pipeline', 'test',
                '--subject_id', '001', '002', '--session_id', '01', '02']

    def test_missing_or_mismatched_ids_and_invalid_counts(self):
        cases = [self.arguments()[:-1], self.arguments()[:-3], self.arguments() + ['--n_jobs', '0'],
                 self.arguments() + ['--n_jobs', '-1'], self.arguments() + ['--n_jobs', '1.5']]
        missing_subjects = self.arguments()
        del missing_subjects[6:9]
        cases.append(missing_subjects)
        for arguments in cases:
            with self.subTest(arguments=arguments), patch('sys.argv', arguments):
                with patch.object(cli, 'run_pipeline_batch') as run, patch.object(cli, 'load_config') as load:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as status:
                        cli.main()
                    self.assertEqual(status.exception.code, 2)
                    load.assert_not_called()
                    run.assert_not_called()

    def test_explicit_pairs_and_concurrency_are_forwarded(self):
        config = {'bids_dir': '/bids', 'output_dir': '/outputs', 'pipelines': {'test': {'enabled': True}}}
        for extra, count in [([], 1), (['--n_jobs', '2'], 2)]:
            with self.subTest(count=count), patch('sys.argv', self.arguments() + extra):
                with patch.object(cli, 'load_config', return_value=config), patch.object(cli, 'run_pipeline_batch') as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        cli.main()
                    run.assert_called_once_with('test', [('001', '01'), ('002', '02')], '/bids', '/outputs',
                                                {'enabled': True}, matlab_path=None, n_jobs=count)

    def test_failed_batch_prevents_result_extraction(self):
        config = {'bids_dir': '/bids', 'pipelines': {'test': {'enabled': True}}}
        with patch('sys.argv', self.arguments() + ['--n_jobs', '2', '--extract_results']):
            with patch.object(cli, 'load_config', return_value=config), patch.object(cli, 'PipelineManager') as manager:
                with patch.object(cli, 'run_pipeline_batch', side_effect=RuntimeError('failed visit')):
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'failed visit'):
                        cli.main()
                    manager.assert_not_called()


if __name__ == '__main__':
    unittest.main()
