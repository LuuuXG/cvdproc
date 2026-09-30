"""CLI regression checks for explicit, one-to-one DICOM batch arguments."""
import contextlib
import importlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

cli = importlib.import_module('cvdproc.main')


class Dcm2BidsBatchArgumentsTest(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('CVDPROC_TEST_WORKDIR', tempfile.gettempdir())).resolve()
        self.directory = Path(tempfile.mkdtemp(prefix='cvdproc_cli_', dir=base))
        self.config = self.directory / 'config.yml'
        self.config.write_text(yaml.safe_dump({'bids_dir': str(self.directory), 'dcm2bids': {'config_file': 'conversion.json'}}), encoding='utf-8')
        self.sources = [self.directory / 'sourcedata' / name for name in ('first', 'second')]
        for source in self.sources:
            source.mkdir(parents=True)

    def arguments(self, subjects=('001', '002'), sessions=('01', '02'), option='--dicom_dir', sources=None):
        sources = list(map(str, self.sources)) if sources is None else sources
        return ['cvdproc', '--run_dcm2bids', '--config_file', str(self.config), '--subject_id', *subjects,
                '--session_id', *sessions, option, *sources]

    def reject(self, arguments, message):
        errors = io.StringIO()
        with patch('sys.argv', arguments), patch.object(cli, 'Dcm2BidsProcessor') as processor:
            with patch.object(cli, 'load_config') as load, contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as exit_status:
                    cli.main()
                self.assertEqual(exit_status.exception.code, 2)
                processor.assert_not_called()
                load.assert_not_called()
        self.assertIn(message, errors.getvalue())

    def test_missing_required_arguments(self):
        for option in ('--config_file', '--subject_id', '--session_id'):
            with self.subTest(option=option):
                arguments = self.arguments()
                start = arguments.index(option)
                end = start + 1
                while end < len(arguments) and not arguments[end].startswith('--'):
                    end += 1
                del arguments[start:end]
                self.reject(arguments, f'requires {option}')

    def test_missing_dicom_source(self):
        arguments = self.arguments()
        self.reject(arguments[:arguments.index('--dicom_dir')], 'exactly one')

    def test_both_source_options_rejected(self):
        self.reject(self.arguments() + ['--dicom_subdir', 'first', 'second'], 'exactly one')

    def test_session_count_mismatch(self):
        for sessions in [('01',), ('01', '02', '03')]:
            with self.subTest(sessions=sessions):
                self.reject(self.arguments(sessions=sessions), 'same number of values')

    def test_source_count_mismatch(self):
        for option in ('--dicom_dir', '--dicom_subdir'):
            for sources in [('first',), ('first', 'second', 'third')]:
                with self.subTest(option=option, sources=sources):
                    self.reject(self.arguments(option=option, sources=sources), 'same number of values')

    def test_validation_precedes_initialization(self):
        self.reject(self.arguments(sessions=('01',)) + ['--run_initialization', '--bids_dir', str(self.directory)], 'same number of values')

    def test_valid_batches_keep_explicit_pairing(self):
        for option in ('--dicom_dir', '--dicom_subdir'):
            for sessions in [('01', '02'), ('01', '01')]:
                with self.subTest(option=option, sessions=sessions):
                    sources = [p.name for p in self.sources] if option == '--dicom_subdir' else list(map(str, self.sources))
                    with patch('sys.argv', self.arguments(sessions=sessions, option=option, sources=sources)):
                        with patch.object(cli, 'Dcm2BidsProcessor') as factory, contextlib.redirect_stdout(io.StringIO()):
                            factory.return_value.find_first_dicom.return_value = None
                            cli.main()
                            calls = factory.return_value.convert.call_args_list
                            actual = [(call.kwargs['subject_id'], call.kwargs['session_id'], call.kwargs['dicom_directory']) for call in calls]
                            self.assertEqual(actual, [('001', sessions[0], str(self.sources[0])), ('002', sessions[1], str(self.sources[1]))])


if __name__ == '__main__':
    unittest.main()
