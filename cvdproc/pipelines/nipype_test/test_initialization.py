"""Ensure failed scaffolding never proceeds to dataset metadata writes."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cvdproc.pipelines.dcm2bids.dcm2bids_processor import Dcm2BidsProcessor
from cvdproc.pipelines.dcm2bids.scaffold import main as scaffold_main


class InitializationTests(unittest.TestCase):
    def test_update_failure_does_not_stop_scaffold(self):
        from dcm2bids.cli import dcm2bids_scaffold
        def run_scaffold():
            dcm2bids_scaffold.check_latest('dcm2bids')
            return 'completed'
        with patch.object(dcm2bids_scaffold, 'check_latest', side_effect=TimeoutError('network timeout')) as check:
            with patch.object(dcm2bids_scaffold, 'main', side_effect=run_scaffold):
                self.assertEqual(scaffold_main(), 'completed')
            self.assertIs(dcm2bids_scaffold.check_latest, check)

    def test_real_scaffold_failure_propagates(self):
        from dcm2bids.cli import dcm2bids_scaffold
        with patch.object(dcm2bids_scaffold, 'main', side_effect=OSError('cannot write dataset')):
            with self.assertRaises(OSError):
                scaffold_main()

    def test_failed_scaffold_preserves_existing_files_and_stops(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('CVDPROC_TEST_WORKDIR')) as directory:
            root = Path(directory)
            for name in ('participants.tsv', 'participants.json'):
                (root / name).write_text('existing content', encoding='utf-8')
            output = io.StringIO()
            def failed_command(command, **kwargs):
                if kwargs.get('check'):
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 1)
            with patch('cvdproc.pipelines.dcm2bids.dcm2bids_processor.subprocess.run', side_effect=failed_command), contextlib.redirect_stdout(output):
                with self.assertRaises(subprocess.CalledProcessError):
                    Dcm2BidsProcessor(directory).initialize()
            for name in ('participants.tsv', 'participants.json'):
                self.assertEqual((root / name).read_text(encoding='utf-8'), 'existing content')
            self.assertFalse((root / 'code').exists())
            self.assertFalse((root / 'derivatives').exists())
            self.assertFalse((root / 'AGENTS.md').exists())
            self.assertNotIn('Initialization completed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
