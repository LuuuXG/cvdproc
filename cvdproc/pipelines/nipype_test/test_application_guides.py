"""Check that installing application instructions preserves dataset content."""
import os
from pathlib import Path
import tempfile
import unittest

from cvdproc.guides import install_application_guides


class ApplicationGuidesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get('CVDPROC_TEST_WORKDIR'))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_install_and_repeat_preserves_guidance(self):
        install_application_guides(self.root)
        entry = self.root / 'AGENTS.md'
        guide = self.root / 'code/agent/guides/dcm2bids.md'
        self.assertIn('code/agent/guides/dcm2bids.md', entry.read_text(encoding='utf-8'))
        self.assertIn('## 7. Convert the cohort', guide.read_text(encoding='utf-8'))
        for name in ('pipelines.md', 'extract_results.md'):
            self.assertIn(f'code/agent/guides/{name}', entry.read_text(encoding='utf-8'))
            self.assertTrue((guide.parent / name).is_file())
        entry.write_text('Study rules', encoding='utf-8')
        guide.write_text('Study guide', encoding='utf-8')
        install_application_guides(self.root)
        self.assertEqual(entry.read_text(encoding='utf-8'), 'Study rules')
        self.assertEqual(guide.read_text(encoding='utf-8'), 'Study guide')
        self.assertEqual((self.root / '.bidsignore').read_text(encoding='utf-8'), 'AGENTS.md\n')

    def test_existing_dataset_content_is_unchanged(self):
        paths = ['participants.tsv', 'code/config.yml', 'sub-001/ses-01/anat/example.nii.gz']
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'existing study content')
        (self.root / '.bidsignore').write_text('custom/', encoding='utf-8')
        install_application_guides(self.root)
        for name in paths:
            self.assertEqual((self.root / name).read_bytes(), b'existing study content')
        self.assertEqual((self.root / '.bidsignore').read_text(encoding='utf-8'), 'custom/\nAGENTS.md\n')

    def test_missing_root_is_rejected_without_creation(self):
        with self.assertRaises(ValueError):
            install_application_guides(self.root / 'missing')
        self.assertFalse((self.root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
