"""Default output paths must retain the dataset, pipeline, subject, and session."""
import ntpath
import posixpath
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cvdproc.controllers import pipeline_manager


class PipelineManagerPathsTest(unittest.TestCase):
    def setUp(self):
        self.manager = pipeline_manager.PipelineManager()
        self.subject = SimpleNamespace(bids_dir='/bids', subject_id='001')
        self.session = SimpleNamespace(session_id='01')

    def test_default_path_retains_all_components(self):
        for path_module, root in ((posixpath, '/bids'), (ntpath, 'E:\\BIDS')):
            with self.subTest(platform=path_module.__name__):
                subject = SimpleNamespace(bids_dir=root, subject_id='001')
                with patch.object(pipeline_manager, 'os', SimpleNamespace(path=path_module)):
                    actual = self.manager._generate_default_output_path(subject, self.session, 'wmh_quantification')
                expected = path_module.join(root, 'derivatives', 'wmh_quantification', 'sub-001', 'ses-01')
                self.assertEqual(actual, expected)

    def test_subjects_have_distinct_paths(self):
        second = SimpleNamespace(bids_dir='/bids', subject_id='002')
        first_path = self.manager._generate_default_output_path(self.subject, self.session, 'test')
        second_path = self.manager._generate_default_output_path(second, self.session, 'test')
        self.assertNotEqual(first_path, second_path)

    def test_missing_or_empty_session_is_rejected(self):
        for session in (None, SimpleNamespace(session_id=''), SimpleNamespace(session_id=None)):
            with self.subTest(session=session):
                with self.assertRaisesRegex(ValueError, 'session_id is required'):
                    self.manager.get_pipeline('test', self.subject, session=session)

    def test_explicit_output_does_not_generate_a_default(self):
        module = ModuleType('cvdproc.pipelines.nipype_test.test')
        module.TestPipeline = Mock()
        with patch.dict('sys.modules', {module.__name__: module}):
            with patch.object(self.manager, '_generate_default_output_path', side_effect=AssertionError('Unexpected default path')):
                self.manager.get_pipeline('test', self.subject, output_path='/custom/output')
        module.TestPipeline.assert_called_once_with(self.subject, None, output_path='/custom/output')

    def test_population_extraction_does_not_require_a_session(self):
        module = ModuleType('cvdproc.pipelines.nipype_test.test')
        module.TestPipeline = Mock()
        with patch.dict('sys.modules', {module.__name__: module}):
            self.manager.get_pipeline('test', subject=None, session=None, output_path='/population')
        module.TestPipeline.assert_called_once_with(None, None, output_path='/population')


if __name__ == '__main__':
    unittest.main()
