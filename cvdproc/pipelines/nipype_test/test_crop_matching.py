"""Functional robustness checks using isolated synthetic NIfTI files."""
from pathlib import Path
import tempfile
import os
import json
import shutil
import subprocess
from unittest.mock import patch
import unittest

import nibabel as nib
import numpy as np

from cvdproc.pipelines.dcm2bids.crop_matching import load_candidates, match_crop, replace_cropped_images

ROOT = Path(os.environ.get('CVDPROC_TEST_WORKDIR', tempfile.gettempdir())).resolve()


class CropMatchingTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix='cvdproc_crop_', dir=ROOT)).resolve()
        self.directory.relative_to(ROOT)
        self.data = np.arange(12 * 13 * 14, dtype=np.int16).reshape(12, 13, 14)
        self.affine = np.diag([1.2, 1.4, 1.6, 1.0])
        self.offset = np.array([2, 3, 4])
        self.crop_data = self.data[2:9, 3:11, 4:12]
        self.crop_affine = self.affine.copy()
        self.crop_affine[:3, 3] = self.affine[:3, :3] @ self.offset
        self.parent = self.save('parent.nii', self.data, self.affine)
        self.crop = self.save('crop.nii', self.crop_data, self.crop_affine)

    def save(self, name, data, affine, qform=None, slope=None, intercept=None, units='mm'):
        image = nib.Nifti1Image(data, affine)
        image.header.set_xyzt_units(units)
        image.set_sform(affine, code=1)
        image.set_qform(affine if qform is None else qform, code=1)
        if slope is not None:
            image.header.set_slope_inter(slope, intercept)
        path = self.directory / name
        nib.save(image, path)
        return path

    def match(self, crop=None, candidates=None):
        return match_crop(crop or self.crop, load_candidates(candidates if candidates is not None else [self.parent]))

    def test_correct_subvolume(self):
        result = self.match()
        self.assertEqual(result['status'], 'unique')
        self.assertEqual(result['verified_mapping']['offset'], [2, 3, 4])

    def test_single_voxel_difference(self):
        changed = self.crop_data.copy()
        changed[-1, -1, -1] += 1
        crop = self.save('changed.nii', changed, self.crop_affine)
        self.assertEqual(self.match(crop)['status'], 'unmatched')

    def test_same_geometry_wrong_sequence(self):
        wrong = self.save('wrong.nii', self.data + 50, self.affine)
        result = self.match(candidates=[wrong, self.parent])
        self.assertEqual(result['matches'], [str(self.parent)])

    def test_duplicate_parent_is_ambiguous(self):
        duplicate = self.save('duplicate.nii', self.data, self.affine)
        self.assertEqual(self.match(candidates=[self.parent, duplicate])['status'], 'ambiguous')

    def test_fractional_offset_rejected(self):
        affine = self.crop_affine.copy()
        affine[0, 3] += 0.2
        crop = self.save('subvoxel.nii', self.crop_data, affine)
        self.assertEqual(self.match(crop)['status'], 'unmatched')

    def test_wrong_integer_offset_rejected_by_content(self):
        affine = self.crop_affine.copy()
        affine[0, 3] += 1.2
        crop = self.save('wrong_offset.nii', self.crop_data, affine)
        self.assertEqual(self.match(crop)['status'], 'unmatched')

    def test_out_of_bounds_rejected(self):
        affine = self.crop_affine.copy()
        affine[0, 3] += 120
        crop = self.save('outside.nii', self.crop_data, affine)
        self.assertEqual(self.match(crop)['voxel_reads'], 0)

    def test_axis_flip(self):
        transform = np.eye(4)
        transform[0, 0] = -1
        transform[0, 3] = self.crop_data.shape[0] - 1
        crop = self.save('flipped.nii', self.crop_data[::-1], self.crop_affine @ transform)
        self.assertEqual(self.match(crop)['status'], 'unique')

    def test_axis_permutation(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        crop = self.save('permuted.nii', self.crop_data.transpose(2, 0, 1), self.crop_affine @ transform)
        self.assertEqual(self.match(crop)['status'], 'unique')

    def test_four_dimensional_candidate_rejected(self):
        parent = self.save('4d.nii', self.data[..., None], self.affine)
        self.assertEqual(self.match(candidates=[parent])['status'], 'unmatched')

    def test_no_candidate(self):
        self.assertEqual(self.match(candidates=[])['status'], 'unmatched')

    def test_compressed_files(self):
        crop = self.save('crop.nii.gz', self.crop_data, self.crop_affine)
        parent = self.save('parent.nii.gz', self.data, self.affine)
        self.assertEqual(self.match(crop, [parent])['status'], 'unique')

    def test_scaled_values(self):
        parent = self.save('scaled_parent.nii', self.data, self.affine, slope=2, intercept=10)
        crop = self.save('scaled_crop.nii', self.crop_data * 2 + 10, self.crop_affine)
        self.assertEqual(self.match(crop, [parent])['status'], 'unique')

    def test_nonfinite_crop_rejected(self):
        data = self.crop_data.astype(np.float32)
        data[-1, -1, -1] = np.nan
        crop = self.save('nan.nii', data, self.crop_affine)
        self.assertEqual(self.match(crop)['status'], 'invalid_crop')

    def test_constant_crop_rejected(self):
        crop = self.save('constant.nii', np.zeros_like(self.crop_data), self.crop_affine)
        self.assertEqual(self.match(crop)['status'], 'invalid_crop')

    def test_corrupt_candidate_blocks_unique_decision(self):
        broken = self.directory / 'broken.nii'
        broken.write_bytes(b'not a nifti')
        self.assertEqual(self.match(candidates=[broken, self.parent])['status'], 'incomplete')

    def test_unknown_units_rejected(self):
        parent = self.save('unknown.nii', self.data, self.affine, units='unknown')
        self.assertEqual(self.match(candidates=[parent])['status'], 'unmatched')

    def test_geometry_filter_avoids_voxel_reads(self):
        candidates = [self.parent]
        for index in range(20):
            affine = self.affine.copy()
            affine[:3, :3] *= index + 2
            candidates.append(self.save(f'decoy_{index}.nii', self.data, affine))
        result = self.match(candidates=candidates)
        self.assertEqual(result['status'], 'unique')
        self.assertEqual(result['voxel_reads'], 1)

    def replacement_layout(self):
        temporary = self.directory / 'temporary'
        subject = self.directory / 'subject'
        temporary.mkdir()
        subject.mkdir()
        crop = temporary / 'scan_Crop_1.nii'
        parent = subject / 'parent.nii'
        shutil.copy2(self.crop, crop)
        shutil.copy2(self.parent, parent)
        return temporary, subject, crop, parent

    def test_replacement_preserves_bytes_backup_and_json(self):
        temporary, subject, crop, parent = self.replacement_layout()
        crop_bytes, parent_bytes = crop.read_bytes(), parent.read_bytes()
        sidecar = parent.with_suffix('.json')
        sidecar.write_text('{"marker": "unchanged"}')
        rows = replace_cropped_images(temporary, subject)
        self.assertEqual(rows[0]['action'], 'replaced')
        self.assertEqual(parent.read_bytes(), crop_bytes)
        self.assertEqual(Path(rows[0]['backup']).read_bytes(), parent_bytes)
        self.assertEqual(sidecar.read_text(), '{"marker": "unchanged"}')
        self.assertFalse(crop.exists())
        self.assertEqual(json.loads(Path(rows[0]['report']).read_text())['action'], 'replaced')

    def test_replacement_rejects_duplicate_candidates(self):
        temporary, subject, crop, parent = self.replacement_layout()
        original = parent.read_bytes()
        shutil.copy2(parent, subject / 'duplicate.nii')
        rows = replace_cropped_images(temporary, subject)
        self.assertEqual(rows[0]['status'], 'ambiguous')
        self.assertTrue(crop.exists())
        self.assertEqual(parent.read_bytes(), original)

    def test_multiple_crops_do_not_overwrite_same_parent(self):
        temporary, subject, crop, parent = self.replacement_layout()
        original = parent.read_bytes()
        shutil.copy2(crop, temporary / 'another_Crop_2.nii')
        rows = replace_cropped_images(temporary, subject)
        self.assertEqual([row['status'] for row in rows], ['ambiguous_crops', 'ambiguous_crops'])
        self.assertEqual(parent.read_bytes(), original)
        self.assertTrue(crop.exists())

    def test_absent_directories_are_noop(self):
        self.assertEqual(replace_cropped_images(self.directory / 'absent', self.directory), [])

    def test_nonorphan_crop_is_not_replaced(self):
        temporary, subject, crop, parent = self.replacement_layout()
        shutil.copy2(parent, temporary / 'scan.nii')
        self.assertEqual(replace_cropped_images(temporary, subject), [])
        self.assertTrue(crop.exists())

    def test_existing_backup_is_preserved(self):
        temporary, subject, crop, parent = self.replacement_layout()
        backup = temporary / 'crop_backups' / parent.name
        backup.parent.mkdir()
        backup.write_bytes(b'previous backup')
        rows = replace_cropped_images(temporary, subject)
        self.assertEqual(rows[0]['status'], 'backup_exists')
        self.assertEqual(backup.read_bytes(), b'previous backup')
        self.assertTrue(crop.exists())

    def test_replacement_failure_preserves_original(self):
        temporary, subject, crop, parent = self.replacement_layout()
        original = parent.read_bytes()
        with patch('cvdproc.pipelines.dcm2bids.crop_matching.os.replace', side_effect=OSError('simulated failure')):
            with self.assertRaises(OSError):
                replace_cropped_images(temporary, subject)
        self.assertEqual(parent.read_bytes(), original)
        self.assertTrue(crop.exists())
        self.assertEqual((temporary / 'crop_backups' / parent.name).read_bytes(), original)
        report = next(temporary.glob('crop_matching_*.jsonl'))
        self.assertEqual(json.loads(report.read_text())['action'], 'failed')

    def test_invalid_crop_is_reported(self):
        temporary, subject, crop, parent = self.replacement_layout()
        crop.write_bytes(b'broken crop')
        rows = replace_cropped_images(temporary, subject)
        self.assertEqual(rows[0]['status'], 'invalid_crop')
        self.assertTrue(crop.exists())

    def test_compression_mismatch_is_not_replaced(self):
        temporary, subject, crop, parent = self.replacement_layout()
        compressed = temporary / 'different_Crop_1.nii.gz'
        nib.save(nib.load(crop), compressed)
        crop.unlink()
        rows = replace_cropped_images(temporary, subject)
        self.assertEqual(rows[0]['status'], 'compression_mismatch')
        self.assertTrue(compressed.exists())

    def test_failed_conversion_does_not_start_replacement(self):
        from cvdproc.pipelines.dcm2bids.dcm2bids_processor import Dcm2BidsProcessor
        processor = Dcm2BidsProcessor(str(self.directory))
        with patch('cvdproc.pipelines.dcm2bids.dcm2bids_processor.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'dcm2bids')) as command:
            with patch('cvdproc.pipelines.dcm2bids.dcm2bids_processor.replace_cropped_images') as replace:
                with self.assertRaises(subprocess.CalledProcessError):
                    processor.convert('config.json', 'dicom', 'test', '01')
                self.assertTrue(command.call_args.kwargs['check'])
                replace.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
