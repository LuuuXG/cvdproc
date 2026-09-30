"""Exact NIfTI crop matching and byte-preserving replacement without JSON sidecars."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import time
import json
import logging
import os
import re
import shutil
import tempfile
from collections import Counter

import nibabel as nib
import numpy as np


@dataclass
class Candidate:
    path: Path
    image: object = None
    error: str = ''


def load_candidates(paths):
    candidates = []
    for path in sorted(map(Path, paths)):
        try:
            candidates.append(Candidate(path, nib.load(str(path))))
        except Exception as exc:
            candidates.append(Candidate(path, error=f'header_error: {type(exc).__name__}'))
    return candidates


def geometry_mapping(crop, parent):
    crop_affine = crop.affine
    if crop.ndim != 3 or parent.ndim != 3:
        return None, 'requires_3d'
    if crop.header.get_xyzt_units()[0] != parent.header.get_xyzt_units()[0]:
        return None, 'spatial_units_differ'
    if not np.isfinite(crop_affine).all() or not np.isfinite(parent.affine).all():
        return None, 'invalid_affine'
    try:
        transform = np.linalg.solve(parent.affine, crop_affine)
    except np.linalg.LinAlgError:
        return None, 'singular_affine'
    linear = np.rint(transform[:3, :3]).astype(int)
    if not (np.allclose(transform[:3, :3], linear, rtol=0, atol=1e-5)
            and np.all(np.abs(linear).sum(axis=0) == 1)
            and np.all(np.abs(linear).sum(axis=1) == 1)):
        return None, 'not_signed_axis_permutation'
    offset = np.rint(transform[:3, 3]).astype(int)
    if not np.allclose(transform[:3, 3], offset, rtol=0, atol=1e-3):
        return None, 'noninteger_voxel_offset'
    corners = np.array(list(product(*[(0, n - 1) for n in crop.shape])), dtype=float)
    parent_corners = corners @ linear.T + offset
    low = parent_corners.min(axis=0).astype(int)
    high = parent_corners.max(axis=0).astype(int)
    if np.any(low < 0) or np.any(high >= np.asarray(parent.shape)):
        return None, 'crop_outside_parent'
    crop_world = nib.affines.apply_affine(crop_affine, corners)
    parent_world = nib.affines.apply_affine(parent.affine, parent_corners)
    error_mm = float(np.max(np.linalg.norm(crop_world - parent_world, axis=1)))
    units = crop.header.get_xyzt_units()[0]
    if units != 'mm':
        return None, 'requires_explicit_mm_units'
    if error_mm > 0.01:
        return None, 'world_coordinate_error'
    axes = np.argmax(np.abs(linear), axis=0)
    flips = [axis for axis, parent_axis in enumerate(axes) if linear[parent_axis, axis] < 0]
    mapping = {'slices': tuple(slice(int(a), int(b) + 1) for a, b in zip(low, high)), 'axes': tuple(map(int, axes)), 'flips': tuple(flips), 'offset': offset.tolist(), 'linear': linear.tolist(), 'world_error_mm': error_mm}
    return mapping, 'geometry_pass'


def match_crop(crop_path, candidates):
    """Identify a crop using the default image affine and exact scaled voxel values."""
    started = time.perf_counter()
    result = {'crop': str(crop_path), 'status': 'unmatched', 'matches': [], 'candidates': [], 'voxel_reads': 0}
    try:
        crop = nib.load(str(crop_path))
        if crop.ndim != 3:
            result.update(status='invalid_crop', reason='requires_3d')
            return result
    except Exception as exc:
        result.update(status='invalid_crop', reason=f'header_error: {type(exc).__name__}')
        return result
    data = None
    confirmed = []
    incomplete = False
    for candidate in candidates:
        record = {'candidate': str(candidate.path)}
        result['candidates'].append(record)
        if candidate.error:
            record['reason'] = candidate.error
            incomplete = True
            continue
        try:
            mapping, reason = geometry_mapping(crop, candidate.image)
        except Exception as exc:
            mapping, reason = None, f'geometry_error: {type(exc).__name__}'
            incomplete = True
        record['reason'] = reason
        if mapping is None:
            incomplete |= reason in {'invalid_affine', 'singular_affine'}
            continue
        if data is None:
            try:
                data = np.asanyarray(crop.dataobj)
            except Exception as exc:
                result.update(status='invalid_crop', reason=f'voxel_read_error: {type(exc).__name__}')
                return result
            if not np.isfinite(data).all() or np.min(data) == np.max(data):
                result.update(status='invalid_crop', reason='nonfinite_or_constant_data')
                return result
        try:
            subvolume = np.asanyarray(candidate.image.dataobj[mapping['slices']])
            result['voxel_reads'] += 1
            subvolume = np.transpose(subvolume, mapping['axes'])
            if mapping['flips']:
                subvolume = np.flip(subvolume, axis=mapping['flips'])
            equal = np.array_equal(data, subvolume)
        except Exception as exc:
            record['reason'] = f'voxel_read_error: {type(exc).__name__}'
            incomplete = True
            continue
        if not equal:
            record['reason'] = 'voxel_values_differ'
            continue
        record.update(reason='exact_match', offset=mapping['offset'], axis_mapping=mapping['linear'], world_error_mm=mapping['world_error_mm'])
        confirmed.append(record)
    result['matches'] = [row['candidate'] for row in confirmed]
    result['status'] = 'incomplete' if incomplete else ('unique' if len(confirmed) == 1 else 'ambiguous' if confirmed else 'unmatched')
    if result['status'] == 'unique':
        result['verified_mapping'] = confirmed[0]
    result['elapsed_seconds'] = time.perf_counter() - started
    return result


def replace_cropped_images(temporary_dir, subject_dir):
    """Install uniquely matched orphan crops byte-for-byte, keeping backups and a log.

    Matching is completed before any replacement so multiple crops targeting the
    same parent cannot silently overwrite one another. JSON and spatial headers
    are never rewritten.
    """
    temporary_dir, subject_dir = Path(temporary_dir), Path(subject_dir)
    if not temporary_dir.is_dir() or not subject_dir.is_dir():
        return []
    logger = logging.getLogger(__name__)
    crops = []
    for path in sorted(temporary_dir.iterdir()):
        match = re.fullmatch(r'(.+)_Crop_\d+(\.nii(?:\.gz)?)', path.name)
        if match and path.is_file() and not (temporary_dir / (match[1] + match[2])).exists():
            crops.append(path)
    if not crops:
        return []
    paths = [path for path in subject_dir.rglob('*') if path.is_file() and path.name.endswith(('.nii', '.nii.gz'))]
    candidates = load_candidates(paths)
    snapshots = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths + crops}
    results = [match_crop(path, candidates) for path in crops]
    accepted = {'unique'}
    targets = Counter(row['matches'][0] for row in results if row['status'] in accepted)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='crop_matching_', suffix='.jsonl', dir=temporary_dir, delete=False) as report:
        for row in results:
            row['report'] = report.name
            row['action'] = 'skipped'
            try:
                if row['status'] not in accepted:
                    logger.warning('Crop not replaced (%s): %s', row['status'], row['crop'])
                    continue
                crop, target = Path(row['crop']), Path(row['matches'][0])
                if targets[str(target)] != 1:
                    row['status'] = 'ambiguous_crops'
                    logger.warning('Multiple crops match the same target; preserving all files: %s', target)
                    continue
                if crop.name.endswith('.gz') != target.name.endswith('.gz'):
                    row['status'] = 'compression_mismatch'
                    logger.warning('Crop and target compression differ; preserving both files: %s', crop)
                    continue
                if any((path.stat().st_size, path.stat().st_mtime_ns) != snapshots[str(path)] for path in (crop, target)):
                    row['status'] = 'files_changed'
                    logger.warning('Files changed during crop matching; preserving both files: %s', crop)
                    continue
                backup = temporary_dir / 'crop_backups' / target.relative_to(subject_dir)
                if backup.exists():
                    row['status'] = 'backup_exists'
                    logger.warning('Existing crop backup will not be overwritten: %s', backup)
                    continue
                backup.parent.mkdir(parents=True, exist_ok=True)
                with target.open('rb') as source, backup.open('xb') as destination:
                    shutil.copyfileobj(source, destination)
                shutil.copystat(target, backup)
                row['backup'] = str(backup)
                os.replace(crop, target)
                row['action'] = 'replaced'
                logger.info('Installed verified crop: %s -> %s (backup: %s)', crop, target, backup)
            except Exception as exc:
                row['action'] = 'failed'
                row['error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                report.write(json.dumps(row) + '\n')
                report.flush()
    return results
