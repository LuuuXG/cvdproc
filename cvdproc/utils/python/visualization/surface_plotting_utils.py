"""Shared helpers for labelled GIFTI statistical surface visualisations."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

NIFTI_INTENT_LABEL = 1002
NIFTI_INTENT_POINTSET = 1008
NIFTI_INTENT_TRIANGLE = 1009

VALUE_COLUMNS = ("statistic_value", "value", "stat")
SIGNIFICANCE_COLUMNS = ("significant", "is_significant", "significance")
P_VALUE_COLUMNS = ("p_value", "pvalue", "p_val", "pval", "p")
Q_VALUE_COLUMNS = ("q_value", "qvalue", "q_val", "qval", "fdr_q", "fdr_p")


def _intent_array(image, intent, path):
    for array in image.darrays:
        if int(array.intent) == intent:
            return array.data
    raise ValueError(f"GIFTI file '{path}' has no intent {intent} data array")


def load_gifti_surface(path, require_labels=False):
    """Return point, triangle and optional label arrays from a GIFTI file."""
    image = nib.load(path)
    points = np.asarray(_intent_array(image, NIFTI_INTENT_POINTSET, path), np.float32)
    faces = np.asarray(_intent_array(image, NIFTI_INTENT_TRIANGLE, path), np.int32)
    labels = None
    for array in image.darrays:
        if int(array.intent) == NIFTI_INTENT_LABEL:
            labels = np.asarray(array.data, np.int32)
            break
    if require_labels and labels is None:
        raise ValueError(f"GIFTI file '{path}' has no label data array")
    return points, faces, labels


def load_gifti_labels(path):
    """Load the label array from a GIFTI label file."""
    image = nib.load(path)
    return np.asarray(_intent_array(image, NIFTI_INTENT_LABEL, path), np.int32)


def pyvista_faces(faces):
    """Convert an ``(n, 3)`` triangle array to PyVista's packed cell format."""
    faces = np.asarray(faces, np.int64)
    return np.c_[np.full(len(faces), 3, dtype=np.int64), faces].ravel()


def parse_significance(value):
    """Parse a boolean-like significance flag."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            raise ValueError("Significance flag cannot be NaN or infinite")
        if float(value) in {0.0, 1.0}:
            return bool(value)
        raise ValueError(
            f"Numeric significance flags must be 0 or 1, got {value!r}; "
            "supply p-values in a p_value CSV column"
        )
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "significant", "sig"}:
        return True
    if text in {"0", "false", "f", "no", "n", "nonsignificant", "non-significant", "ns"}:
        return False
    raise ValueError(f"Cannot interpret significance value: {value!r}")


def parse_label_value_pairs(text, value_parser=float):
    """Parse comma-separated ``label:value`` pairs into an integer-keyed dict."""
    result = {}
    if text is None or not str(text).strip():
        return result
    for item in str(text).split(","):
        label, value = item.strip().split(":", 1)
        result[int(label.strip())] = value_parser(value.strip())
    return result


def _first_present(row, candidates):
    return next((name for name in candidates if name in row and row[name] != ""), None)


def load_statistical_csv(
    path,
    *,
    label_columns=("label_id", "surface_label_id"),
    mesh_name=None,
    alpha=0.05,
):
    """Load statistic and optional significance mappings from a CSV.

    Optional ``hemisphere`` values (L/R) produce tuple keys. If a significance
    flag is absent but a p-value column is present, significance is ``p <= alpha``.
    Rows for another ``mesh_file`` are ignored when ``mesh_name`` is supplied.
    """
    statistics = {}
    significance = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for raw_row in csv.DictReader(handle):
            row = {
                str(key).strip().lower(): str(value).strip()
                for key, value in raw_row.items()
                if key is not None
            }
            if mesh_name and row.get("mesh_file") not in {None, ""}:
                requested = Path(mesh_name).name.lower()
                recorded = Path(row["mesh_file"]).name.lower()
                # Historical AAL files used an ``_ENIGMA`` infix.  Treat it
                # as an alias so old mapping tables remain valid after the
                # meshes were renamed.
                canonical = lambda name: name.replace("_enigma", "")
                if canonical(recorded) != canonical(requested):
                    continue
            label_column = _first_present(row, label_columns)
            value_column = _first_present(row, VALUE_COLUMNS)
            if label_column is None:
                raise ValueError(f"CSV requires one of these label columns: {label_columns}")
            if value_column is None:
                raise ValueError(f"CSV requires one of these value columns: {VALUE_COLUMNS}")
            label = int(row[label_column])
            hemisphere = row.get("hemisphere", "").upper()
            key = (hemisphere, label) if hemisphere in {"L", "R"} else label
            statistics[key] = float(row[value_column])

            significance_column = _first_present(row, SIGNIFICANCE_COLUMNS)
            q_column = _first_present(row, Q_VALUE_COLUMNS)
            p_column = _first_present(row, P_VALUE_COLUMNS)
            if significance_column is not None:
                significance[key] = parse_significance(row[significance_column])
            elif q_column is not None:
                significance[key] = float(row[q_column]) <= alpha
            elif p_column is not None:
                significance[key] = float(row[p_column]) <= alpha
    if not statistics:
        mesh_hint = f" for mesh '{Path(mesh_name).name}'" if mesh_name else ""
        raise ValueError(f"CSV '{path}' contains no usable statistic rows{mesh_hint}")
    return statistics, significance


def normalize_label_mapping(values, label_ids, *, name="values", value_parser=float):
    """Normalize a label mapping or an ordered sequence to an integer-keyed dict."""
    label_ids = sorted(int(label) for label in label_ids)
    if isinstance(values, Mapping):
        result = {int(key): value_parser(value) for key, value in values.items()}
    elif isinstance(values, np.ndarray) or (
        isinstance(values, Sequence) and not isinstance(values, (str, bytes))
    ):
        if len(values) != len(label_ids):
            raise ValueError(f"Expected {len(label_ids)} ordered {name}, got {len(values)}")
        result = dict(zip(label_ids, (value_parser(value) for value in values)))
    else:
        raise TypeError(f"{name} must be a label mapping or an ordered sequence")
    missing = sorted(set(label_ids) - set(result))
    if missing:
        raise ValueError(f"Missing {name} for labels: {missing}")
    return result


def normalize_significance_mapping(values, label_ids):
    """Normalize optional significance input; omitted mapping keys default to significant."""
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {int(key): parse_significance(value) for key, value in values.items()}
    return normalize_label_mapping(
        values,
        label_ids,
        name="significance flags",
        value_parser=parse_significance,
    )


def label_lookup(mapping, label, *, hemisphere=None, default=None):
    """Look up a hemisphere-specific key first, then its shared label key."""
    label = int(label)
    if mapping is None:
        return default
    if hemisphere is not None and (hemisphere, label) in mapping:
        return mapping[(hemisphere, label)]
    return mapping.get(label, default)


def vertex_statistic_and_opacity(
    labels,
    statistics,
    significance=None,
    *,
    hemisphere=None,
    nonsignificant_opacity=0.35,
    missing_value=np.nan,
):
    """Expand label mappings into per-vertex statistic and opacity arrays."""
    if not 0.0 <= nonsignificant_opacity <= 1.0:
        raise ValueError("nonsignificant_opacity must be between 0 and 1")
    labels = np.asarray(labels, np.int32)
    scalar = np.full(len(labels), missing_value, dtype=np.float32)
    opacity = np.ones(len(labels), dtype=np.float32)
    for label in np.unique(labels):
        selection = labels == label
        scalar[selection] = label_lookup(
            statistics, label, hemisphere=hemisphere, default=missing_value
        )
        is_significant = label_lookup(
            significance, label, hemisphere=hemisphere, default=True
        )
        opacity[selection] = 1.0 if parse_significance(is_significant) else nonsignificant_opacity
    return scalar, opacity


def colour_limits(values, vmin=None, vmax=None, symmetric_if_diverging=False):
    """Infer finite colour limits, optionally symmetric around zero."""
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite statistic values were supplied")
    data_min, data_max = float(finite.min()), float(finite.max())
    if vmin is None and vmax is None and symmetric_if_diverging and data_min < 0 < data_max:
        limit = max(abs(data_min), abs(data_max))
        return -limit, limit
    return data_min if vmin is None else float(vmin), data_max if vmax is None else float(vmax)
