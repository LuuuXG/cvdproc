"""Conservatively remove isolated zig-zags from a GIFTI cortical label map."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix


def adjacency_from_faces(n_vertices: int, faces: np.ndarray):
    rows = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    cols = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    adjacency = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    adjacency = adjacency + adjacency.T
    adjacency.data[:] = 1
    return adjacency


def smooth_labels(labels: np.ndarray, faces: np.ndarray, passes: int, min_votes: int) -> tuple[np.ndarray, list[int]]:
    adjacency = adjacency_from_faces(len(labels), faces)
    neighbours = [adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]] for i in range(len(labels))]
    current = labels.copy()
    changed_per_pass = []
    for _ in range(passes):
        updated = current.copy()
        changes = 0
        for vertex, nearby in enumerate(neighbours):
            label = current[vertex]
            if label == 0 or not len(nearby):
                continue  # Preserve the medial-wall/background mask.
            values, counts = np.unique(current[nearby], return_counts=True)
            winner_index = np.argmax(counts)
            winner, votes = values[winner_index], counts[winner_index]
            # Only alter a one-vertex protrusion when a clear local consensus
            # exists. Ties retain the original anatomical label.
            if winner != 0 and winner != label and votes >= min_votes and np.sum(counts == votes) == 1:
                updated[vertex] = winner
                changes += 1
        current = updated
        changed_per_pass.append(changes)
    return current, changed_per_pass


def boundary_edge_count(labels: np.ndarray, faces: np.ndarray) -> int:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return int(np.sum(labels[edges[:, 0]] != labels[edges[:, 1]]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_gii", type=Path)
    parser.add_argument("surface_gii", type=Path)
    parser.add_argument("output_gii", type=Path)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--min-votes", type=int, default=4)
    args = parser.parse_args()

    label_image = nib.load(str(args.label_gii))
    surface_image = nib.load(str(args.surface_gii))
    labels = label_image.darrays[0].data.astype(np.int32)
    faces = surface_image.darrays[1].data.astype(np.int32)
    if len(labels) != len(surface_image.darrays[0].data):
        raise ValueError("Label and surface vertex counts differ.")

    smoothed, changes = smooth_labels(labels, faces, args.passes, args.min_votes)
    output = nib.gifti.GiftiImage()
    output.add_gifti_data_array(nib.gifti.GiftiDataArray(smoothed, intent="NIFTI_INTENT_LABEL"))
    output.labeltable = copy.deepcopy(label_image.labeltable)
    args.output_gii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(args.output_gii))
    print({
        "changed_vertices_per_pass": changes,
        "boundary_edges_before": boundary_edge_count(labels, faces),
        "boundary_edges_after": boundary_edge_count(smoothed, faces),
        "labels_before": len(np.unique(labels)),
        "labels_after": len(np.unique(smoothed)),
    })


if __name__ == "__main__":
    main()
