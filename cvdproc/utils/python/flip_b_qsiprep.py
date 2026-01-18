#!/usr/bin/env python3

from pathlib import Path
import numpy as np


QSIPREP_ROOT = Path("/mnt/f/BIDS/WCH_AF_Project/derivatives/qsiprep")
FLIP_AXIS = 1  # y axis
SUFFIX_OUT = "_flipped"
OVERWRITE = False

TARGET_STR = "-0.00000000"  # strict trigger condition


def load_bvec_as_3xn(bvec_path: Path) -> np.ndarray:
    bvecs = np.loadtxt(bvec_path)
    if bvecs.ndim != 2:
        raise ValueError(f"Unexpected bvec ndim: {bvecs.ndim}")

    if bvecs.shape[0] == 3:
        return bvecs
    if bvecs.shape[1] == 3:
        return bvecs.T

    raise ValueError(f"Unexpected bvec shape: {bvecs.shape}")


def need_flip_y(bvec_path: Path) -> bool:
    """
    Return True only if:
      - the first value of the 2nd line (y-axis) is exactly '-0.00000000'
    """
    with open(bvec_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return False

    first_y_value = lines[1].strip().split()[0]
    return first_y_value == TARGET_STR


def flip_bvec_y(in_bvec: Path, out_bvec: Path) -> None:
    bvecs = load_bvec_as_3xn(in_bvec)
    bvecs[FLIP_AXIS, :] = -bvecs[FLIP_AXIS, :]
    np.savetxt(out_bvec, bvecs, fmt="%.8f")


def main() -> None:
    if not QSIPREP_ROOT.exists():
        raise SystemExit(f"ERROR: path not found: {QSIPREP_ROOT}")

    pattern = "sub-*/ses-*/dwi/*.bvec"
    bvec_files = sorted(QSIPREP_ROOT.glob(pattern))

    n_total = n_need = n_done = n_skipped = n_failed = 0

    for bvec in bvec_files:
        n_total += 1

        # Skip already flipped files
        if SUFFIX_OUT in bvec.stem:
            n_skipped += 1
            continue

        # Content-based check
        if not need_flip_y(bvec):
            print(f"[SKIP condition] {bvec}")
            n_skipped += 1
            continue

        out_bvec = bvec.with_name(bvec.stem + SUFFIX_OUT + bvec.suffix)

        if out_bvec.exists() and not OVERWRITE:
            print(f"[SKIP exists] {out_bvec}")
            n_skipped += 1
            continue

        try:
            flip_bvec_y(bvec, out_bvec)
            print(f"[FLIPPED] {bvec} -> {out_bvec}")
            n_done += 1
            n_need += 1
        except Exception as e:
            print(f"[FAIL] {bvec} :: {e}")
            n_failed += 1

    print("\nSummary")
    print(f"  Root:       {QSIPREP_ROOT}")
    print(f"  Total:      {n_total}")
    print(f"  Need flip:  {n_need}")
    print(f"  Done:       {n_done}")
    print(f"  Skipped:    {n_skipped}")
    print(f"  Failed:     {n_failed}")


if __name__ == "__main__":
    main()
