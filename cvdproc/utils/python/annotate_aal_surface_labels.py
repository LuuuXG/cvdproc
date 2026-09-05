"""Add AAL v4 ROI names to the 32k left/right GIFTI label tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib


def cortical_names(roi_txt: Path, side: str) -> list[str]:
    suffix = f"_{side}"
    names = []
    for line in roi_txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        name, atlas_code = parts[1], int(parts[2])
        # Surface map contains the 41 cerebral AAL labels: omit the four
        # separately represented subcortical nuclei and all cerebellar labels.
        if name.endswith(suffix) and 2000 <= atlas_code < 9000 and atlas_code // 100 not in (70, 71):
            names.append(name)
    if len(names) != 41:
        raise ValueError(f"Expected 41 cortical {side} labels, found {len(names)}")
    return names


def aal_codes(roi_txt: Path) -> dict[str, int]:
    return {
        parts[1]: int(parts[2])
        for line in roi_txt.read_text(encoding="utf-8").splitlines()
        if len(parts := line.split()) == 3
    }


def annotate(label_path: Path, names: list[str]) -> None:
    image = nib.load(str(label_path))
    for entry in image.labeltable.labels:
        if entry.key == 0:
            entry.label = "MedialWall"
        elif 1 <= entry.key <= 41:
            entry.label = names[entry.key - 1]
    nib.save(image, str(label_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roi_txt", type=Path)
    parser.add_argument("left_label", type=Path)
    parser.add_argument("right_label", type=Path)
    parser.add_argument("mapping_csv", type=Path)
    args = parser.parse_args()
    left_names, right_names = cortical_names(args.roi_txt, "L"), cortical_names(args.roi_txt, "R")
    annotate(args.left_label, left_names)
    annotate(args.right_label, right_names)
    codes = aal_codes(args.roi_txt)
    with args.mapping_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["hemisphere", "surface_label_id", "aal_name", "aal_volume_code"])
        for side, names in (("L", left_names), ("R", right_names)):
            writer.writerow([side, 0, "MedialWall", ""])
            writer.writerows((side, key, name, codes[name]) for key, name in enumerate(names, start=1))


if __name__ == "__main__":
    main()
