"""Guard project source and documentation against legacy MNI space spelling."""

import os
from pathlib import Path
import unittest


class MniSpaceNamingTest(unittest.TestCase):
    def test_project_files_use_canonical_space(self):
        root = Path(__file__).resolve().parents[3]
        legacy = "MNI152NLin6" + "ASym"
        migration = root / "cvdproc/utils/python/fix_mni_space_filenames.py"
        suffixes = {".py", ".sh", ".m", ".r", ".md", ".yml", ".yaml", ".json", ".toml"}
        violations = []
        for source_root in (root / "cvdproc", root / "docs"):
            for directory, dirs, files in os.walk(source_root):
                dirs[:] = [name for name in dirs if name not in {"data", "trash", "external", "__pycache__", ".ipynb_checkpoints"}]
                for name in files:
                    path = Path(directory) / name
                    if path.suffix.lower() not in suffixes:
                        continue
                    for number, line in enumerate(path.read_bytes().splitlines(), 1):
                        if legacy.encode("ascii") not in line:
                            continue
                        if path == migration and line.strip() == f'OLD_SPACE = "{legacy}"'.encode("ascii"):
                            continue
                        violations.append(f"{path.relative_to(root)}:{number}")
        self.assertEqual(violations, [], "Legacy space names found:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
