"""Correct legacy MNI space filenames under a BIDS derivatives directory.

Preview: cvdproc_tool python/fix_mni_space_filenames.py /path/to/bids
Apply:   cvdproc_tool python/fix_mni_space_filenames.py /path/to/bids --apply

Only regular filenames are changed; file contents and directory names are
preserved. Symbolic links are skipped, and linked directories are not traversed.
Stop processing jobs before applying changes. References inside sidecars or
workflow caches are not updated by this filename-only utility.
"""

import argparse
import os
from pathlib import Path
import sys
import uuid


OLD_SPACE = "MNI152NLin6ASym"
NEW_SPACE = "MNI152NLin6Asym"


def find_renames(root):
    renames, conflicts = [], []

    def raise_walk_error(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
        parent = Path(directory)
        names = set(dirs + files)
        for name in sorted(dirs + [name for name in files if OLD_SPACE in name]):
            path = parent / name
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                print(f"SKIP link: {path}")
                if name in dirs:
                    dirs.remove(name)
        for name in sorted(files):
            source = parent / name
            if OLD_SPACE not in name or source.is_symlink() or not source.is_file():
                continue
            target = source.with_name(name.replace(OLD_SPACE, NEW_SPACE))
            if target.name in names:
                conflicts.append((source, target))
            else:
                renames.append((source, target))
        dirs.sort()
    return renames, conflicts


def rename_file(source, target):
    # An intermediate name makes case-only renames work on Windows as well.
    temporary = source.with_name(f".cvdproc-rename-{uuid.uuid4().hex}")
    if os.path.lexists(temporary):
        raise FileExistsError(temporary)
    if any(path.name == target.name for path in source.parent.iterdir()):
        raise FileExistsError(target)
    source.rename(temporary)
    try:
        if os.path.lexists(target):
            raise FileExistsError(target)
        temporary.rename(target)
    except OSError:
        if not os.path.lexists(source):
            temporary.rename(source)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids_dir", type=Path, help="BIDS root; only its derivatives subdirectory is scanned.")
    parser.add_argument("--apply", action="store_true", help="Apply renames; the default is a read-only preview.")
    args = parser.parse_args(argv)
    root = args.bids_dir.expanduser().resolve() / "derivatives"
    if os.name == "nt" and not str(root).startswith("\\\\?\\"):
        root = Path("\\\\?\\UNC\\" + str(root)[2:] if str(root).startswith("\\\\") else "\\\\?\\" + str(root))
    if not root.is_dir() or root.is_symlink():
        parser.error(f"Expected a real derivatives directory: {root}")
    completed = 0
    try:
        renames, conflicts = find_renames(root)
        for source, target in renames:
            print(f"RENAME {source.relative_to(root)} -> {target.relative_to(root)}")
        for source, target in conflicts:
            print(f"CONFLICT {source.relative_to(root)} -> {target.relative_to(root)}", file=sys.stderr)
        if conflicts:
            print("No files changed. Resolve all target-name conflicts before applying.", file=sys.stderr)
            return 1
        if not args.apply:
            print(f"Preview: {len(renames)} file(s) to rename. Run with --apply to execute.")
            return 0
        for source, target in renames:
            rename_file(source, target)
            completed += 1
            print(f"RENAMED {target.relative_to(root)}", flush=True)
        print(f"Renamed {completed} file(s). File contents were preserved.")
        return 0
    except OSError as error:
        print(f"Stopped after {completed} rename(s): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
