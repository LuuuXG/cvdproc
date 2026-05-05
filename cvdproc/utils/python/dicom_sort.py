from pathlib import Path
import shutil
import re
import pydicom

ROOT_DIR = Path(r"/mnt/f/SVD/樊焱0034077070_20250408")

MOVE_FILES = True
OVERWRITE = False

def sanitize_name(text):
    text = str(text).strip()
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._ ")
    return text if text else "Unknown"

def is_inside_sorted_folder(path, root):
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) == 1:
        return False
    first_level = rel_parts[0]
    return re.match(r"^\d+_", first_level) is not None or first_level.startswith("UnknownSeries_")

def get_unique_path(dst):
    if not dst.exists() or OVERWRITE:
        return dst

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent

    i = 1
    while True:
        new_dst = parent / f"{stem}_{i:04d}{suffix}"
        if not new_dst.exists():
            return new_dst
        i += 1

def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {ROOT_DIR}")

    all_files = [p for p in ROOT_DIR.rglob("*") if p.is_file()]

    total_files = 0
    dicom_files = 0
    skipped_files = 0

    for file_path in all_files:
        total_files += 1

        if is_inside_sorted_folder(file_path, ROOT_DIR):
            skipped_files += 1
            continue

        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        except Exception:
            skipped_files += 1
            continue

        if not hasattr(ds, "SOPInstanceUID"):
            skipped_files += 1
            continue

        series_number = sanitize_name(getattr(ds, "SeriesNumber", "UnknownSeries"))
        series_description = sanitize_name(getattr(ds, "SeriesDescription", "UnknownDescription"))

        folder_name = f"{series_number}_{series_description}"
        target_dir = ROOT_DIR / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / file_path.name
        target_path = get_unique_path(target_path)

        if MOVE_FILES:
            shutil.move(str(file_path), str(target_path))
        else:
            shutil.copy2(str(file_path), str(target_path))

        dicom_files += 1

    print("DICOM sorting finished.")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Total scanned files: {total_files}")
    print(f"DICOM files sorted: {dicom_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Mode: {'move' if MOVE_FILES else 'copy'}")

if __name__ == "__main__":
    main()