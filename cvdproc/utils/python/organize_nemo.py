import os
import re
import shutil
import argparse

def organize_nemo_results(input_dir, output_dir):
    # match format: sub-<subject_id>_ses-<session_id>
    pattern = re.compile(r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)")

    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            subject = match.group("sub")
            session = match.group("ses")

            # create target directory structure
            target_dir = os.path.join(output_dir, f"sub-{subject}", f"ses-{session}")
            os.makedirs(target_dir, exist_ok=True)

            # copy file
            src_file = os.path.join(input_dir, filename)
            dst_file = os.path.join(target_dir, filename)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
        else:
            print(f"Skipped: {filename} (not matching pattern)")

def main():
    parser = argparse.ArgumentParser(description="Organize NEMO result files into BIDS-like folders.")
    parser.add_argument("input_dir", type=str, help="Path to the folder containing NEMO output files")
    parser.add_argument("output_dir", type=str, help="Path to the organized output folder")
    args = parser.parse_args()

    organize_nemo_results(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
