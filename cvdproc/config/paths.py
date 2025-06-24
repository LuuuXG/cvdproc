import os
import glob
import re

# Global path constants
FSAVERAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'standard', 'fsaverage')
LH_MEDIAL_WALL_TXT = os.path.join(FSAVERAGE_DIR, 'lh.aparc.label_medial_wall.txt')
RH_MEDIAL_WALL_TXT = os.path.join(FSAVERAGE_DIR, 'rh.aparc.label_medial_wall.txt')

# Function for metric pair discovery
def find_qcache_metric_pairs(freesurfer_dir, metrics=None, fwhms=None):
    """
    Search FreeSurfer qcache outputs and return matched lh/rh metric pairs.

    Parameters:
        freesurfer_dir (str): Path to FreeSurfer subject/session output directory.
        metrics (list[str], optional): List of metric names to include (e.g., ['thickness', 'area.pial']).
        fwhms (list[int], optional): List of fwhm levels to include (e.g., [0, 5, 10]).

    Returns:
        dict: {
            'thickness.fwhm10': {'lh': path, 'rh': path},
            ...
        }
    """
    surf_dir = os.path.join(freesurfer_dir, 'surf')
    all_mgh_files = glob.glob(os.path.join(surf_dir, '*.fsaverage.mgh'))

    grouped = {}
    pattern = re.compile(r'^(lh|rh)\.(.+)\.(fwhm\d+)\.fsaverage\.mgh$')

    for filepath in all_mgh_files:
        basename = os.path.basename(filepath)
        match = pattern.match(basename)
        if not match:
            continue

        hemi, metric_name, fwhm = match.groups()
        fwhm_val = int(fwhm.replace('fwhm', ''))

        if metrics is not None and metric_name not in metrics:
            continue
        if fwhms is not None and fwhm_val not in fwhms:
            continue

        key = f"{metric_name}.{fwhm}"
        if key not in grouped:
            grouped[key] = {}
        grouped[key][hemi] = filepath

    return {k: v for k, v in grouped.items() if 'lh' in v and 'rh' in v}

# test
if __name__ == "__main__":
    # Example usage
    freesurfer_dir = '/mnt/f/BIDS/SVD_BIDS/derivatives/freesurfer/sub-SVD0042/ses-03'
    pairs = find_qcache_metric_pairs(freesurfer_dir,metrics=['thickness', 'white.K'])
    for metric, paths in pairs.items():
        print(f"{metric}: lh={paths['lh']}, rh={paths['rh']}")
