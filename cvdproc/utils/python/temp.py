import os
import gzip
import numpy as np
from collections import Counter
from scipy.io import loadmat


def parse_tt(track_bytes):
    buf1 = np.array(track_bytes, dtype=np.uint8)
    buf2 = buf1.view(np.int8)
    pos, i, L = [], 0, len(buf1)

    while i < L:
        pos.append(i)
        size = np.frombuffer(buf1[i:i+4].tobytes(), dtype=np.uint32)[0]
        i += size + 13

    streamlines = []
    for p in pos:
        npts = np.frombuffer(buf1[p:p+4].tobytes(), dtype=np.uint32)[0] // 3
        x = np.frombuffer(buf1[p+4:p+8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[p+8:p+12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[p+12:p+16].tobytes(), dtype=np.int32)[0]

        coords = np.zeros((npts, 3), dtype=np.float32)
        coords[0] = [x, y, z]

        q = p + 16
        for j in range(1, npts):
            x += int(buf2[q])
            y += int(buf2[q + 1])
            z += int(buf2[q + 2])
            q += 3
            coords[j] = [x, y, z]

        streamlines.append(coords / 32.0)

    return streamlines


tt_file = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0086/ses-baseline/visual_pathway_analysis/sub-HC0086_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OR_desc-voxelspace_streamlines.tt.gz"

if not os.path.isfile(tt_file):
    raise FileNotFoundError(f"TT file not found: {tt_file}")

with gzip.open(tt_file, "rb") as f:
    mat = loadmat(f, squeeze_me=True, struct_as_record=False)

if "track" not in mat:
    raise KeyError("No 'track' field found in TT file.")

streamlines = parse_tt(mat["track"])
point_counts = np.array([len(sl) for sl in streamlines], dtype=int)

if point_counts.size == 0:
    raise RuntimeError("No streamlines found in the TT file.")

print(f"TT file: {tt_file}")
print(f"Number of streamlines: {point_counts.size}")
print(f"Min points per streamline: {point_counts.min()}")
print(f"Max points per streamline: {point_counts.max()}")
print(f"Mean points per streamline: {point_counts.mean():.2f}")
print(f"Median points per streamline: {np.median(point_counts):.2f}")
print()

count_table = Counter(point_counts)
print("Point count distribution:")
for n_points in sorted(count_table):
    print(f"{n_points}: {count_table[n_points]}")

print()
if np.all(point_counts == 100):
    print("All streamlines have exactly 100 points.")
else:
    print("Not all streamlines have 100 points.")
    bad_idx = np.where(point_counts != 100)[0]
    print(f"Number of streamlines not equal to 100 points: {len(bad_idx)}")
    print("First 20 abnormal streamlines:")
    for idx in bad_idx[:20]:
        print(f"Streamline {idx}: {point_counts[idx]} points")