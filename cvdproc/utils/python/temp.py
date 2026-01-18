import numpy as np

in_file = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/L_OR_ICVF.csv"   # 你的文件路径
out_file = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/L_OR_ICVF_pointwise_mean.txt"

# Load: shape = (n_streamlines, 100)
data = np.loadtxt(in_file)

# Sanity check
if data.ndim != 2 or data.shape[1] != 100:
    raise RuntimeError(f"Unexpected shape: {data.shape}, expected (*, 100)")

# Column-wise mean → shape = (100,)
pointwise_mean = np.nanmean(data, axis=0)

# Save
np.savetxt(out_file, pointwise_mean, fmt="%.6f")

print(f"Number of streamlines: {data.shape[0]}")
print("First 10 point-wise means:")
print(pointwise_mean[:10])
print(f"Saved to: {out_file}")
