import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load skeletonized MD image
# ----------------------------
nii_file = r"F:\BIDS\WCH_AF_Project\derivatives\dwi_pipeline\sub-AFib0241\ses-baseline\psmd\MD_skeletonized_masked.nii.gz"  # replace with your file path
img = nib.load(nii_file)
data = img.get_fdata()

# Mask out zeros or NaNs
md_values = data[np.isfinite(data)]
md_values = md_values[md_values > 0]

# ----------------------------
# 2. Compute percentiles
# ----------------------------
p5 = np.percentile(md_values, 5)
p95 = np.percentile(md_values, 95)

# ----------------------------
# 3. Plot with dark theme
# ----------------------------
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")

# Histogram in light gray
ax.hist(md_values, bins=60, color='lightgray', edgecolor='white')

# Percentile lines (red and cyan for contrast)
ax.axvline(p5, color='red', linestyle='--', linewidth=2)
ax.axvline(p95, color='cyan', linestyle='--', linewidth=2)

# Optional arrow to indicate PSMD width
ax.annotate('', xy=(p5, 0.8*ax.get_ylim()[1]), xytext=(p95, 0.8*ax.get_ylim()[1]),
            arrowprops=dict(arrowstyle='<->', color='yellow', linewidth=2))

# Clean schematic: hide ticks/labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("PSMD Illustration", fontsize=14, weight='bold', color='white')

plt.tight_layout()
plt.show()