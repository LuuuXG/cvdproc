import numpy as np
import pandas as pd

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical

from cvdproc.config.paths import get_package_path

# -------------------------
# User configuration
# -------------------------
# 1. A csv file containing the DK68 values (lh and then rh, 1*68 or 68*1)
csv_path = "/mnt/e/Neuroimage/dk68_ct.csv"
# 2. Color for medial wall
medial_wall_color = (1, 1, 1, 1)  # RGBA (eg. 1, 1, 1, 0 is totally transparent)
# 3. Other settings
# please see lines 42-51 (https://enigma-toolbox.readthedocs.io/en/latest/pages/13.01.apireference/generated/enigmatoolbox.plotting.surface_plotting.plot_cortical.html#enigmatoolbox.plotting.surface_plotting.plot_cortical)

# -------------------------
# Other settings (do not change)
# -------------------------
medial_wall_csv = get_package_path('data', 'standard', 'fsaverage5', 'fsaverage5_cortical_mask.csv')

# -------------------------
# Main script
# -------------------------
dk68 = pd.read_csv(csv_path, header=None).to_numpy().reshape(-1)

dk_fsa5 = parcel_to_surface(
    dk68,
    "aparc_fsa5",
    fill=np.nan,
)

mask = pd.read_csv(
    medial_wall_csv,
    header=None,
).to_numpy().reshape(-1)

dk_fsa5[mask == 0] = np.nan

plot_cortical(
    array_name=dk_fsa5,
    surface_name="fsa5",
    cmap="RdBu",
    color_bar=True,
    color_range=(-3.5, 3.5),
    #background=(1,1,1),
    nan_color=medial_wall_color,
    size=(900, 600),
    scale=(3, 3),
    screenshot=True,
    background=(1,1,1),
    transparent_bg=False,
    filename="/mnt/e/Neuroimage/dk68_ct.jpg"
)
