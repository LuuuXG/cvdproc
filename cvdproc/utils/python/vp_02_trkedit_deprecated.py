import nibabel as nib

trk = nib.streamlines.load("/mnt/f/BIDS/demo_BIDS/derivatives/tmp/rh_OR.trk")
streamlines = trk.streamlines

print(f"Nibabel successfully loaded {len(streamlines)} streamlines")
print(f"First streamline shape: {streamlines[0].shape}")

# check header
header = trk.header
print("Header information:")
for key, value in header.items():
    print(f"{key}: {value}")