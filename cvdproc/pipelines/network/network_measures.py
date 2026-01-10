import numpy as np
import scipy.io as sio
import bct

mat_file_path = "/mnt/e/neuroimage/whole_brain_FreeSurferDKT_Cortical.mat"

# -----------------------------
# Load connectivity
# -----------------------------
mat = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
W = mat["connectivity"].astype(float)

# Basic cleaning
np.fill_diagonal(W, 0.0)
W[~np.isfinite(W)] = 0.0
W[W < 0] = 0.0

# -----------------------------
# Binary adjacency
# -----------------------------
A = (W > 0).astype(int)

# -----------------------------
# Scheme A: max normalization only
# -----------------------------
Wn = W.copy()
mx = Wn.max()
if mx > 0:
    Wn = Wn / mx

# -----------------------------
# Global efficiency
# -----------------------------
Eglob_bin = bct.efficiency_bin(A)
Eglob_wei = bct.efficiency_wei(Wn)

print(f"Global efficiency (binary):  {Eglob_bin:.6f}")
print(f"Global efficiency (weighted, max-norm): {Eglob_wei:.6f}")

# -----------------------------
# Local efficiency
# -----------------------------
Eloc_bin = bct.efficiency_bin(A, local=True)
Eloc_wei = bct.efficiency_wei(Wn, local=True)

print(f"Local efficiency (binary):  mean={Eloc_bin.mean():.6f}")
print(f"Local efficiency (weighted, max-norm): mean={Eloc_wei.mean():.6f}")

# -----------------------------
# Modularity
# -----------------------------
n_rep = 100  # common choices: 50-500

best_Q_bin = -np.inf
best_Ci_bin = None

for _ in range(n_rep):
    Ci, Q = bct.modularity_louvain_und(A)
    if Q > best_Q_bin:
        best_Q_bin = Q
        best_Ci_bin = Ci

print(f"Binary modularity: Q={best_Q_bin:.6f}, n_communities={len(np.unique(best_Ci_bin))}")

best_Q_wei = -np.inf
best_Ci_wei = None

for _ in range(n_rep):
    Ci, Q = bct.modularity_louvain_und(Wn)
    if Q > best_Q_wei:
        best_Q_wei = Q
        best_Ci_wei = Ci

print(f"Weighted modularity (max-norm): Q={best_Q_wei:.6f}, n_communities={len(np.unique(best_Ci_wei))}")