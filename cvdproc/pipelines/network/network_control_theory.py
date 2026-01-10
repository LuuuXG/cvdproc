#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd

from nctpy.utils import matrix_normalization
from nctpy.energies import get_control_inputs, integrate_u

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# Config
# =========================================================
system = "continuous"  # "continuous" or "discrete"
c = 1.0                # normalization constant in matrix_normalization
T = 1.0                # time horizon
rho = 1.0              # mixing parameter for optimal control
thr = 1e-8             # numerical error threshold

enforce_symmetry = True
zero_diagonal = True

use_full_control = True
make_plots = True

# Compare normalized vs raw:
# - True: use matrix_normalization(A)
# - False: use raw A
use_matrix_normalization = True

atrophy_xlsx_file = "/mnt/e/WPS_Cloud/1136007837/WPS云盘/paper/rssi_glymphatic_analysis/data/analysis/data_for_analysis_wide_20260109.xlsx"
atrophy_sheet_name = 0
atrophy_subject_col = "subject"

lh_start_col_idx = 974
lh_end_col_idx = 974 + 34
rh_start_col_idx = 1042
rh_end_col_idx = 1042 + 34

scn_root = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/scn"
out_root = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/population/atrophy_nct/norm"
os.makedirs(out_root, exist_ok=True)

# =========================================================
# Paste your subject-session list here (baseline will be skipped)
# =========================================================
pairs_text = """
sub-SSI0008	ses-baseline
sub-SSI0008	ses-F2
sub-SSI0045	ses-baseline
sub-SSI0045	ses-F1
sub-SSI0103	ses-baseline
sub-SSI0103	ses-F1
sub-SSI0113	ses-baseline
sub-SSI0113	ses-F1
sub-SSI0114	ses-baseline
sub-SSI0114	ses-F1
sub-SSI0117	ses-baseline
sub-SSI0117	ses-F1
sub-SSI0123	ses-baseline
sub-SSI0123	ses-F1
sub-SSI0129	ses-baseline
sub-SSI0129	ses-F1
sub-SSI0130	ses-baseline
sub-SSI0130	ses-F1
sub-SSI0132	ses-baseline
sub-SSI0132	ses-F1
sub-SSI0133	ses-baseline
sub-SSI0133	ses-F1
sub-SSI0139	ses-baseline
sub-SSI0139	ses-F1
sub-SSI0140	ses-baseline
sub-SSI0140	ses-F1
sub-SSI0144	ses-baseline
sub-SSI0144	ses-F1
sub-SSI0149	ses-baseline
sub-SSI0149	ses-F1
sub-SSI0150	ses-baseline
sub-SSI0150	ses-F1
sub-SSI0152	ses-baseline
sub-SSI0152	ses-F2
sub-SSI0158	ses-baseline
sub-SSI0158	ses-F1
sub-SSI0159	ses-baseline
sub-SSI0159	ses-F2
sub-SSI0160	ses-baseline
sub-SSI0160	ses-F1
sub-SSI0162	ses-baseline
sub-SSI0162	ses-F1
sub-SSI0166	ses-baseline
sub-SSI0166	ses-F1
sub-SSI0169	ses-baseline
sub-SSI0169	ses-F1
sub-SSI0170	ses-baseline
sub-SSI0170	ses-F1
sub-SSI0174	ses-baseline
sub-SSI0174	ses-F1
sub-SSI0176	ses-baseline
sub-SSI0176	ses-F1
sub-SSI0179	ses-baseline
sub-SSI0179	ses-F1
sub-SSI0181	ses-baseline
sub-SSI0181	ses-F1
sub-SSI0182	ses-baseline
sub-SSI0182	ses-F1
sub-SSI0183	ses-baseline
sub-SSI0183	ses-F1
sub-SSI0184	ses-baseline
sub-SSI0184	ses-F1
sub-SSI0188	ses-baseline
sub-SSI0188	ses-F1
sub-SSI0191	ses-baseline
sub-SSI0191	ses-F1
sub-SSI0193	ses-baseline
sub-SSI0193	ses-F1
sub-SSI0194	ses-baseline
sub-SSI0194	ses-F1
sub-SSI0196	ses-baseline
sub-SSI0196	ses-F1
sub-SSI0205	ses-baseline
sub-SSI0205	ses-F1
sub-SSI0206	ses-baseline
sub-SSI0206	ses-F1
sub-SSI0211	ses-baseline
sub-SSI0211	ses-F1
sub-SSI0212	ses-baseline
sub-SSI0212	ses-F1
sub-SSI0216	ses-baseline
sub-SSI0216	ses-F1
sub-SSI0219	ses-baseline
sub-SSI0219	ses-F1
sub-SSI0220	ses-baseline
sub-SSI0220	ses-F1
sub-SSI0221	ses-baseline
sub-SSI0221	ses-F1
sub-SSI0222	ses-baseline
sub-SSI0222	ses-F1
sub-SSI0225	ses-baseline
sub-SSI0225	ses-F1
sub-SSI0228	ses-baseline
sub-SSI0228	ses-F1
sub-SSI0248	ses-baseline
sub-SSI0248	ses-F1
sub-SSI0256	ses-baseline
sub-SSI0256	ses-F1
sub-SSI0261	ses-baseline
sub-SSI0261	ses-F1
sub-SSI0268	ses-baseline
sub-SSI0268	ses-F1
sub-SSI0270	ses-baseline
sub-SSI0270	ses-F1
sub-SSI0273	ses-baseline
sub-SSI0273	ses-F1
sub-SSI0288	ses-baseline
sub-SSI0288	ses-F1
sub-SSI0293	ses-baseline
sub-SSI0293	ses-F1
sub-SSI0297	ses-baseline
sub-SSI0297	ses-F1
sub-SSI0303	ses-baseline
sub-SSI0303	ses-F1
sub-SSI0310	ses-baseline
sub-SSI0310	ses-F1
"""

# =========================================================
# Helper functions
# =========================================================
def parse_pairs(text: str):
    pairs = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        sub = parts[0].strip()
        ses = parts[1].strip()
        if ses.lower() == "ses-baseline":
            continue
        pairs.append((sub, ses))
    return pairs

def load_atrophy_from_xlsx(
    xlsx_path: str,
    sheet_name,
    subject_value: str,
    subject_col: str,
    lh_start: int,
    lh_end: int,
    rh_start: int,
    rh_end: int,
) -> np.ndarray:
    df_x = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if subject_col not in df_x.columns:
        raise ValueError(
            f"Subject column '{subject_col}' not found in xlsx. "
            f"Available columns: {list(df_x.columns)}"
        )

    df_x[subject_col] = df_x[subject_col].astype(str)
    hit = df_x.loc[df_x[subject_col] == subject_value]
    if hit.shape[0] == 0:
        raise ValueError(
            f"Subject '{subject_value}' not found in xlsx column '{subject_col}'. "
            f"Examples: {df_x[subject_col].dropna().astype(str).head(5).to_list()}"
        )
    if hit.shape[0] > 1:
        raise ValueError(
            f"Subject '{subject_value}' has {hit.shape[0]} duplicated rows in xlsx. "
            "Please deduplicate."
        )

    row = hit.iloc[0]

    n_cols = df_x.shape[1]
    for name, a, b in [("LH", lh_start, lh_end), ("RH", rh_start, rh_end)]:
        if not (0 <= a < b <= n_cols):
            raise ValueError(f"{name} index range out of bounds: [{a}, {b}) with total columns {n_cols}.")
        if (b - a) != 34:
            raise ValueError(f"{name} range must contain exactly 34 columns, got {b - a}.")

    lh_vals = row.iloc[lh_start:lh_end].to_numpy()
    rh_vals = row.iloc[rh_start:rh_end].to_numpy()

    lh_vals = pd.to_numeric(pd.Series(lh_vals), errors="raise").to_numpy(dtype=float)
    rh_vals = pd.to_numeric(pd.Series(rh_vals), errors="raise").to_numpy(dtype=float)

    x = np.concatenate([lh_vals, rh_vals], axis=0).reshape((68, 1))
    if not np.all(np.isfinite(x)):
        bad_idx = np.where(~np.isfinite(x[:, 0]))[0].tolist()
        raise ValueError(f"Atrophy vector contains NaN/Inf at indices: {bad_idx}")
    return x

def load_connectivity_csv(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    df = df.apply(pd.to_numeric, errors="raise")
    roi_rows = df.index.to_list()
    roi_cols = df.columns.to_list()
    if roi_rows != roi_cols:
        raise ValueError(
            "Row/column ROI order mismatch. "
            f"First row ROI: {roi_rows[0]}, first col ROI: {roi_cols[0]}"
        )
    A = df.to_numpy(dtype=float)
    if enforce_symmetry:
        A = (A + A.T) / 2.0
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    if not np.all(np.isfinite(A)):
        raise ValueError("A contains NaN or Inf.")
    return A, roi_rows

def solve_nct(A_in: np.ndarray, x0: np.ndarray, xf: np.ndarray):
    n = A_in.shape[0]
    B = np.eye(n, dtype=float) if use_full_control else np.ones((n, 1), dtype=float)
    S = np.eye(n, dtype=float)

    x_traj, u, n_err = get_control_inputs(
        A_norm=A_in,
        T=T,
        B=B,
        x0=x0,
        xf=xf,
        system=system,
        rho=rho,
        S=S
    )

    inv_err = float(n_err[0])
    rec_err = float(n_err[1])

    node_energy = integrate_u(u)
    node_energy = np.asarray(node_energy, dtype=float).reshape(-1)
    total_energy = float(node_energy.sum())

    return x_traj, u, inv_err, rec_err, node_energy, total_energy

# =========================================================
# Main loop
# =========================================================
pairs = parse_pairs(pairs_text)
if len(pairs) == 0:
    raise SystemExit("No non-baseline (subject, session) pairs found.")

summary_rows = []
failed_rows = []

for sub, ses in pairs:
    subject_id = sub.replace("sub-", "")
    session_id = ses.replace("ses-", "")

    mind_csv = f"{scn_root}/sub-{subject_id}/ses-{session_id}/MIND/sub-{subject_id}_ses-{session_id}_seg-DK_desc-MIND_connectivity.csv"
    if not os.path.exists(mind_csv):
        failed_rows.append({"subject": sub, "session": ses, "reason": f"Missing connectivity: {mind_csv}"})
        continue

    try:
        A, roi_rows = load_connectivity_csv(mind_csv)
        n_nodes = A.shape[0]
        if A.shape != (68, 68):
            raise ValueError(f"Expected (68, 68), got {A.shape}")

        x_atrophy = load_atrophy_from_xlsx(
            xlsx_path=atrophy_xlsx_file,
            sheet_name=atrophy_sheet_name,
            subject_value=sub,
            subject_col=atrophy_subject_col,
            lh_start=lh_start_col_idx,
            lh_end=lh_end_col_idx,
            rh_start=rh_start_col_idx,
            rh_end=rh_end_col_idx,
        )
        x_baseline = np.zeros((n_nodes, 1), dtype=float)

        A_used = matrix_normalization(A=A, c=c, system=system) if use_matrix_normalization else A
        if not np.all(np.isfinite(A_used)):
            raise ValueError("A_used contains NaN/Inf.")

        x_traj, u, inv_err, rec_err, node_energy, total_energy = solve_nct(A_used, x_baseline, x_atrophy)

        # Save stats
        out_stat_csv = f"{out_root}/{sub}_{ses}_nct_stats.csv"
        stat_df = pd.DataFrame({
            "subject": [sub],
            "session": [ses],
            "use_matrix_normalization": [bool(use_matrix_normalization)],
            "inversion_error": [inv_err],
            "reconstruction_error": [rec_err],
            "total_energy": [total_energy],
        })
        stat_df.to_csv(out_stat_csv, index=False)

        # Save nodal energy
        out_energy_csv = f"{out_root}/{sub}_{ses}_nodal_energy.csv"
        energy_df = pd.DataFrame({"roi": roi_rows, "node_energy": node_energy}).sort_values("node_energy", ascending=False)
        energy_df.to_csv(out_energy_csv, index=False)

        # Save plot
        out_plot_path = f"{out_root}/{sub}_{ses}_control_trajectory.png"
        if make_plots:
            f, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].plot(u)
            ax[0].set_title("control signals")
            ax[1].plot(x_traj)
            ax[1].set_title("state trajectory (neural activity)")
            for cax in ax.reshape(-1):
                cax.set_ylabel("activity")
                cax.set_xlabel("time (arbitrary units)")
                cax.set_xticks([0, x_traj.shape[0]])
                cax.set_xticklabels([0, T])
            f.tight_layout()
            f.savefig(out_plot_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
            plt.close(f)

        # Record summary
        summary_rows.append({
            "subject": sub,
            "session": ses,
            "use_matrix_normalization": bool(use_matrix_normalization),
            "inversion_error": inv_err,
            "reconstruction_error": rec_err,
            "total_energy": total_energy,
            "stat_csv": out_stat_csv,
            "nodal_energy_csv": out_energy_csv,
            "plot_png": out_plot_path if make_plots else "",
            "inv_ok": inv_err < thr,
            "rec_ok": rec_err < thr,
        })

        print(f"[OK] {sub} {ses} | total_energy={total_energy:.6f} | inv={inv_err:.2E} | rec={rec_err:.2E}")

    except Exception as e:
        failed_rows.append({"subject": sub, "session": ses, "reason": str(e)})
        print(f"[FAIL] {sub} {ses} | {str(e)}")

# Save summaries
summary_csv = f"{out_root}/atrophy_nct_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
print(f"\nSaved summary to: {summary_csv}")

failed_csv = f"{out_root}/atrophy_nct_failed.csv"
pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)
print(f"Saved failures to: {failed_csv}")
