import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
DIR_METHOD1 = r"E:/Neuroimage/compare/default"
DIR_METHOD2 = r"E:/Neuroimage/compare/synthstrip"
OUTPUT_DIR = r"E:/Neuroimage/compare/freesurfer_method_agreement"

ID_COLUMNS = ["subject", "session"]

# Which agreement metric to emphasize in the main heatmap
PRIMARY_HEATMAP_METRIC = "ICC3_1"   # options: "ICC3_1", "pearson_r", "mean_abs_diff"

# Heatmap settings
FIG_DPI = 300
CELL_TEXT = False
X_LABEL_ROTATION = 90

# Agreement thresholds for summary
ICC_THRESHOLDS = {
    "excellent": 0.90,
    "good": 0.75,
    "moderate": 0.50
}


# =========================
# Utility functions
# =========================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def list_common_csv_files(dir1: str, dir2: str) -> List[str]:
    files1 = {f.name for f in Path(dir1).glob("*.csv")}
    files2 = {f.name for f in Path(dir2).glob("*.csv")}
    common = sorted(files1.intersection(files2))
    return common


def parse_region_and_metric(col: str) -> Tuple[str, str, str]:
    """
    Example:
        lh.aparc.a2009s_Lat_Fis-ant-Horizont_ThickAvg
    -> hemisphere = lh
       region = aparc.a2009s_Lat_Fis-ant-Horizont
       metric = ThickAvg

    If the column does not match the expected pattern, return best-effort parsing.
    """
    metric = col.split("_")[-1]

    if col.startswith("lh."):
        hemisphere = "lh"
        core = col[3:]
    elif col.startswith("rh."):
        hemisphere = "rh"
        core = col[3:]
    else:
        hemisphere = "unknown"
        core = col

    if "_" in core:
        region = core[: -(len(metric) + 1)]
    else:
        region = core

    return hemisphere, region, metric


def build_column_map(columns: List[str]) -> Dict[Tuple[str, str, str], str]:
    """
    Map:
        (hemisphere, region, metric) -> original column name
    """
    mapping = {}
    for col in columns:
        hemi, region, metric = parse_region_and_metric(col)
        mapping[(hemi, region, metric)] = col
    return mapping


def icc_3_1(x: np.ndarray, y: np.ndarray) -> float:
    """
    Two-way mixed, single measure, consistency ICC(3,1).
    """
    X = np.vstack([x, y]).T
    X = X[~np.isnan(X).any(axis=1)]
    n, k = X.shape

    if n < 3:
        return np.nan

    row_means = np.mean(X, axis=1)
    grand_mean = np.mean(X)

    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_within = np.sum((X - row_means[:, None]) ** 2)

    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    denominator = ms_between + (k - 1) * ms_within
    if denominator == 0:
        return np.nan

    return (ms_between - ms_within) / denominator


def compute_pair_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "n_pairs": len(x),
            "pearson_r": np.nan,
            "ICC3_1": np.nan,
            "mean_difference_m1_minus_m2": np.nan,
            "mean_abs_difference": np.nan,
            "sd_difference": np.nan
        }

    if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        pearson_r = np.nan
    else:
        pearson_r = np.corrcoef(x, y)[0, 1]

    diff = x - y

    return {
        "n_pairs": len(x),
        "pearson_r": pearson_r,
        "ICC3_1": icc_3_1(x, y),
        "mean_difference_m1_minus_m2": float(np.mean(diff)),
        "mean_abs_difference": float(np.mean(np.abs(diff))),
        "sd_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    }


def reorder_metrics(metrics: List[str]) -> List[str]:
    preferred_order = [
        "ThickAvg",
        "ThickStd",
        "SurfArea",
        "GrayVol",
        "MeanCurv",
        "GausCurv",
        "FoldInd",
        "CurvInd",
        "NumVert"
    ]
    metrics_set = set(metrics)
    ordered = [m for m in preferred_order if m in metrics_set]
    ordered.extend(sorted(metrics_set - set(ordered)))
    return ordered


def make_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    out_png: str,
    value_name: str,
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
) -> None:
    data = matrix_df.to_numpy(dtype=float)
    n_rows, n_cols = data.shape

    fig_width = max(12, n_cols * 0.28)
    fig_height = max(4, n_rows * 0.55)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=FIG_DPI)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix_df.columns.tolist(), rotation=X_LABEL_ROTATION, ha="right", fontsize=8)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix_df.index.tolist(), fontsize=9)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Region")
    ax.set_ylabel("Metric")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(value_name)

    if CELL_TEXT:
        for i in range(n_rows):
            for j in range(n_cols):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\\/*?:\[\]]", "_", name)
    return name[:31]


# =========================
# Core analysis
# =========================
def analyze_one_file(file1: str, file2: str, output_dir: str) -> Dict[str, object]:
    basename = Path(file1).name
    stem = Path(file1).stem

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    missing_id_1 = [c for c in ID_COLUMNS if c not in df1.columns]
    missing_id_2 = [c for c in ID_COLUMNS if c not in df2.columns]
    if missing_id_1:
        raise ValueError(f"{basename}: missing ID columns in method1 file: {missing_id_1}")
    if missing_id_2:
        raise ValueError(f"{basename}: missing ID columns in method2 file: {missing_id_2}")

    data_cols_1 = [c for c in df1.columns if c not in ID_COLUMNS]
    data_cols_2 = [c for c in df2.columns if c not in ID_COLUMNS]

    common_data_cols = sorted(set(data_cols_1).intersection(data_cols_2))
    if len(common_data_cols) == 0:
        raise ValueError(f"{basename}: no common data columns found.")

    keep_cols_1 = ID_COLUMNS + common_data_cols
    keep_cols_2 = ID_COLUMNS + common_data_cols

    df1 = df1[keep_cols_1].copy()
    df2 = df2[keep_cols_2].copy()

    merged = pd.merge(
        df1,
        df2,
        on=ID_COLUMNS,
        how="inner",
        suffixes=("_m1", "_m2")
    )

    if merged.empty:
        raise ValueError(f"{basename}: no matched rows after merging on {ID_COLUMNS}.")

    # Column parsing based on original common columns
    col_map = build_column_map(common_data_cols)

    # Determine all metrics and regions in file
    items = []
    for col in common_data_cols:
        hemi, region, metric = parse_region_and_metric(col)
        items.append((hemi, region, metric))

    metrics = reorder_metrics(sorted(set(x[2] for x in items)))
    regions = sorted(set(x[1] for x in items))

    results = []

    for metric in metrics:
        for region in regions:
            hemi_candidates = []
            for hemi in ["lh", "rh", "unknown"]:
                key = (hemi, region, metric)
                if key in col_map:
                    hemi_candidates.append(key)

            # Usually only one candidate exists within one file
            for hemi, region_name, metric_name in hemi_candidates:
                original_col = col_map[(hemi, region_name, metric_name)]
                col1 = f"{original_col}_m1"
                col2 = f"{original_col}_m2"

                x = pd.to_numeric(merged[col1], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(merged[col2], errors="coerce").to_numpy(dtype=float)

                stats = compute_pair_stats(x, y)

                results.append({
                    "file": basename,
                    "hemisphere": hemi,
                    "region": region_name,
                    "metric": metric_name,
                    "method1_column": col1,
                    "method2_column": col2,
                    **stats
                })

    res_df = pd.DataFrame(results)

    if res_df.empty:
        raise ValueError(f"{basename}: no analyzable region-metric pairs found.")

    # Matrices for heatmaps
    icc_matrix = res_df.pivot_table(index="metric", columns="region", values="ICC3_1", aggfunc="first")
    pearson_matrix = res_df.pivot_table(index="metric", columns="region", values="pearson_r", aggfunc="first")
    mad_matrix = res_df.pivot_table(index="metric", columns="region", values="mean_abs_difference", aggfunc="first")

    icc_matrix = icc_matrix.reindex(index=metrics)
    pearson_matrix = pearson_matrix.reindex(index=metrics)
    mad_matrix = mad_matrix.reindex(index=metrics)

    # Summary
    summary_rows = []
    for metric in metrics:
        sub = res_df[res_df["metric"] == metric]
        summary_rows.append({
            "file": basename,
            "metric": metric,
            "n_regions": int(sub["region"].nunique()),
            "mean_pearson_r": sub["pearson_r"].mean(),
            "median_pearson_r": sub["pearson_r"].median(),
            "mean_ICC3_1": sub["ICC3_1"].mean(),
            "median_ICC3_1": sub["ICC3_1"].median(),
            "mean_abs_difference": sub["mean_abs_difference"].mean(),
            "median_abs_difference": sub["mean_abs_difference"].median(),
            "n_ICC_gt_0_90": int((sub["ICC3_1"] > 0.90).sum()),
            "n_0_75_lt_ICC_le_0_90": int(((sub["ICC3_1"] > 0.75) & (sub["ICC3_1"] <= 0.90)).sum()),
            "n_ICC_le_0_75": int((sub["ICC3_1"] <= 0.75).sum())
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save Excel
    out_xlsx = os.path.join(output_dir, f"{stem}_agreement.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name="region_metric_stats", index=False)
        summary_df.to_excel(writer, sheet_name="metric_summary", index=False)
        icc_matrix.to_excel(writer, sheet_name=safe_sheet_name("ICC3_1_heatmap_matrix"))
        pearson_matrix.to_excel(writer, sheet_name=safe_sheet_name("Pearson_heatmap_matrix"))
        mad_matrix.to_excel(writer, sheet_name=safe_sheet_name("MAD_heatmap_matrix"))

    # Save heatmaps
    icc_png = os.path.join(output_dir, f"{stem}_ICC3_1_heatmap.png")
    pearson_png = os.path.join(output_dir, f"{stem}_Pearson_r_heatmap.png")
    mad_png = os.path.join(output_dir, f"{stem}_MAD_heatmap.png")

    make_heatmap(
        matrix_df=icc_matrix,
        title=f"{basename} - ICC(3,1)",
        out_png=icc_png,
        value_name="ICC(3,1)",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0
    )

    make_heatmap(
        matrix_df=pearson_matrix,
        title=f"{basename} - Pearson r",
        out_png=pearson_png,
        value_name="Pearson r",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0
    )

    make_heatmap(
        matrix_df=mad_matrix,
        title=f"{basename} - Mean absolute difference",
        out_png=mad_png,
        value_name="Mean absolute difference",
        cmap="magma",
        vmin=0.0,
        vmax=np.nanpercentile(mad_matrix.to_numpy(dtype=float), 95)
    )

    primary_matrix_map = {
        "ICC3_1": icc_matrix,
        "pearson_r": pearson_matrix,
        "mean_abs_diff": mad_matrix
    }
    primary_cmap_map = {
        "ICC3_1": ("viridis", 0.0, 1.0, "ICC(3,1)"),
        "pearson_r": ("viridis", 0.0, 1.0, "Pearson r"),
        "mean_abs_diff": ("magma", 0.0, np.nanpercentile(mad_matrix.to_numpy(dtype=float), 95), "Mean absolute difference")
    }

    primary_png = os.path.join(output_dir, f"{stem}_primary_heatmap.png")
    primary_matrix = primary_matrix_map[PRIMARY_HEATMAP_METRIC]
    cmap, vmin, vmax, label = primary_cmap_map[PRIMARY_HEATMAP_METRIC]
    make_heatmap(
        matrix_df=primary_matrix,
        title=f"{basename} - {PRIMARY_HEATMAP_METRIC}",
        out_png=primary_png,
        value_name=label,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    return {
        "file": basename,
        "n_subject_rows_method1": len(df1),
        "n_subject_rows_method2": len(df2),
        "n_merged_rows": len(merged),
        "n_common_data_columns": len(common_data_cols),
        "n_metrics": len(metrics),
        "n_regions": len(regions),
        "output_xlsx": out_xlsx,
        "output_primary_heatmap": primary_png,
        "mean_ICC3_1_all": res_df["ICC3_1"].mean(),
        "median_ICC3_1_all": res_df["ICC3_1"].median(),
        "mean_pearson_r_all": res_df["pearson_r"].mean(),
        "median_pearson_r_all": res_df["pearson_r"].median(),
        "mean_abs_difference_all": res_df["mean_abs_difference"].mean(),
        "median_abs_difference_all": res_df["mean_abs_difference"].median(),
    }


def main():
    ensure_dir(OUTPUT_DIR)

    common_files = list_common_csv_files(DIR_METHOD1, DIR_METHOD2)
    if len(common_files) == 0:
        raise FileNotFoundError("No common CSV files were found in the two directories.")

    print(f"Found {len(common_files)} common CSV files.")

    report_rows = []

    for fname in common_files:
        file1 = os.path.join(DIR_METHOD1, fname)
        file2 = os.path.join(DIR_METHOD2, fname)

        print(f"Analyzing: {fname}")
        try:
            report = analyze_one_file(file1, file2, OUTPUT_DIR)
            report_rows.append(report)
            print(f"  Done: {fname}")
        except Exception as e:
            print(f"  Failed: {fname}")
            print(f"  Reason: {e}")
            report_rows.append({
                "file": fname,
                "error": str(e)
            })

    report_df = pd.DataFrame(report_rows)
    report_csv = os.path.join(OUTPUT_DIR, "agreement_analysis_report.csv")
    report_xlsx = os.path.join(OUTPUT_DIR, "agreement_analysis_report.xlsx")

    report_df.to_csv(report_csv, index=False)
    report_df.to_excel(report_xlsx, index=False)

    print("\nAll finished.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report CSV: {report_csv}")
    print(f"Report Excel: {report_xlsx}")


if __name__ == "__main__":
    main()