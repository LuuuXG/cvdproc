from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

_FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


# -----------------------------------------------------------------------------
# Table header detection: support both "# ColHeaders ..." and "# TableCol ..."
# -----------------------------------------------------------------------------
def _get_colheaders_from_colheaders(lines: List[str]) -> Optional[List[str]]:
    for line in reversed(lines):
        if line.startswith("#") and re.match(r"^#\s*ColHeaders\s+", line):
            cols = re.split(r"\s+", re.sub(r"^#\s*ColHeaders\s+", "", line).strip())
            return cols if cols else None
    return None


def _get_colheaders_from_tablecol(lines: List[str]) -> Optional[List[str]]:
    col_headers: Dict[int, str] = {}
    for line in lines:
        if not line.startswith("#"):
            continue
        s = line.lstrip("#").strip()
        if not s.startswith("TableCol"):
            continue
        parts = s.split()
        if len(parts) >= 4 and parts[2] == "ColHeader":
            try:
                idx = int(parts[1])
                col_headers[idx] = parts[3]
            except ValueError:
                continue
    if not col_headers:
        return None
    return [col_headers[i] for i in sorted(col_headers.keys())]


def _get_table_columns(lines: List[str]) -> List[str]:
    cols = _get_colheaders_from_colheaders(lines)
    if cols is not None:
        return cols
    cols = _get_colheaders_from_tablecol(lines)
    if cols is not None:
        return cols
    raise ValueError("Cannot find table headers: expected '# ColHeaders' or '# TableCol ... ColHeader ...'.")


# -----------------------------------------------------------------------------
# Generic stats table reader (aparc/aseg/wmparc/w-g.pct)
# -----------------------------------------------------------------------------
def read_stats_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cols = _get_table_columns(lines)

    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        sep=r"\s+",
        engine="python",
        names=cols,
    )

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    return df


# -----------------------------------------------------------------------------
# brainvol.stats reader (long format; name uses the FIRST name field)
# -----------------------------------------------------------------------------
def read_brainvol_stats_long(path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse brainvol.stats into a long table with one row per '# Measure' line.

    Output columns:
      - name: first field after 'Measure' (e.g., BrainSeg, SupraTentorial)
      - value
      - unit
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    meas_re = re.compile(
        rf"^#\s*Measure\s+([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*({_FLOAT_RE})\s*,\s*(.+)\s*$"
    )

    rows: List[Dict[str, Union[str, float]]] = []
    for line in lines:
        if not line.startswith("#"):
            continue
        m = meas_re.match(line)
        if not m:
            continue

        first_name = m.group(1).strip()
        value = float(m.group(4))
        unit = m.group(5).strip()

        rows.append({"name": first_name, "value": value, "unit": unit})

    if not rows:
        raise ValueError(f"No '# Measure' lines found in: {path}")

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Post-processing: drop columns and reorder
# -----------------------------------------------------------------------------
def drop_columns_if_exist(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    existing = [c for c in cols_to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def move_column_to_first(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    cols = [col] + [c for c in df.columns if c != col]
    return df.loc[:, cols]


def postprocess_table(
    df: pd.DataFrame,
    drop_index_segid: bool = False,
    structname_first: bool = False,
) -> pd.DataFrame:
    if drop_index_segid:
        df = drop_columns_if_exist(df, ["Index", "SegId"])
    if structname_first:
        df = move_column_to_first(df, "StructName")
    return df


# -----------------------------------------------------------------------------
# Export utilities (one stats file -> one CSV and/or one XLSX)
# -----------------------------------------------------------------------------
def export_table(
    df: pd.DataFrame,
    out_csv: Optional[Union[str, Path]] = None,
    out_xlsx: Optional[Union[str, Path]] = None,
    sheet_name: str = "table",
) -> None:
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if out_xlsx is not None:
        out_xlsx = Path(out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)


# -----------------------------------------------------------------------------
# Dispatcher: 5 file types
# -----------------------------------------------------------------------------
def read_any_of_five(path: Union[str, Path]) -> Tuple[str, pd.DataFrame]:
    path = Path(path)
    name = path.name.lower()

    if name.endswith("brainvol.stats"):
        return "brainvol", read_brainvol_stats_long(path)

    if name.endswith("aseg.stats"):
        df = read_stats_table(path)
        df = postprocess_table(df, drop_index_segid=True, structname_first=True)
        return "aseg", df

    if name.endswith("wmparc.stats"):
        df = read_stats_table(path)
        df = postprocess_table(df, drop_index_segid=True, structname_first=True)
        return "wmparc", df

    if "w-g.pct.stats" in name:
        df = read_stats_table(path)
        df = postprocess_table(df, drop_index_segid=True, structname_first=True)
        return "wg_pct", df

    if name.endswith(".aparc.stats") or "aparc.a2009s.stats" in name:
        df = read_stats_table(path)
        # Keep aparc intact by default (no dropping/reordering requested)
        return "aparc", df

    raise ValueError(f"Unknown stats type: {path.name}")


# -----------------------------------------------------------------------------
# Example usage: each stats file -> CSV and/or XLSX
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    inputs = [
        "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline/stats/lh.aparc.stats",
        "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline/stats/aseg.stats",
        "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline/stats/wmparc.stats",
        "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline/stats/lh.w-g.pct.stats",
        "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline/stats/brainvol.stats",
    ]

    output_dir = Path(
        "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline"
    )

    out_csv_dir = output_dir / "fs_stats_csv"
    out_xlsx_dir = output_dir / "fs_stats_xlsx"

    # User-controlled outputs:
    # - Set any of these to None if you want to skip that format globally.
    write_csv = True
    write_xlsx = True

    for p in inputs:
        p = Path(p)
        kind, df = read_any_of_five(p)

        out_csv = (out_csv_dir / f"{p.stem}.csv") if write_csv else None
        out_xlsx = (out_xlsx_dir / f"{p.stem}.xlsx") if write_xlsx else None

        export_table(df, out_csv=out_csv, out_xlsx=out_xlsx, sheet_name="table")

        print(f"Exported {kind}: {p} -> rows={df.shape[0]}, cols={df.shape[1]}")
