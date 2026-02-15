import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class FreesurferStatsExtractorMixin:
    # -----------------------------
    # Helpers (shared by cross/long)
    # -----------------------------
    @staticmethod
    def _safe_name(x: str) -> str:
        x = str(x).strip()
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", x)
        return x

    @staticmethod
    def _infer_hemi_from_filename(fname: str) -> Optional[str]:
        f = fname.lower()
        if f.startswith("lh.") or ".lh." in f:
            return "lh"
        if f.startswith("rh.") or ".rh." in f:
            return "rh"
        return None

    @staticmethod
    def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    @staticmethod
    def _infer_struct_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["StructName", "Structure", "LabelName", "Name", "ROI", "Region"]:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _infer_volume_col(df: pd.DataFrame) -> Optional[str]:
        candidates = ["Volume_mm3", "volume_mm3", "Volume", "volume"]
        for c in candidates:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                return c
        for c in df.columns:
            if "vol" in str(c).lower() and pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    @staticmethod
    def _stable_union(existing: List[str], incoming: List[str]) -> List[str]:
        seen = set(existing)
        out = list(existing)
        for k in incoming:
            if k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def _wide_from_roi_table(
        self,
        df: pd.DataFrame,
        prefix: str,
        struct_col: Optional[str],
        drop_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Build a wide dict from an ROI table.

        Ordering requirement:
        - Metric-major
        - Within each metric, preserve ROI order as in the input CSV (df row order)
        - Preserve metric order as in the input CSV (df column order)
        """
        if df is None or df.empty:
            return {}

        df = df.copy()

        if drop_cols:
            for c in drop_cols:
                if c in df.columns:
                    df = df.drop(columns=[c])

        if struct_col is None:
            struct_col = self._infer_struct_col(df)
        if struct_col is None or struct_col not in df.columns:
            return {}

        df[struct_col] = df[struct_col].astype(str).map(self._safe_name)

        exclude = {struct_col, "Index", "SegId"}
        value_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not value_cols:
            return {}

        # Preserve ROI order from df row order
        roi_names: List[str] = df[struct_col].tolist()

        out: Dict[str, float] = {}

        # Metric-major: iterate metrics first (in df column order), then ROIs (in df row order)
        for m in value_cols:
            metric_name = self._safe_name(m)
            col_vals = pd.to_numeric(df[m], errors="coerce").tolist()
            for roi, v in zip(roi_names, col_vals):
                if pd.isna(v):
                    continue
                key = f"{prefix}_{roi}_{metric_name}"
                out[key] = float(v)

        return out

    def _wide_volume_only(self, df: pd.DataFrame, prefix: str, fname: str) -> Dict[str, float]:
        """
        Volume-only files:
        - Preserve ROI order as in the input CSV (df row order)
        """
        if df is None or df.empty:
            return {}

        df = df.copy()

        for c in ["Index", "SegId"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        struct_col = self._infer_struct_col(df)
        vol_col = self._infer_volume_col(df)
        if struct_col is None or vol_col is None:
            return {}

        df[struct_col] = df[struct_col].astype(str).map(self._safe_name)
        vol = pd.to_numeric(df[vol_col], errors="coerce")

        out: Dict[str, float] = {}
        for name, v in zip(df[struct_col].tolist(), vol.tolist()):
            if pd.isna(v):
                continue
            out[f"{prefix}_{name}_Volume_mm3"] = float(v)

        hemi = self._infer_hemi_from_filename(fname)
        if hemi is not None:
            out[f"{prefix}_hemi"] = hemi

        return out

    def _wide_brainvol(self, df: pd.DataFrame, prefix: str) -> Dict[str, float]:
        """
        brainvol:
        - Preserve row order in the input CSV
        """
        if df is None or df.empty:
            return {}

        if "name" not in df.columns or "value" not in df.columns:
            return {}

        df = df.copy()
        df["name"] = df["name"].astype(str).map(self._safe_name)
        vals = pd.to_numeric(df["value"], errors="coerce")

        out: Dict[str, float] = {}
        for n, v in zip(df["name"].tolist(), vals.tolist()):
            if pd.isna(v):
                continue
            out[f"{prefix}_{n}"] = float(v)

        return out

    # ---------------------------------------
    # Shared streaming writers for summary csv
    # ---------------------------------------
    def _extract_merge_stats_dirs(
        self,
        stats_dir_items: List[Dict[str, str]],
        output_path: str,
    ) -> None:
        os.makedirs(output_path, exist_ok=True)
        out_dir = Path(output_path)

        cortical_files = [
            "lh.aparc.csv",
            "rh.aparc.csv",
            "lh.aparc.a2009s.csv",
            "rh.aparc.a2009s.csv",
            "lh.aparc.DKTatlas.csv",
            "rh.aparc.DKTatlas.csv",
            "lh.aparc.pial.csv",
            "rh.aparc.pial.csv",
            "lh.BA_exvivo.csv",
            "rh.BA_exvivo.csv",
            "lh.BA_exvivo.thresh.csv",
            "rh.BA_exvivo.thresh.csv",
            "lh.w-g.pct.csv",
            "rh.w-g.pct.csv",
        ]

        volume_only_files = [
            "aseg.csv",
            "wmparc.csv",
            "amygdalar-nuclei.lh.T1.v22.csv",
            "amygdalar-nuclei.rh.T1.v22.csv",
            "brainstem.v13.csv",
            "hipposubfields.lh.T1.v22.csv",
            "hipposubfields.rh.T1.v22.csv",
            "hypothalamic_subunits_volumes.v1.csv",
            "thalamic-nuclei.lh.v13.T1.csv",
            "thalamic-nuclei.rh.v13.T1.csv",
        ]

        brainvol_file = "brainvol.csv"

        writers: Dict[str, Dict[str, object]] = {}
        seen: Dict[str, set] = {}

        def _open_writer(filekey: str, out_path: Path, fieldnames: List[str], is_cortical: bool) -> csv.DictWriter:
            out_fh = open(out_path, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            writers[filekey] = {
                "fh": out_fh,
                "writer": w,
                "fieldnames": fieldnames,
                "is_cortical": is_cortical,
                "path": out_path,
            }
            seen[filekey] = set()
            return w

        def _close_all():
            for v in writers.values():
                try:
                    v["fh"].close()
                except Exception:
                    pass

        def _rewrite_with_expanded_header(filekey: str, new_fieldnames: List[str]):
            info = writers[filekey]
            out_path: Path = info["path"]

            old_rows: List[Dict[str, object]] = []
            try:
                if out_path.exists():
                    old_df = pd.read_csv(out_path)
                    old_rows = old_df.to_dict(orient="records")
            except Exception:
                old_rows = []

            try:
                info["fh"].close()
            except Exception:
                pass

            out_fh = open(out_path, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(out_fh, fieldnames=new_fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in old_rows:
                w.writerow(r)

            writers[filekey] = {
                "fh": out_fh,
                "writer": w,
                "fieldnames": new_fieldnames,
                "is_cortical": info["is_cortical"],
                "path": out_path,
            }

        def _ensure_writer_and_write_row(filekey: str, out_path: Path, row: Dict[str, object], is_cortical: bool):
            key_pair = (row.get("subject"), row.get("session"))
            if key_pair[0] is None or key_pair[1] is None:
                return

            if filekey in seen and key_pair in seen[filekey]:
                return

            # IMPORTANT: do not sort feature keys
            # Keep insertion order from `row` (which is built from df row/col order)
            feat_keys = [k for k in row.keys() if k not in ("subject", "session")]
            fieldnames = ["subject", "session"] + feat_keys

            if filekey not in writers:
                _open_writer(filekey, out_path, fieldnames, is_cortical)
            else:
                existing = writers[filekey]["fieldnames"]
                existing_feats = existing[2:]

                # Stable expansion: keep existing order, append new keys at the end
                new_feats = self._stable_union(existing_feats, feat_keys)
                new_fieldnames = ["subject", "session"] + new_feats

                if new_fieldnames != existing:
                    _rewrite_with_expanded_header(filekey, new_fieldnames)

            writers[filekey]["writer"].writerow(row)
            seen[filekey].add(key_pair)

        try:
            for item in stats_dir_items:
                subject = item["subject"]
                session = item["session"]
                stats_dir = item["stats_dir"]

                if not os.path.isdir(stats_dir):
                    continue

                # cortical
                for fname in cortical_files:
                    fpath = os.path.join(stats_dir, fname)
                    df = self._read_csv_if_exists(fpath)
                    if df is None:
                        continue

                    prefix = self._safe_name(Path(fname).stem)
                    row: Dict[str, object] = {"subject": subject, "session": session}
                    row.update(
                        self._wide_from_roi_table(
                            df=df,
                            prefix=prefix,
                            struct_col=self._infer_struct_col(df),
                            drop_cols=["Index", "SegId"],
                        )
                    )
                    if len(row) <= 2:
                        continue

                    out_path = out_dir / f"{Path(fname).stem}_summary.csv"
                    _ensure_writer_and_write_row(fname, out_path, row, True)

                # volume only
                for fname in volume_only_files:
                    fpath = os.path.join(stats_dir, fname)
                    df = self._read_csv_if_exists(fpath)
                    if df is None:
                        continue

                    prefix = self._safe_name(Path(fname).stem)
                    row = {"subject": subject, "session": session}
                    row.update(self._wide_volume_only(df=df, prefix=prefix, fname=fname))
                    if len(row) <= 2:
                        continue

                    out_path = out_dir / f"{Path(fname).stem}_summary.csv"
                    _ensure_writer_and_write_row(fname, out_path, row, False)

                # brainvol
                fpath = os.path.join(stats_dir, brainvol_file)
                df = self._read_csv_if_exists(fpath)
                if df is not None:
                    row = {"subject": subject, "session": session}
                    row.update(self._wide_brainvol(df=df, prefix="brainvol"))
                    if len(row) <= 2:
                        continue

                    out_path = out_dir / "brainvol_summary.csv"
                    _ensure_writer_and_write_row("brainvol", out_path, row, False)

        finally:
            _close_all()
