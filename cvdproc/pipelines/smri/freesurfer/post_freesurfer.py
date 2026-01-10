from __future__ import annotations
import os
import re
import pandas as pd
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from cvdproc.config.paths import get_package_path

# -----------------------------------------------------------------------------
# FSQC
class FSQCInputSpec(CommandLineInputSpec):
    subjects_dir = Directory(exists=True, mandatory=True, argstr="%s", position=0, desc="Freesurfer SUBJECTS_DIR")
    subject_id = Str(mandatory=True, argstr="%s", position=1, desc="Subject ID")
    fsqc_output_dir = Str(mandatory=True, argstr="%s", position=2, desc="FSQC output directory")

class FSQCOutputSpec(TraitedSpec):
    fsqc_output_dir = Directory(desc='FSQC output directory')

class FSQC(CommandLine):
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'fsqc', 'fsqc_single.sh')
    input_spec = FSQCInputSpec
    output_spec = FSQCOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['fsqc_output_dir'] = os.path.abspath(self.inputs.fsqc_output_dir)
        return outputs

# -----------------------------------------------------------------------------
# stats2csv
_FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


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


def read_stats_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a FreeSurfer stats file containing a whitespace-delimited table.
    Supports both ColHeaders and TableCol header styles.
    """
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


def read_brainvol_stats_long(path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse brainvol.stats into a long table with one row per '# Measure' line.

    Output:
      name (first field), value, unit
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


def infer_hemi_from_filename(path: Union[str, Path]) -> Optional[str]:
    name = Path(path).name.lower()
    if ".lh." in name or name.startswith("lh."):
        return "lh"
    if ".rh." in name or name.startswith("rh."):
        return "rh"
    return None


def read_subseg_volume_only(path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse subsegmentation stats that look like:
      1 1 0 708.978202 Lateral-nucleus
    i.e., 5 columns: Index SegId NVoxels Volume_mm3 StructName
    Keep only StructName and Volume_mm3, add hemi if inferable.
    """
    path = Path(path)
    hemi = infer_hemi_from_filename(path)

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: List[Dict[str, Union[str, float]]] = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"\s+", s, maxsplit=4)
        if len(parts) < 5:
            continue

        vol_str = parts[3]
        name = parts[4].strip()

        try:
            vol = float(vol_str)
        except ValueError:
            continue

        row: Dict[str, Union[str, float]] = {"StructName": name, "Volume_mm3": vol}
        if hemi is not None:
            row["hemi"] = hemi
        rows.append(row)

    if not rows:
        raise ValueError(f"No data rows parsed from: {path}")

    df = pd.DataFrame(rows)

    if "hemi" in df.columns:
        df = df.loc[:, ["hemi", "StructName", "Volume_mm3"]]
    else:
        df = df.loc[:, ["StructName", "Volume_mm3"]]

    return df


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


def postprocess_segmentation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For aseg/wmparc/w-g.pct and similar segmentation-style tables:
      - drop Index and SegId if present
      - move StructName to the first column if present
    """
    df = drop_columns_if_exist(df, ["Index", "SegId"])
    df = move_column_to_first(df, "StructName")
    return df


def parse_stats_file(stats_path: Union[str, Path]) -> pd.DataFrame:
    """
    Dispatch parsing based on filename patterns.
    """
    stats_path = Path(stats_path)
    name = stats_path.name.lower()

    if name.endswith("brainvol.stats"):
        return read_brainvol_stats_long(stats_path)

    subseg_keywords = [
        "amygdalar-nuclei",
        "hipposubfields",
        "thalamic-nuclei",
        "brainstem",
        "hypothalamic",
    ]
    if any(k in name for k in subseg_keywords):
        return read_subseg_volume_only(stats_path)

    # Default: table-like stats
    df = read_stats_table(stats_path)

    # Apply your cleanup to specific file types
    if (
        name.endswith("aseg.stats")
        or name.endswith("wmparc.stats")
        or "w-g.pct.stats" in name
    ):
        df = postprocess_segmentation_table(df)

    return df


# -----------------------------------------------------------------------------
# Nipype interface
# -----------------------------------------------------------------------------
class Stats2CSVInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, mandatory=True, desc="Freesurfer SUBJECTS_DIR")
    subject_id = Str(mandatory=True, desc="Subject ID")
    output_dir = Directory(mandatory=True, desc="Output directory for CSV files")


class Stats2CSVOutputSpec(TraitedSpec):
    output_dir = Directory(desc="Output directory for CSV files")

class Stats2CSV(BaseInterface):
    input_spec = Stats2CSVInputSpec
    output_spec = Stats2CSVOutputSpec

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = str(self.inputs.subject_id)
        output_dir = Path(self.inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats_dir = subjects_dir / subject_id / "stats"
        if not stats_dir.exists():
            # Do not crash; just create output_dir and return
            return runtime

        base_stats_filenames = [
            "aseg.stats",
            "brainvol.stats",
            "lh.aparc.a2009s.stats",
            "rh.aparc.a2009s.stats",
            "lh.aparc.DKTatlas.stats",
            "rh.aparc.DKTatlas.stats",
            "lh.aparc.pial.stats",
            "rh.aparc.pial.stats",
            "lh.aparc.stats",
            "rh.aparc.stats",
            "lh.BA_exvivo.stats",
            "rh.BA_exvivo.stats",
            "lh.BA_exvivo.thresh.stats",
            "rh.BA_exvivo.thresh.stats",
            "lh.w-g.pct.stats",
            "rh.w-g.pct.stats",
            "wmparc.stats",
        ]

        amygdalar_stats_filenames = [
            "amygdalar-nuclei.lh.T1.v22.stats",
            "amygdalar-nuclei.rh.T1.v22.stats",
        ]

        brainstem_stats_filenames = [
            "brainstem.v13.stats",
        ]

        hippocampal_stats_filenames = [
            "hipposubfields.lh.T1.v22.stats",
            "hipposubfields.rh.T1.v22.stats",
        ]

        hypothalamic_stats_filenames = [
            "hypothalamic_subunits_volumes.v1.stats",
        ]

        thalamic_stats_filenames = [
            "thalamic-nuclei.lh.v13.T1.stats",
            "thalamic-nuclei.rh.v13.T1.stats",
        ]

        all_stats_filenames = (
            base_stats_filenames
            + amygdalar_stats_filenames
            + brainstem_stats_filenames
            + hippocampal_stats_filenames
            + hypothalamic_stats_filenames
            + thalamic_stats_filenames
        )

        for fname in all_stats_filenames:
            in_path = stats_dir / fname
            if not in_path.exists():
                # Skip missing files without error
                continue

            try:
                df = parse_stats_file(in_path)
            except Exception:
                # Skip files that fail to parse, without crashing the pipeline
                continue

            out_path = output_dir / f"{in_path.stem}.csv"
            try:
                df.to_csv(out_path, index=False)
            except Exception:
                # If write fails, skip
                continue

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_dir"] = str(Path(self.inputs.output_dir).resolve())
        return outputs

if __name__ == "__main__":
    # Example usage
    stats2csv = Stats2CSV()
    stats2csv.inputs.subjects_dir = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008"
    stats2csv.inputs.subject_id = "ses-baseline"
    stats2csv.inputs.output_dir = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0008/ses-baseline/stats"
    stats2csv.run()