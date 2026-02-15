import pandas as pd
import os
from typing import List, Optional, Dict, Union, Literal


def merge_and_save_excel(
    base_file_path: str,
    additional_file_path: str,
    match_column_base: List[str],
    match_column_additional: List[str],
    selected_columns: Union[List[str], str],
    prefix: str = "",
    output_file_path: Optional[str] = None,
    zfill_columns: Optional[Dict[str, int]] = None,
    update_same_name_mode: Literal["overwrite", "fillna", "keep_base"] = "fillna",
) -> None:
    """
    Merge two Excel files based on matching columns, then update base columns.

    Matching:
      - Always done on STRING keys (temporary key columns).
      - Original columns in both dataframes are NOT modified.

    Column behavior:
      - If a selected column (after applying `prefix`) already exists in base_df:
          overwrite: base <- additional (additional non-empty wins)
          fillna:    base is filled only where base is empty/NaN (do not overwrite)
          keep_base: keep base as-is, do not write into it
      - If not exists in base_df:
          create a new column and copy from additional.

    Notes:
      - "Empty" is defined as NaN or string after strip == "".
      - `prefix` is applied to target column name in base.
    """

    base_df = pd.read_excel(base_file_path)
    additional_df = pd.read_excel(additional_file_path)

    original_order = base_df.columns.tolist()

    if len(match_column_base) != len(match_column_additional):
        raise ValueError("Matching column lists must be of the same length.")

    # 1) Build string key columns (do NOT modify originals)
    base_key_cols = []
    add_key_cols = []

    for i, (col_base, col_add) in enumerate(zip(match_column_base, match_column_additional)):
        base_key = f"__key_base_{i}"
        add_key = f"__key_add_{i}"

        if zfill_columns and col_base in zfill_columns:
            base_df[base_key] = base_df[col_base].apply(
                lambda x: str(int(float(x))).zfill(zfill_columns[col_base]) if pd.notna(x) else ""
            )
        else:
            base_df[base_key] = base_df[col_base].astype(str).str.strip()

        if zfill_columns and col_add in zfill_columns:
            additional_df[add_key] = additional_df[col_add].apply(
                lambda x: str(int(float(x))).zfill(zfill_columns[col_add]) if pd.notna(x) else ""
            )
        else:
            additional_df[add_key] = additional_df[col_add].astype(str).str.strip()

        base_key_cols.append(base_key)
        add_key_cols.append(add_key)

    # 2) Determine selected columns from additional_df
    if isinstance(selected_columns, str) and selected_columns != "ALL_EXCEPT_MATCH":
        selected_columns = [selected_columns]

    if selected_columns == "ALL_EXCEPT_MATCH":
        selected_columns = [
            col for col in additional_df.columns
            if col not in match_column_additional and col not in add_key_cols
        ]

    # 3) Avoid name collisions during merge by adding a temporary prefix to right-side columns
    tmp_right_prefix = "__right__"
    right_rename_map = {c: f"{tmp_right_prefix}{c}" for c in selected_columns}
    additional_tmp = additional_df[add_key_cols + selected_columns].rename(columns=right_rename_map)

    merged_df = pd.merge(
        base_df,
        additional_tmp,
        left_on=base_key_cols,
        right_on=add_key_cols,
        how="left",
    )

    def is_empty_series(s: pd.Series) -> pd.Series:
        # Empty if NaN OR (string after strip == "")
        # For non-strings, only NaN is treated as empty
        is_na = s.isna()
        as_str = s.astype(str)
        is_blank_str = (~is_na) & as_str.str.strip().eq("")
        return is_na | is_blank_str

    # 4) Update / create columns in base according to mode
    for col in selected_columns:
        right_col = f"{tmp_right_prefix}{col}"
        if right_col not in merged_df.columns:
            continue

        target_col = f"{prefix}{col}" if prefix else col

        # If target does not exist, create it
        if target_col not in merged_df.columns:
            merged_df[target_col] = merged_df[right_col]
            continue

        if update_same_name_mode == "keep_base":
            continue

        right_vals = merged_df[right_col]
        right_empty = is_empty_series(right_vals)

        if update_same_name_mode == "overwrite":
            # overwrite only where right is non-empty; keep base where right is empty
            merged_df.loc[~right_empty, target_col] = right_vals.loc[~right_empty]
        elif update_same_name_mode == "fillna":
            base_empty = is_empty_series(merged_df[target_col])
            to_fill = base_empty & (~right_empty)
            merged_df.loc[to_fill, target_col] = right_vals.loc[to_fill]
        else:
            raise ValueError("update_same_name_mode must be one of: overwrite, fillna, keep_base")

    # 5) Cleanup key columns and temp right columns
    drop_cols = []
    drop_cols.extend(base_key_cols)
    drop_cols.extend(add_key_cols)
    drop_cols.extend([f"{tmp_right_prefix}{c}" for c in selected_columns])

    for c in drop_cols:
        if c in merged_df.columns:
            merged_df.drop(columns=[c], inplace=True)

    # 6) Restore original base order, then append newly created columns
    final_columns = [c for c in original_order if c in merged_df.columns]
    for c in merged_df.columns:
        if c not in final_columns:
            final_columns.append(c)
    merged_df = merged_df[final_columns]

    # 7) Output
    if output_file_path is None:
        base_name, ext = os.path.splitext(base_file_path)
        output_file_path = f"{base_name}_merged{ext}"

    merged_df.to_excel(output_file_path, index=False)
    print(f"File saved to: {output_file_path}")


if __name__ == "__main__":
    merge_and_save_excel(
        base_file_path=r"D:\WYJ\数据表\提取_merged.xlsx",
        additional_file_path=r"D:\WYJ\数据表\HC.xlsx",
        match_column_base=["ID"],
        match_column_additional=["受试者编号"],
        selected_columns="ALL_EXCEPT_MATCH",
        prefix="",
        zfill_columns={},
        update_same_name_mode="fillna",   # "overwrite" or "fillna" or "keep_base"
    )
