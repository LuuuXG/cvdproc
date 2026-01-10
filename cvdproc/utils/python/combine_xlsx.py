import pandas as pd
import os
from typing import List, Optional, Dict, Union

def merge_and_save_excel(
    base_file_path: str,
    additional_file_path: str,
    match_column_base: List[str],
    match_column_additional: List[str],
    selected_columns: Union[List[str], str],
    prefix: str = "",
    output_file_path: Optional[str] = None,
    zfill_columns: Optional[Dict[str, int]] = None
):
    """
    Merge two Excel files based on matching columns.
    - Matching is always done on STRING keys (temporary key columns).
    - Original columns in both dataframes are NOT modified.
    - Additional selected columns are added with an optional prefix.
    - Original base column order is preserved.
    """

    base_df = pd.read_excel(base_file_path)
    additional_df = pd.read_excel(additional_file_path)

    # Save original base column order
    original_order = base_df.columns.tolist()

    if len(match_column_base) != len(match_column_additional):
        raise ValueError("Matching column lists must be of the same length.")

    # ------------------------------------------------------------------
    # 1) Build string key columns for matching (do NOT modify originals)
    # ------------------------------------------------------------------
    base_key_cols = []
    add_key_cols = []

    for i, (col_base, col_add) in enumerate(zip(match_column_base, match_column_additional)):
        base_key = f"__key_base_{i}"
        add_key = f"__key_add_{i}"

        # Base side: convert to string key
        if zfill_columns and col_base in zfill_columns:
            base_df[base_key] = base_df[col_base].apply(
                lambda x: str(int(float(x))).zfill(zfill_columns[col_base]) if pd.notna(x) else ""
            )
        else:
            base_df[base_key] = base_df[col_base].astype(str).str.strip()

        # Additional side: convert to string key
        if zfill_columns and col_add in zfill_columns:
            additional_df[add_key] = additional_df[col_add].apply(
                lambda x: str(int(float(x))).zfill(zfill_columns[col_add]) if pd.notna(x) else ""
            )
        else:
            additional_df[add_key] = additional_df[col_add].astype(str).str.strip()

        base_key_cols.append(base_key)
        add_key_cols.append(add_key)

    # ------------------------------------------------------------------
    # 2) Handle selected columns from additional_df
    # ------------------------------------------------------------------
    if isinstance(selected_columns, str) and selected_columns != 'ALL_EXCEPT_MATCH':
        selected_columns = [selected_columns]

    if selected_columns == 'ALL_EXCEPT_MATCH':
        # 排除：匹配列 + 我们刚刚创建的临时 key 列
        selected_columns = [
            col for col in additional_df.columns
            if col not in match_column_additional and col not in add_key_cols
        ]

    # Columns to bring over from right dataframe: keys + selected columns
    columns_to_merge = add_key_cols + selected_columns
    additional_filtered = additional_df[columns_to_merge]

    # ------------------------------------------------------------------
    # 3) Merge using the temporary key columns (string-based match)
    # ------------------------------------------------------------------
    merged_df = pd.merge(
        base_df,
        additional_filtered,
        left_on=base_key_cols,
        right_on=add_key_cols,
        how='left',
        suffixes=('_base', '_right')
    )

    # ------------------------------------------------------------------
    # 4) Create new prefixed columns from additional data
    # ------------------------------------------------------------------
    for col in selected_columns:
        right_col = col  # in additional_filtered there is no suffix
        # After merge, columns from additional_filtered keep their names
        if right_col in merged_df.columns:
            new_name = prefix + col
            merged_df[new_name] = merged_df[right_col]

    # ------------------------------------------------------------------
    # 5) Clean up: drop raw right-side selected columns and key columns
    # ------------------------------------------------------------------
    # Drop the extra right-side copies of selected columns
    for col in selected_columns:
        if col in merged_df.columns and (prefix + col) != col:
            # Only drop if this is the original from right side
            merged_df.drop(columns=[col], inplace=True)

    # Drop temporary key columns
    for k in base_key_cols + add_key_cols:
        if k in merged_df.columns:
            merged_df.drop(columns=[k], inplace=True)

    # ------------------------------------------------------------------
    # 6) Restore original base column order, then append new columns
    # ------------------------------------------------------------------
    final_columns = []
    for col in original_order:
        if col in merged_df.columns:
            final_columns.append(col)

    for col in merged_df.columns:
        if col not in final_columns:
            final_columns.append(col)

    merged_df = merged_df[final_columns]

    # ------------------------------------------------------------------
    # 7) Output file
    # ------------------------------------------------------------------
    if output_file_path is None:
        base_name, ext = os.path.splitext(base_file_path)
        output_file_path = f"{base_name}_merged{ext}"

    merged_df.to_excel(output_file_path, index=False)
    print(f"✅ File saved to: {output_file_path}")

if __name__ == "__main__":
    merge_and_save_excel(
        base_file_path=r"E:\WPS_Cloud\1136007837\WPS云盘\paper\rssi_glymphatic_analysis\data\source\SVD_jobs_new.xlsx",
        additional_file_path=r"E:\WPS_Cloud\1136007837\WPS云盘\paper\rssi_glymphatic_analysis\data\source\pved_results.xlsx",
        match_column_base=["Subject", "Session"],
        match_column_additional=["Subject", "Session"],
        #selected_columns=["3D-T1w", "2D-FLAIR(tra)", "DTI", "SWI", "NODDI"],
        selected_columns='ALL_EXCEPT_MATCH',
        prefix="",  
        zfill_columns={},
    )
