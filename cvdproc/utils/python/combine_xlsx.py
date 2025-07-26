import pandas as pd
import os
from typing import List, Optional, Dict

def merge_and_save_excel(
    base_file_path: str,
    additional_file_path: str,
    match_column_base: List[str],
    match_column_additional: List[str],
    selected_columns: List[str] or str,
    prefix: str = "",
    output_file_path: Optional[str] = None,
    zfill_columns: Optional[Dict[str, int]] = None
):
    """
    Merge two Excel files based on matching columns, and always add additional columns with a prefix,
    without overwriting base file columns. Preserves original base column order.
    """

    base_df = pd.read_excel(base_file_path)
    additional_df = pd.read_excel(additional_file_path)

    # Save original base column order
    original_order = base_df.columns.tolist()

    # Normalize matching columns
    for col in match_column_base:
        if zfill_columns and col in zfill_columns:
            base_df[col] = base_df[col].apply(
                lambda x: str(int(float(x))).zfill(zfill_columns[col]) if pd.notna(x) else x
            )
        else:
            base_df[col] = base_df[col].astype(str).str.strip()

    for col in match_column_additional:
        if zfill_columns and col in zfill_columns:
            additional_df[col] = additional_df[col].apply(
                lambda x: str(int(float(x))).zfill(zfill_columns[col]) if pd.notna(x) else x
            )
        else:
            additional_df[col] = additional_df[col].astype(str).str.strip()

    if len(match_column_base) != len(match_column_additional):
        raise ValueError("Matching column lists must be of the same length.")

    # Handle selected columns
    if isinstance(selected_columns, str) and selected_columns != 'ALL_EXCEPT_MATCH':
        selected_columns = [selected_columns]
    if selected_columns == 'ALL_EXCEPT_MATCH':
        selected_columns = [col for col in additional_df.columns if col not in match_column_additional]

    # Prepare merging
    columns_to_merge = match_column_additional + selected_columns
    additional_filtered = additional_df[columns_to_merge]

    # Merge with suffixes
    merged_df = pd.merge(
        base_df,
        additional_filtered,
        left_on=match_column_base,
        right_on=match_column_additional,
        how='left',
        suffixes=('_base', '_right')
    )

    # Always create new prefixed columns from additional data
    for col in selected_columns:
        right_col = col + "_right"
        if right_col in merged_df.columns:
            merged_df[prefix + col] = merged_df[right_col]
            merged_df.drop(columns=[right_col], inplace=True)

    # Remove right-side matching columns if any
    for col in match_column_additional:
        if col in merged_df.columns and col not in match_column_base:
            merged_df.drop(columns=[col], inplace=True)

    for col in match_column_base:
        right_name = col + "_right"
        if right_name in merged_df.columns:
            merged_df.drop(columns=[right_name], inplace=True)

    # Restore original column order + append new columns
    final_columns = []
    for col in original_order:
        if col in merged_df.columns:
            final_columns.append(col)
    for col in merged_df.columns:
        if col not in final_columns:
            final_columns.append(col)
    merged_df = merged_df[final_columns]

    # Output file name
    if output_file_path is None:
        base_name, ext = os.path.splitext(base_file_path)
        output_file_path = f"{base_name}_merged{ext}"

    merged_df.to_excel(output_file_path, index=False)
    print(f"✅ File saved to: {output_file_path}")


# Example usage
if __name__ == "__main__":
    merge_and_save_excel(
        base_file_path=r"E:\WPS_Cloud\1136007837\WPS云盘\paper\lesion_connection\data\rawdata.xlsx",
        additional_file_path=r"F:\BIDS\SVD_BIDS\derivatives\population\wmh_quantification\wmh_quantification_results.xlsx",
        match_column_base=["Subject_id", "Session_id"],
        match_column_additional=["Subject", "Session"],
        selected_columns=["ICV(ml)", "Total_WMH_percentICV(%)", "PWMH_percentICV(%)", "DWMH_percentICV(%)"],
        #selected_columns=["PSMD"],
        prefix="",  
        zfill_columns={}
    )
