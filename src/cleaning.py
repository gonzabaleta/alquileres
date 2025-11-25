import pandas as pd
import numpy as np
from src.utils import get_existing_columns
from typing import List, Dict, NotRequired, TypedDict


# --- Configuration Object Definition ---
class CleaningConfig(TypedDict):
    """
    A TypedDict to define the structure of the cleaning configuration object.
    All keys are optional.
    """

    cols_to_drop: NotRequired[List[str]]
    bool_cols: NotRequired[List[str]]
    amenities_to_combine: NotRequired[Dict[str, str]]
    age_col: NotRequired[str]
    date_features: NotRequired[Dict[str, str]]
    negative_val_cols: NotRequired[List[str]]
    zero_surface_cols: NotRequired[List[str]]
    zero_room_cols: NotRequired[List[str]]
    room_inconsistency_cols: NotRequired[Dict[str, str]]
    surface_inconsistency_cols: NotRequired[Dict[str, str]]


def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drops specified columns from the DataFrame."""
    found_cols = get_existing_columns(df, cols)
    if not found_cols:
        return df
    print(f"Dropping columns: {', '.join(found_cols)}")
    df = df.drop(columns=found_cols)
    return df


def unify_boolean_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Unifies values in specified columns to True, False, or pd.NA."""
    true_values = ["1", "1.0", "sí", "si", "yes"]
    false_values = ["0", "0.0", "no"]
    found_cols = get_existing_columns(df, cols)
    if not found_cols:
        return df
    print(f"Unifying columns to boolean format: {', '.join(found_cols)}")
    for col in found_cols:
        new_col = pd.Series(index=df.index, dtype="boolean")
        original_col_str = df[col].astype(str).str.strip().str.lower()
        truthy_mask = original_col_str.isin(true_values)
        falsy_mask = original_col_str.isin(false_values)
        new_col[truthy_mask] = True
        new_col[falsy_mask] = False
        new_col[df[col].isnull()] = pd.NA
        df[col] = new_col
    return df


def combine_amenities(
    df: pd.DataFrame, col1: str, col2: str, new_col_name: str
) -> pd.DataFrame:
    """Combines two boolean amenity columns into a new one using OR logic."""
    if col1 not in df.columns or col2 not in df.columns:
        print(
            f"Warning: Cannot combine amenities. One or both columns not found: {col1}, {col2}"
        )
        return df
    print(f"Combining '{col1}' and '{col2}' into '{new_col_name}'...")
    df[new_col_name] = df[col1] | df[col2]
    df = df.drop(columns=[col1, col2])
    return df


def convert_age_to_numeric(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts the age column to a numeric type."""
    if col_name not in df.columns:
        print(f"Warning: Column '{col_name}' not found for age conversion.")
        return df
    print(f"Converting column '{col_name}' to numeric...")
    col_data = df[col_name].copy()
    a_estrenar_mask = col_data.str.lower() == "a estrenar"
    col_data.loc[a_estrenar_mask.fillna(False)] = "0"
    # Use a raw string (r"...") for the regex pattern to avoid SyntaxWarning
    df[col_name] = pd.to_numeric(
        col_data.str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    return df


def engineer_date_features(
    df: pd.DataFrame, date_col: str, start_date_str: str
) -> pd.DataFrame:
    """Engineers new features from a date column and drops the original."""
    if date_col not in df.columns:
        print(f"Warning: Date column '{date_col}' not found for feature engineering.")
        return df
    print(f"Engineering features from date column '{date_col}'...")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    start_date = pd.to_datetime(start_date_str)
    df["meses_desde_comienzo"] = (df[date_col] - start_date).dt.days / 30.44
    df["mes_listing"] = df[date_col].dt.month
    df = df.drop(columns=[date_col])
    return df


def remove_negative_values(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replaces negative (impossible) values in specified columns with NaN."""
    found_cols = get_existing_columns(df, cols)
    if not found_cols:
        return df
    print(f"Replacing negative values with NaN in: {', '.join(found_cols)}")
    for col in found_cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        mask = numeric_col < 0
        if mask.any():
            print(f"  - Found {mask.sum()} negative values in '{col}'.")
            df.loc[mask, col] = np.nan
    return df


def handle_zero_surface_values(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replaces 0 values in surface columns with NaN."""
    found_cols = get_existing_columns(df, cols)
    if not found_cols:
        return df
    print(f"Replacing 0-value surfaces with NaN in: {', '.join(found_cols)}")
    for col in found_cols:
        mask = df[col] == 0
        if mask.any():
            print(f"  - Found {mask.sum()} zero values in '{col}'.")
            df.loc[mask, col] = np.nan
    return df


def handle_zero_room_values(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replaces 0 values in room-related columns with NaN."""
    found_cols = get_existing_columns(df, cols)
    if not found_cols:
        return df
    print(f"Replacing 0-value rooms with NaN in: {', '.join(found_cols)}")
    for col in found_cols:
        mask = df[col] == 0
        if mask.any():
            print(f"  - Found {mask.sum()} zero values in '{col}'.")
            df.loc[mask, col] = np.nan
    return df


def handle_room_inconsistencies(
    df: pd.DataFrame, total_room_col: str, sub_room_col: str
) -> pd.DataFrame:
    """Replaces values with NaN where the sub-room count is greater than the total."""
    if total_room_col not in df.columns or sub_room_col not in df.columns:
        print(f"Warning: Columns '{total_room_col}' or '{sub_room_col}' not found.")
        return df
    print(
        f"Correcting inconsistencies between '{total_room_col}' and '{sub_room_col}'..."
    )
    mask = df[sub_room_col] > df[total_room_col]
    if mask.any():
        print(
            f"  - Found {mask.sum()} rows where '{sub_room_col}' > '{total_room_col}'. Replacing '{sub_room_col}' with NaN."
        )
        df.loc[mask, sub_room_col] = np.nan
    return df


def handle_surface_inconsistencies(
    df: pd.DataFrame, total_surface_col: str, constructed_surface_col: str
) -> pd.DataFrame:
    """Sets constructed surface to NaN where it's greater than the total surface."""
    if total_surface_col not in df.columns or constructed_surface_col not in df.columns:
        print(
            f"Warning: Columns '{total_surface_col}' or '{constructed_surface_col}' not found."
        )
        return df
    print(
        f"Correcting inconsistencies between '{total_surface_col}' and '{constructed_surface_col}'..."
    )
    mask = df[constructed_surface_col] > df[total_surface_col]
    if mask.any():
        print(
            f"  - Found {mask.sum()} rows where '{constructed_surface_col}' > '{total_surface_col}'. Replacing '{constructed_surface_col}' with NaN."
        )
        df.loc[mask, constructed_surface_col] = np.nan
    return df


def clean_data(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    """
    Applies a sequence of cleaning and feature engineering steps to the raw property data.
    """
    print("--- Starting Data Cleaning Pipeline ---")
    df_clean = df.copy()

    if "cols_to_drop" in config:
        df_clean = drop_columns(df_clean, config["cols_to_drop"])
    if "bool_cols" in config:
        df_clean = unify_boolean_columns(df_clean, config["bool_cols"])
    if "amenities_to_combine" in config:
        amenities_config = config["amenities_to_combine"]
        df_clean = combine_amenities(
            df_clean,
            amenities_config["col1"],
            amenities_config["col2"],
            amenities_config["new_name"],
        )
    if "age_col" in config:
        df_clean = convert_age_to_numeric(df_clean, config["age_col"])
    if "date_features" in config:
        date_config = config["date_features"]
        df_clean = engineer_date_features(
            df_clean, date_config["date_col"], date_config["start_date"]
        )
    if "negative_val_cols" in config:
        df_clean = remove_negative_values(df_clean, config["negative_val_cols"])
    if "zero_surface_cols" in config:
        df_clean = handle_zero_surface_values(df_clean, config["zero_surface_cols"])
    if "zero_room_cols" in config:
        df_clean = handle_zero_room_values(df_clean, config["zero_room_cols"])
    if "room_inconsistency_cols" in config:
        room_config = config["room_inconsistency_cols"]
        df_clean = handle_room_inconsistencies(
            df_clean, room_config["total"], room_config["sub"]
        )
    if "surface_inconsistency_cols" in config:
        surface_config = config["surface_inconsistency_cols"]
        df_clean = handle_surface_inconsistencies(
            df_clean, surface_config["total"], surface_config["constructed"]
        )

    print("--- Data Cleaning Pipeline Finished ---")
    return df_clean
