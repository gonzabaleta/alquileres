import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, NotRequired, TypedDict

# --- Configuration Object Definition ---
class PreprocessingConfig(TypedDict):
    """
    A TypedDict to define the structure of the preprocessing configuration object.
    All keys are optional.
    """
    log_transform_cols: NotRequired[List[str]]
    cap_outliers_cols: NotRequired[Dict[str, List]] # e.g. {"cols": [...], "percentiles": (0.01, 0.99)}
    standardize_cols: NotRequired[List[str]]

# --- Preprocessing Functions ---

def log_transform_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Applies a log1p transformation to specified columns to reduce right-skewness."""
    df_processed = df.copy()
    print(f"Applying log transformation to columns: {', '.join(cols)}")
    for col in cols:
        if col in df_processed.columns:
            df_processed[col] = np.log1p(df_processed[col])
        else:
            print(f"Warning: Column '{col}' not found for log transformation. Skipping.")
    return df_processed

def cap_outliers(df: pd.DataFrame, cols: List[str], percentiles: Tuple[float, float]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Caps outliers in specified columns based on given percentiles.
    This function is meant to be used on the **training set**.
    """
    df_processed = df.copy()
    capping_params = {}
    print(f"Capping outliers for columns: {', '.join(cols)}")
    
    for col in cols:
        if col not in df_processed.columns:
            print(f"Warning: Column '{col}' not found for capping. Skipping.")
            continue
            
        lower_bound = df_processed[col].quantile(percentiles[0])
        upper_bound = df_processed[col].quantile(percentiles[1])
        
        capping_params[col] = {"lower": lower_bound, "upper": upper_bound}
        
        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
    return df_processed, capping_params

def apply_capping(df: pd.DataFrame, capping_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Applies pre-calculated capping to a DataFrame (e.g., test set)."""
    df_processed = df.copy()
    print("Applying pre-calculated outlier capping...")
    
    for col, params in capping_params.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].clip(lower=params['lower'], upper=params['upper'])
        else:
            print(f"Warning: Column '{col}' not found for applying capping. Skipping.")
            
    return df_processed

def standardize_features(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Standardizes specified numeric columns using Z-score scaling (for training set)."""
    df_processed = df.copy()
    scaling_params = {}
    print(f"Standardizing columns: {', '.join(cols)}")
    for col in cols:
        if col not in df_processed.columns:
            print(f"Warning: Column '{col}' not found for standardization. Skipping.")
            continue
        mean_val = df_processed[col].mean()
        std_val = df_processed[col].std()
        scaling_params[col] = {"mean": mean_val, "std": std_val}
        df_processed[col] = (df_processed[col] - mean_val) / std_val
    return df_processed, scaling_params

def apply_standardization(df: pd.DataFrame, scaling_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Applies a pre-calculated standardization to a DataFrame (e.g., test set)."""
    df_processed = df.copy()
    print("Applying pre-calculated standardization...")
    for col, params in scaling_params.items():
        if col not in df_processed.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue
        mean_val = params['mean']
        std_val = params['std']
        df_processed[col] = (df_processed[col] - mean_val) / std_val
    return df_processed

# --- Main Preprocessing Pipeline ---

def preprocess_features(df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict]:
    """Applies a sequence of preprocessing steps to the cleaned data (for training set)."""
    print("--- Starting Feature Preprocessing Pipeline ---")
    df_processed = df.copy()
    learned_params = {}

    if "log_transform_cols" in config:
        df_processed = log_transform_features(df_processed, config["log_transform_cols"])
    if "cap_outliers_cols" in config:
        cap_config = config["cap_outliers_cols"]
        df_processed, capping_params = cap_outliers(df_processed, cap_config["cols"], cap_config["percentiles"])
        learned_params["capping"] = capping_params
    if "standardize_cols" in config:
        df_processed, scaling_params = standardize_features(df_processed, config["standardize_cols"])
        learned_params["scaling"] = scaling_params

    print("--- Feature Preprocessing Pipeline Finished ---")
    return df_processed, learned_params

def apply_preprocessing(df: pd.DataFrame, config: PreprocessingConfig, learned_params: Dict) -> pd.DataFrame:
    """Applies learned preprocessing steps to new data (e.g., test set)."""
    print("--- Applying Learned Preprocessing to New Data ---")
    df_processed = df.copy()

    if "log_transform_cols" in config:
        df_processed = log_transform_features(df_processed, config["log_transform_cols"])
    if "cap_outliers_cols" in config and "capping" in learned_params:
        df_processed = apply_capping(df_processed, learned_params["capping"])
    if "standardize_cols" in config and "scaling" in learned_params:
        df_processed = apply_standardization(df_processed, learned_params["scaling"])

    print("--- Preprocessing Application Finished ---")
    return df_processed
