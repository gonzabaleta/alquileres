import pandas as pd
from typing import List, Dict, Tuple

def standardize_features(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Standardizes specified numeric columns in a DataFrame using Z-score scaling.

    For each column, it calculates the mean and standard deviation, then applies
    the formula: z = (x - mean) / std.

    This function is meant to be used on the **training set**.

    Args:
        df: The input DataFrame (e.g., training data).
        cols: A list of numeric column names to standardize.

    Returns:
        A tuple containing:
        - pd.DataFrame: The DataFrame with the specified columns standardized.
        - Dict[str, Dict[str, float]]: A dictionary containing the calculated 
          mean and std for each column, to be used for transforming test data.
          Example: {'col_name': {'mean': 10.5, 'std': 2.1}}
    """
    df_processed = df.copy()
    scaling_params = {}

    print(f"Standardizing columns: {', '.join(cols)}")

    for col in cols:
        if col not in df_processed.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        mean_val = df_processed[col].mean()
        std_val = df_processed[col].std()

        # Store the calculated parameters
        scaling_params[col] = {"mean": mean_val, "std": std_val}

        # Apply standardization
        df_processed[col] = (df_processed[col] - mean_val) / std_val

    return df_processed, scaling_params


def apply_standardization(df: pd.DataFrame, scaling_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Applies a pre-calculated standardization to a DataFrame.

    This function is meant to be used on the **validation or test set**, using
    the parameters learned from the training set.

    Args:
        df: The input DataFrame (e.g., test data).
        scaling_params: A dictionary with the mean and std for each column,
                        as returned by `standardize_features`.

    Returns:
        pd.DataFrame: The DataFrame with columns standardized using the provided parameters.
    """
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
