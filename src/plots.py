import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from typing import List


def plot_boolean_impact(
    df: pd.DataFrame, bool_cols: List[str], target_col: str, n_cols: int = 3
):
    """
    Generates a grid of boxplots to analyze the impact of boolean columns on a target variable.

    For each boolean column, it plots the distribution of the target variable for
    True, False, and NaN/Unknown values.

    Args:
        df: The input DataFrame.
        bool_cols: A list of boolean column names to analyze.
        target_col: The name of the target variable.
        n_cols: The number of columns for the subplot grid.
    """

    n_rows = math.ceil(len(bool_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()
    log_target = np.log1p(df[target_col])

    for i, col in enumerate(bool_cols):
        ax = axes[i]
        
        # --- CORRECTED LOGIC ---
        # 1. Convert the strict 'boolean' type to a generic 'object' type that allows any value.
        # 2. Fill the pd.NA values with the string 'Unknown'.
        plot_series = df[col].astype("object").fillna("Unknown")

        # Define the order for the plot
        plot_order = [True, False, "Unknown"]

        sns.boxplot(x=plot_series, y=log_target, ax=ax, order=plot_order)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel(f"Log(precio)")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str] = None):
    """
    Generates and plots a correlation heatmap for the numeric columns of a DataFrame.

    Args:
        df: The input DataFrame.
        numeric_cols: Optional. A list of numeric column names to include.
                      If None, all numeric columns will be used.
    """
    if numeric_cols:
        df_numeric = df[numeric_cols]
    else:
        df_numeric = df.select_dtypes(include=np.number)

    if df_numeric.empty:
        print("No numeric columns found to plot correlation heatmap.")
        return

    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Variables", fontsize=16)
    plt.show()


def plot_histograms(df: pd.DataFrame, cols: List[str], n_cols: int = 3):
    """
    Generates a grid of histograms for specified columns in a DataFrame.

    Each histogram includes a vertical dashed red line indicating the mean.

    Args:
        df: The input DataFrame.
        cols: A list of column names to plot.
        n_cols: The number of columns for the subplot grid.
    """
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        if col not in df.columns:
            ax.set_title(f'Column "{col}" not found')
            ax.set_visible(False)
            continue

        # Plot histogram using seaborn
        sns.histplot(df[col], kde=True, ax=ax)

        # Calculate and plot the mean line
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

        ax.set_title(f'Distribution of "{col}"')
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_boxplots(df: pd.DataFrame, cols: List[str], n_cols: int = 3):
    """
    Generates a grid of boxplots for specified numeric columns in a DataFrame.

    Args:
        df: The input DataFrame.
        cols: A list of column names to plot.
        n_cols: The number of columns for the subplot grid.
    """
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        if col not in df.columns:
            ax.set_title(f'Column "{col}" not found')
            ax.set_visible(False)
            continue

        # Plot boxplot using seaborn
        sns.boxplot(y=df[col], ax=ax)
        
        ax.set_title(f'Distribution of "{col}"')
        ax.set_ylabel(col)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()
