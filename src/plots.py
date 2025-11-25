import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from typing import List, Tuple
import contextily as cx


def plot_boolean_impact(
    df: pd.DataFrame, bool_cols: List[str], target_col: str, n_cols: int = 3
):
    """
    Generates a grid of boxplots to analyze the impact of boolean columns on a target variable.
    """
    n_rows = math.ceil(len(bool_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()
    log_target = np.log1p(df[target_col])

    for i, col in enumerate(bool_cols):
        ax = axes[i]
        plot_series = df[col].astype("object").fillna("Unknown")
        plot_order = [True, False, "Unknown"]
        sns.boxplot(x=plot_series, y=log_target, ax=ax, order=plot_order)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel(f"Log(precio)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str] = None):
    """
    Generates and plots a correlation heatmap for the numeric columns of a DataFrame.
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


def plot_histograms(
    df: pd.DataFrame,
    cols: List[str],
    n_cols: int = 3,
    clip_percentiles: Tuple[float, float] = None,
):
    """
    Generates a grid of histograms for specified columns in a DataFrame.
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
        data_to_plot = df[col].dropna()
        title = f'Distribution of "{col}"'
        if clip_percentiles:
            lower_quantile = data_to_plot.quantile(clip_percentiles[0])
            upper_quantile = data_to_plot.quantile(clip_percentiles[1])
            data_to_plot = data_to_plot.clip(lower=lower_quantile, upper=upper_quantile)
            title += f"\n(Clipped at {clip_percentiles[0]*100:.0f}-{clip_percentiles[1]*100:.0f}th percentiles)"
        sns.histplot(data_to_plot, kde=True, ax=ax)
        mean_val = data_to_plot.mean()
        ax.axvline(
            mean_val,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.2f}",
        )
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, cols: List[str], n_cols: int = 3):
    """
    Generates a grid of boxplots for specified numeric columns in a DataFrame.
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
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f'Distribution of "{col}"')
        ax.set_ylabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_bar_charts(
    df: pd.DataFrame, cols: List[str], n_cols: int = 2, top_n: int = 15
):
    """
    Generates a grid of bar charts for specified categorical columns.
    """
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        if col not in df.columns:
            ax.set_title(f'Column "{col}" not found')
            ax.set_visible(False)
            continue
        counts = df[col].value_counts()
        is_numeric_like = pd.to_numeric(counts.index, errors="coerce").notna().all()
        if is_numeric_like:
            counts = counts.sort_index()
        if len(counts) > top_n:
            top_counts = counts.nlargest(top_n)
            other_count = counts.iloc[top_n:].sum()
            top_counts["Other"] = other_count
            data_to_plot = top_counts
        else:
            data_to_plot = counts
        plot_order = data_to_plot.index
        sns.barplot(
            x=data_to_plot.index, y=data_to_plot.values, ax=ax, order=plot_order
        )
        ax.set_title(f'Frequency of Categories in "{col}"')
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_geo_scatterplot(
    df: pd.DataFrame,
    geo_cols: Tuple[str, str],
    color_col: str,
    sample_size: int = None,
    log_scale: bool = True,
    clip_percentiles: Tuple[float, float] = None,
    cmap: str = "viridis",
    add_basemap: bool = False,
    alpha=0.3,
):
    """
    Generates a scatter plot of geographical data, with points colored by another variable.
    """
    lon_col, lat_col = geo_cols

    if not all(c in df.columns for c in [lon_col, lat_col, color_col]):
        print(f"Error: One or more specified columns not found in the DataFrame.")
        return

    df_plot = df.copy()

    color_data = df_plot[color_col].dropna()

    if clip_percentiles:
        lower_quantile = color_data.quantile(clip_percentiles[0])
        upper_quantile = color_data.quantile(clip_percentiles[1])
        color_data = color_data.clip(lower=lower_quantile, upper=upper_quantile)

    if log_scale:
        df_plot["color_values"] = np.log1p(color_data)
        cbar_label = f"Log({color_col})"
    else:
        df_plot["color_values"] = color_data
        cbar_label = color_col

    df_plot = df_plot.dropna(subset=["color_values", lon_col, lat_col])

    if sample_size and sample_size < len(df_plot):
        df_plot = df_plot.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 12))

    scatter = ax.scatter(
        x=df_plot[lon_col],
        y=df_plot[lat_col],
        c=df_plot["color_values"],
        cmap=cmap,
        s=5,
        edgecolors=None,
        alpha=alpha,
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    title = f"Geographical Distribution by {cbar_label}"
    if clip_percentiles:
        title += f"\n(Color scale clipped at {clip_percentiles[0]*100:.0f}-{clip_percentiles[1]*100:.0f}th percentiles)"
    ax.set_title(title, fontsize=16)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis("equal")

    if add_basemap:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik)

    plt.show()

def plot_median_price_impact(df: pd.DataFrame, bool_cols: List[str], target_col: str):
    """
    Calculates and plots the percentage impact on median price for boolean features.
    """
    impacts = {}
    overall_median = df[target_col].median()

    for col in bool_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
        median_true = df[df[col] == True][target_col].median()
        if pd.isna(median_true):
            impact = 0
        else:
            impact = ((median_true - overall_median) / overall_median) * 100
        impacts[col] = impact

    impact_df = pd.DataFrame.from_dict(impacts, orient='index', columns=['impact_pct'])
    impact_df = impact_df.sort_values(by='impact_pct', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=impact_df.index, y=impact_df['impact_pct'], palette='viridis')
    plt.title('Percentage Impact on Median Price by Amenity', fontsize=16)
    plt.xlabel('Amenity')
    plt.ylabel('Median Price Impact (%) vs. Overall Median')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_categorical_impact(df: pd.DataFrame, cat_cols: List[str], target_col: str, n_cols: int = 2):
    """
    Generates a grid of bar charts showing the impact of categorical features on the median price.
    """
    overall_median = df[target_col].median()
    n_rows = math.ceil(len(cat_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        if col not in df.columns:
            ax.set_title(f'Column "{col}" not found')
            ax.set_visible(False)
            continue

        grouped = df.groupby(col)[target_col].median()
        impact_pct = ((grouped - overall_median) / overall_median) * 100
        
        # --- Smart Sorting Logic ---
        is_numeric_like = pd.to_numeric(impact_pct.index, errors='coerce').notna().all()
        if is_numeric_like:
            impact_pct = impact_pct.sort_index()
        else:
            impact_pct = impact_pct.sort_values(ascending=False)
        
        sns.barplot(x=impact_pct.index, y=impact_pct.values, ax=ax, palette="viridis", order=impact_pct.index)
        
        ax.set_title(f'Median Price Impact of "{col}"')
        ax.set_xlabel(col)
        ax.set_ylabel('Median Price Impact (%) vs. Overall')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
