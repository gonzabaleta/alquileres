import math
import os
from typing import Dict, List, Tuple, Union

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import plot_tree

from src.constants import COLUMN_NAMES_LEGIBLE

sns.set_theme(style="whitegrid")

PLOTS_PATH = "../plots/"


def finalize_plot(filename: str = None):
    # Crear el directorio si no existe
    if filename:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_PATH, filename + ".png"), bbox_inches="tight", dpi=300
        )
    else:
        plt.tight_layout()

    plt.show()


def plot_boolean_impact(
    df: pd.DataFrame,
    bool_cols: List[str],
    target_col: str,
    n_cols: int = 3,
    filename: str = None,
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

    finalize_plot(filename)


def plot_correlation_heatmap(
    df: pd.DataFrame, numeric_cols: List[str] = None, filename: str = None
):
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

    # Mapear nombres de columnas a versiones legibles
    legible_names = [COLUMN_NAMES_LEGIBLE.get(col, col) for col in corr_matrix.columns]
    corr_matrix.columns = legible_names
    corr_matrix.index = legible_names

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    finalize_plot(filename)


def plot_histograms(
    df: pd.DataFrame,
    cols: List[str],
    n_cols: int = 3,
    clip_percentiles: Tuple[float, float] = None,
    filename: str = None,
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
            ax.set_visible(False)
            continue
        data_to_plot = df[col].dropna()
        if clip_percentiles:
            lower_quantile = data_to_plot.quantile(clip_percentiles[0])
            upper_quantile = data_to_plot.quantile(clip_percentiles[1])
            data_to_plot = data_to_plot.clip(lower=lower_quantile, upper=upper_quantile)
        sns.histplot(data_to_plot, kde=True, ax=ax)
        mean_val = data_to_plot.mean()
        median_val = data_to_plot.median()
        ax.axvline(
            mean_val,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Media: {mean_val:.2f}",
        )
        ax.axvline(
            median_val,
            color="g",
            linestyle=":",
            linewidth=2,
            label=f"Mediana: {median_val:.2f}",
        )
        # Usar nombre legible para xlabel si está disponible
        xlabel = COLUMN_NAMES_LEGIBLE.get(col, col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frecuencia")
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    finalize_plot(filename)


def plot_boxplots(
    df: pd.DataFrame, cols: List[str], n_cols: int = 3, filename: str = None
):
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

    finalize_plot(filename)


def plot_bar_charts(
    df: pd.DataFrame,
    cols: List[str],
    n_cols: int = 2,
    top_n: int = 15,
    filename: str = None,
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
            x=data_to_plot.index,
            y=data_to_plot.values,
            ax=ax,
            order=plot_order,
            hue=data_to_plot.index,
            palette="viridis",
            legend=False,
        )
        # Usar nombre legible para xlabel
        xlabel = COLUMN_NAMES_LEGIBLE.get(col, col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Conteo")
        ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    finalize_plot(filename)


def plot_geo_scatterplot(
    df: pd.DataFrame,
    geo_cols: Tuple[str, str],
    color_col: str,
    sample_size: int = None,
    log_scale: bool = True,
    clip_percentiles: Tuple[float, float] = None,
    cmap: str = "viridis",
    add_basemap: bool = False,
    alpha=0.5,
    filename: str = None,
):
    """
    Generates a scatter plot of geographical data, with points colored by another variable.
    """
    lon_col, lat_col = geo_cols

    if not all(c in df.columns for c in [lon_col, lat_col, color_col]):
        print(
            f"Error: Una o más columnas especificadas no se encontraron en el DataFrame."
        )
        return

    df_plot = df.copy()
    color_data = df_plot[color_col].dropna()

    if clip_percentiles:
        lower_quantile = color_data.quantile(clip_percentiles[0])
        upper_quantile = color_data.quantile(clip_percentiles[1])
        color_data = color_data.clip(lower=lower_quantile, upper=upper_quantile)

    # Mapear nombre de columna a versión legible
    legible_color_name = COLUMN_NAMES_LEGIBLE.get(color_col, color_col)

    if log_scale:
        df_plot["color_values"] = np.log1p(color_data)
        cbar_label = f"Log({legible_color_name})"
    else:
        df_plot["color_values"] = color_data
        cbar_label = legible_color_name

    df_plot = df_plot.dropna(subset=["color_values", lon_col, lat_col])

    if sample_size and sample_size < len(df_plot):
        df_plot = df_plot.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 10))
    hexbin = ax.hexbin(
        x=df_plot[lon_col],
        y=df_plot[lat_col],
        C=df_plot["color_values"],
        cmap=cmap,
        gridsize=65,
        mincnt=1,
        alpha=0.68,
    )
    cbar = fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    # Mejorar aspect ratio - no forzar equal que distorsiona
    ax.set_aspect("auto")

    # Grid sutil para mejor orientación
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if add_basemap:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik)

    finalize_plot(filename)


def plot_median_price_impact(
    df: pd.DataFrame, bool_cols: List[str], target_col: str, filename: str = None
):
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
    impact_df = pd.DataFrame.from_dict(impacts, orient="index", columns=["impact_pct"])
    impact_df = impact_df.sort_values(by="impact_pct", ascending=False)

    # Mapear nombres a versiones legibles
    legible_names = [COLUMN_NAMES_LEGIBLE.get(col, col) for col in impact_df.index]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        range(len(impact_df)),
        impact_df["impact_pct"],
        color=sns.color_palette("viridis", len(impact_df)),
        edgecolor="black",
        alpha=0.8,
    )
    plt.xticks(range(len(impact_df)), legible_names, rotation=45, ha="right")
    plt.ylabel("Impacto (%)")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    finalize_plot(filename)


def plot_categorical_impact(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    n_cols: int = 2,
    filename: str = None,
):
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
            ax.set_visible(False)
            continue
        grouped = df.groupby(col)[target_col].median()
        impact_pct = ((grouped - overall_median) / overall_median) * 100
        is_numeric_like = pd.to_numeric(impact_pct.index, errors="coerce").notna().all()
        if is_numeric_like:
            impact_pct = impact_pct.sort_index()
        else:
            impact_pct = impact_pct.sort_values(ascending=False)
        sns.barplot(
            x=impact_pct.index,
            y=impact_pct.values,
            ax=ax,
            hue=impact_pct.index,
            palette="viridis",
            order=impact_pct.index,
            legend=False,
        )
        # Usar nombre legible para xlabel
        xlabel = COLUMN_NAMES_LEGIBLE.get(col, col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Impacto (%)")
        if len(impact_pct.index) > 20:
            ax.set_xticks([])
            ax.set_xlabel(f"{xlabel} ({len(impact_pct.index)} categorías)")
        else:
            ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    finalize_plot(filename)


def plot_feature_impact_ranking(
    df: pd.DataFrame, cat_cols: List[str], target_col: str, filename: str = None
):
    """
    Ranks and plots the overall impact of categorical features on a target variable.
    """
    overall_median = df[target_col].median()
    impact_scores = {}
    for col in cat_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
        grouped = df.groupby(col)[target_col].median()
        impact_pct = ((grouped - overall_median) / overall_median) * 100
        impact_score = impact_pct.std()
        impact_scores[col] = impact_score
    ranked_df = pd.DataFrame.from_dict(
        impact_scores, orient="index", columns=["impact_score"]
    )
    ranked_df = ranked_df.sort_values(by="impact_score", ascending=False)

    # Mapear nombres a versiones legibles
    legible_names = [COLUMN_NAMES_LEGIBLE.get(col, col) for col in ranked_df.index]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        range(len(ranked_df)),
        ranked_df["impact_score"],
        color=sns.color_palette("viridis", len(ranked_df)),
        edgecolor="black",
        alpha=0.8,
    )
    plt.xticks(range(len(ranked_df)), legible_names, rotation=45, ha="right")
    plt.ylabel("Impacto")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    finalize_plot(filename)


def plot_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str] = None,
    bool_cols: List[str] = None,
    cat_cols: List[str] = None,
    filename: str = None,
):
    """
    Calculates and plots the percentage of missing values, coloring bars by column type.
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        print("No se encontraron valores faltantes en el DataFrame.")
        return

    # Colores estilo viridis para diferentes tipos de variables
    color_map = {
        "numeric": "#440154",  # Viridis dark purple
        "boolean": "#31688E",  # Viridis blue
        "categorical": "#35B779",  # Viridis green
        "other": "#FDE725",  # Viridis yellow
    }

    col_to_type = {}
    if numeric_cols:
        for col in numeric_cols:
            col_to_type[col] = "numeric"
    if bool_cols:
        for col in bool_cols:
            col_to_type[col] = "boolean"
    if cat_cols:
        for col in cat_cols:
            col_to_type[col] = "categorical"

    # Mapear nombres de columnas a versiones legibles
    legible_names = [COLUMN_NAMES_LEGIBLE.get(col, col) for col in missing_pct.index]

    palette = {
        col: color_map.get(col_to_type.get(col, "other"), "gray")
        for col in missing_pct.index
    }

    plt.figure(figsize=(15, 8))
    bars = plt.bar(
        range(len(missing_pct)),
        missing_pct.values,
        color=[palette[col] for col in missing_pct.index],
        edgecolor="black",
        alpha=0.8,
    )

    plt.xticks(range(len(missing_pct)), legible_names, rotation=45, ha="right")
    plt.ylabel("Valores Faltantes (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    from matplotlib.patches import Patch

    # Traducir labels
    type_labels = {
        "numeric": "Numérico",
        "boolean": "Booleano",
        "categorical": "Categórico",
        "other": "Otro",
    }

    legend_elements = [
        Patch(
            facecolor=color_map.get(type_name),
            label=type_labels.get(type_name, type_name),
        )
        for type_name in color_map
        if type_name in col_to_type.values()
    ]
    plt.legend(handles=legend_elements, title="Tipo de Variable")

    finalize_plot(filename)

    not_plotted_cols = df.columns[df.isnull().sum() == 0].tolist()
    if not_plotted_cols:
        print(
            f"\nInfo: The following {len(not_plotted_cols)} columns are not shown because they have no missing values:"
        )
        col_str = ", ".join(not_plotted_cols)
        print(col_str)


def plot_missing_data_impact(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: List[str] = None,
    bool_cols: List[str] = None,
    cat_cols: List[str] = None,
    n_cols: int = 3,
    exclude_cols: List[str] = None,
    filename: str = None,
):
    """
    Generates a grid of boxplots to analyze the impact of missing data on the target variable.
    """
    df_plot = df.copy()
    df_plot[target_col] = np.log1p(df_plot[target_col])

    cols_with_missing = df_plot.columns[df_plot.isnull().any()].tolist()

    if exclude_cols:
        cols_with_missing = [
            col for col in cols_with_missing if col not in exclude_cols
        ]

    if not cols_with_missing:
        print("No columns with missing values to plot.")
        return

    n_rows = math.ceil(len(cols_with_missing) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    color_map = {
        "numeric": "#1f77b4",
        "boolean": "#ff7f0e",
        "categorical": "#2ca02c",
        "other": "#d62728",
    }
    col_to_type = {}
    if numeric_cols:
        for col in numeric_cols:
            col_to_type[col] = "numeric"
    if bool_cols:
        for col in bool_cols:
            col_to_type[col] = "boolean"
    if cat_cols:
        for col in cat_cols:
            col_to_type[col] = "categorical"

    for i, col in enumerate(cols_with_missing):
        ax = axes[i]

        is_missing_col = f"{col}_is_missing"
        df_plot[is_missing_col] = (
            df_plot[col].isnull().map({True: "Missing", False: "Present"})
        )

        color = color_map.get(col_to_type.get(col, "other"), "gray")

        palette = {"Missing": color, "Present": sns.set_hls_values(color, l=0.8)}

        sns.boxplot(
            data=df_plot,
            x=is_missing_col,
            y=target_col,
            ax=ax,
            hue=is_missing_col,
            palette=palette,
            order=["Present", "Missing"],
            legend=False,
        )
        ax.set_title(f'Impact of Missing "{col}"')
        ax.set_xlabel(None)
        ax.set_ylabel(f"Log({target_col})")

    for j in range(len(cols_with_missing), len(axes)):
        axes[j].set_visible(False)

    finalize_plot(filename)


def plot_interaction(
    df: pd.DataFrame,
    target_col: str,
    x_cols: Union[str, List[str]],
    facet_col: str,
    filename: str = None,
):
    """
    Visualizes the interaction between one or more variables and a facet variable.
    """
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    df_plot = df.copy()
    df_plot[target_col] = np.log1p(df_plot[target_col])

    for x_col in x_cols:
        if pd.api.types.is_numeric_dtype(df_plot[x_col].dropna()):
            x_order = sorted(df_plot[x_col].dropna().unique())
        else:
            x_order = df_plot.groupby(x_col)[target_col].median().sort_values().index

        g = sns.catplot(
            data=df_plot,
            x=x_col,
            y=target_col,
            col=facet_col,
            kind="box",
            order=x_order,
            col_wrap=4,
            height=5,
            aspect=1.2,
            hue=x_col,
            palette="viridis",
            legend=False,
        )

        g.fig.suptitle(
            f'Interaction between "{x_col}" and "{facet_col}" on Log({target_col})',
            y=1.03,
        )
        g.set_axis_labels(x_col, f"Log({target_col})")
        g.set_titles(f"{facet_col}: {{col_name}}")
        g.fig.tight_layout()

        if filename:
            # Create a unique filename for each plot in the loop
            base, ext = os.path.splitext(filename)
            loop_filename = f"{base}_{x_col}{ext}"
            os.makedirs(PLOTS_PATH, exist_ok=True)
            g.savefig(os.path.join(PLOTS_PATH, loop_filename))

        plt.show()


def plot_numeric_vs_target(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    n_cols: int = 2,
    sample_size: int = 2000,
    filename: str = None,
):
    """
    Generates a grid of scatter plots for numeric features against a target variable.
    """
    df_plot = df.copy()
    df_plot[target_col] = np.log1p(df_plot[target_col])

    if sample_size and sample_size < len(df_plot):
        df_plot = df_plot.sample(n=sample_size, random_state=42)

    n_rows = math.ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        if col not in df.columns:
            ax.set_title(f'Column "{col}" not found')
            ax.set_visible(False)
            continue

        sns.scatterplot(
            data=df_plot, x=col, y=target_col, ax=ax, alpha=0.3, edgecolor=None
        )
        ax.set_title(f'"{col}" vs. Log({target_col})')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    finalize_plot(filename)


def plot_decision_tree(
    tree_model, feature_names, figsize=(20, 10), filename: str = None
):
    plt.figure(figsize=figsize)
    plot_tree(
        tree_model.named_steps["regressor"],
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Reglas del Árbol de Decisión")
    finalize_plot(filename)


def plot_feature_importance(xgb_model, feature_names, filename: str = None):
    xgb_regressor = xgb_model.named_steps["regressor"]

    importances = xgb_regressor.feature_importances_

    feat_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    # Mapear a nombres legibles
    feat_imp_df["Feature"] = feat_imp_df["Feature"].map(COLUMN_NAMES_LEGIBLE)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feat_imp_df.head(20),
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.xlabel("Importancia (Gain)")
    plt.ylabel("Feature")
    finalize_plot(filename)


def plot_model_comparison(models_predictions: Dict[str, List], filename: str = None):
    """
    Plotea métricas comparativas (RMSE, MAE, R2, MAPE) para múltiples modelos.
    Maneja conjuntos de prueba distintos para RAW vs PROCESSED.

    Args:
        models_predictions (dict):
            {
                'model':List[str],
                'y_pred': List[np.array],
                'y_pred_processed': List[np.array] (opcional),
                'y_test': np.array (para raw),
                'y_test_processed': np.array (para processed/opcional)
            }
    """

    metrics_data = []
    has_processed = "y_pred_processed" in models_predictions
    y_test_raw = models_predictions["y_test"]
    y_test_proc = models_predictions.get("y_test_processed")

    # Iterar sobre cada modelo
    for i, name in enumerate(models_predictions["model"]):
        # 1. Versión Raw (usa y_test)
        y_pred_raw = models_predictions["y_pred"][i]
        rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
        mae = mean_absolute_error(y_test_raw, y_pred_raw)
        r2 = r2_score(y_test_raw, y_pred_raw)
        mape = mean_absolute_percentage_error(y_test_raw, y_pred_raw)

        metrics_data.append(
            {
                "Model": name,
                "Variant": "Raw",
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "MAPE": mape,
            }
        )

        # 2. Versión Procesada (si existe, usa y_test_processed)
        if has_processed and y_test_proc is not None:
            y_pred_proc = models_predictions["y_pred_processed"][i]
            rmse_p = np.sqrt(mean_squared_error(y_test_proc, y_pred_proc))
            mae_p = mean_absolute_error(y_test_proc, y_pred_proc)
            r2_p = r2_score(y_test_proc, y_pred_proc)
            mape_p = mean_absolute_percentage_error(y_test_proc, y_pred_proc)

            metrics_data.append(
                {
                    "Model": name,
                    "Variant": "Processed",
                    "RMSE": rmse_p,
                    "MAE": mae_p,
                    "R2": r2_p,
                    "MAPE": mape_p,
                }
            )

    df_metrics = pd.DataFrame(metrics_data)

    # Baseline (Dummy Raw - Primer modelo)
    dummy_baseline = df_metrics[
        (df_metrics["Model"] == models_predictions["model"][0])
        & (df_metrics["Variant"] == "Raw")
    ].iloc[0]

    # Configurar plot (Solo RMSE y MAE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    metrics_to_plot = [
        ("RMSE", "RMSE", "%.0f"),
        ("MAE", "MAE", "%.0f"),
    ]

    hue_col = "Variant" if has_processed else "Model"
    palette = "Set2" if has_processed else "viridis"

    for i, (metric, title, fmt) in enumerate(metrics_to_plot):
        ax = axes[i]

        sns.barplot(
            data=df_metrics,
            x="Model",
            y=metric,
            hue=hue_col,
            palette=palette,
            ax=ax,
            edgecolor="black",
        )

        # Línea de Baseline
        baseline_val = dummy_baseline[metric]
        ax.axhline(
            baseline_val, color="red", linestyle="--", linewidth=1.5, label="Baseline"
        )

        # Formato de eje Y
        ax.ticklabel_format(style="plain", axis="y")

        # Etiquetas de valores sobre las barras
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=9)

        if i == 0:
            ax.legend(loc="upper right")
        else:
            if ax.get_legend():
                ax.legend_.remove()

    finalize_plot(filename)

    return df_metrics


def plot_cv_results(
    results_df: pd.DataFrame,
    baseline_name: str = "Baseline",
    vertical_layout: bool = True,
    filename: str = None,
):
    """
    Genera dos gráficos de barras verticales para visualizar los resultados de CV.
    Si hay métricas de train, muestra barras agrupadas (train vs test).

    Args:
        results_df (pd.DataFrame): DataFrame con los resultados.
        baseline_name (str): Nombre de la configuración a usar como línea de referencia.
        vertical_layout (bool): Si True, apila gráficos verticalmente (2 filas, 1 col).
                                Si False, coloca gráficos lado a lado (1 fila, 2 cols).
    """
    # Detectar si hay métricas de train
    has_train_metrics = "train_mae_mean" in results_df.columns

    if vertical_layout:
        fig, axes = plt.subplots(2, 1, figsize=(14, 18))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    metrics = [("rmse", "Avg RMSE"), ("mae", "Avg MAE")]

    # Crear una paleta fija mapeando cada configuración a un color específico
    # Esto asegura que "Config A" tenga el mismo color en ambos gráficos sin importar el orden
    unique_configs = results_df.index.unique()
    colors = sns.color_palette("viridis", len(unique_configs))
    palette_dict = dict(zip(unique_configs, colors))

    for i, (metric_key, title) in enumerate(metrics):
        ax = axes[i]
        # Ordenar data por la métrica actual (Ascendente: menor es mejor)
        data_sorted = results_df.sort_values(by=f"{metric_key}_mean")

        x_pos = np.arange(len(data_sorted))

        if has_train_metrics:
            # Barras agrupadas: Train vs Test
            bar_width = 0.35

            # Barras de TRAIN
            bars_train = ax.bar(
                x_pos - bar_width / 2,
                data_sorted[f"train_{metric_key}_mean"],
                bar_width,
                label="Train",
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                linewidth=1,
            )

            # Barras de TEST
            bars_test = ax.bar(
                x_pos + bar_width / 2,
                data_sorted[f"{metric_key}_mean"],
                bar_width,
                label="Test (CV)",
                alpha=0.8,
                color="coral",
                edgecolor="black",
                linewidth=1,
            )

            # Error bars solo para test
            ax.errorbar(
                x=x_pos + bar_width / 2,
                y=data_sorted[f"{metric_key}_mean"],
                yerr=data_sorted[f"{metric_key}_std"],
                fmt="none",
                c="black",
                capsize=5,
                linewidth=1.5,
            )

            ax.legend()
        else:
            # Barras simples (solo test)
            colors = sns.color_palette("viridis", len(data_sorted))

            bars = ax.bar(
                x_pos,
                data_sorted[f"{metric_key}_mean"],
                color=colors,
                edgecolor="black",
                linewidth=1,
            )

            # Barras de error
            ax.errorbar(
                x=x_pos,
                y=data_sorted[f"{metric_key}_mean"],
                yerr=data_sorted[f"{metric_key}_std"],
                fmt="none",
                c="black",
                capsize=5,
                linewidth=1.5,
            )

        # Baseline Line (Referencia)
        if baseline_name in results_df.index:
            baseline_val = results_df.loc[baseline_name, f"{metric_key}_mean"]
            ax.axhline(
                baseline_val,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Baseline",
            )
            ax.legend()

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(title)
        ax.set_xlabel("Configuración")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data_sorted.index, rotation=45, ha="right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    finalize_plot(filename)


def plot_grid_search_results(
    results_df: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple = (16, 7),
    filename: str = None,
):
    """
    Visualiza las mejores N configuraciones de un grid search.

    MAE plot muestra top N por MAE, RMSE plot muestra top N por RMSE.

    Args:
        results_df: DataFrame retornado por grid_search_cv()
        top_n: Número de mejores configuraciones a mostrar (default: 10)
        figsize: Tamaño de la figura (default: (16, 7))
    """
    # Top N por MAE y por RMSE (pueden ser diferentes)
    top_by_mae = results_df.sort_values("mae_mean").head(top_n).copy()
    top_by_rmse = results_df.sort_values("rmse_mean").head(top_n).copy()

    # Identificar columnas de parámetros
    metric_cols = [
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "train_mae_mean",
        "train_mae_std",
        "train_rmse_mean",
        "train_rmse_std",
        "overfit_gap_%",
    ]
    param_cols = [col for col in results_df.columns if col not in metric_cols]

    # Detectar columna de ID
    id_col = None
    for potential_id in ["config_id", "id"]:
        if potential_id in results_df.columns:
            id_col = potential_id
            break

    # Crear etiquetas para cada subplot
    def get_labels(df):
        if id_col:
            return [f"ID {int(row[id_col])}" for _, row in df.iterrows()]
        else:
            return [f"Config {i+1}" for i in range(len(df))]

    labels_mae = get_labels(top_by_mae)
    labels_rmse = get_labels(top_by_rmse)

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Ancho de barras
    bar_width = 0.35

    # ============================================================
    # SUBPLOT 1: MAE (Top N por MAE)
    # ============================================================
    ax_mae = axes[0]
    x_pos_mae = np.arange(len(top_by_mae))

    bars_train = ax_mae.bar(
        x_pos_mae - bar_width / 2,
        top_by_mae["train_mae_mean"],
        bar_width,
        label="Train",
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        linewidth=1,
    )

    bars_test = ax_mae.bar(
        x_pos_mae + bar_width / 2,
        top_by_mae["mae_mean"],
        bar_width,
        label="Test (CV)",
        alpha=0.8,
        color="coral",
        edgecolor="black",
        linewidth=1,
    )

    # Destacar mejor
    bars_train[0].set_edgecolor("darkgoldenrod")
    bars_train[0].set_linewidth(2)
    bars_test[0].set_edgecolor("darkgoldenrod")
    bars_test[0].set_linewidth(2)

    ax_mae.set_xlabel("Configuración", fontsize=12, fontweight="bold")
    ax_mae.set_ylabel("MAE", fontsize=12, fontweight="bold")
    ax_mae.set_title(
        f"Top {top_n} por MAE - Train vs Test", fontsize=14, fontweight="bold"
    )
    ax_mae.set_xticks(x_pos_mae)
    ax_mae.set_xticklabels(labels_mae, rotation=45, ha="right", fontsize=10)
    ax_mae.legend()
    ax_mae.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax_mae.set_axisbelow(True)

    # ============================================================
    # SUBPLOT 2: RMSE (Top N por RMSE)
    # ============================================================
    ax_rmse = axes[1]
    x_pos_rmse = np.arange(len(top_by_rmse))

    bars_train_rmse = ax_rmse.bar(
        x_pos_rmse - bar_width / 2,
        top_by_rmse["train_rmse_mean"],
        bar_width,
        label="Train",
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        linewidth=1,
    )

    bars_test_rmse = ax_rmse.bar(
        x_pos_rmse + bar_width / 2,
        top_by_rmse["rmse_mean"],
        bar_width,
        label="Test (CV)",
        alpha=0.8,
        color="coral",
        edgecolor="black",
        linewidth=1,
    )

    # Destacar mejor
    bars_train_rmse[0].set_edgecolor("darkgoldenrod")
    bars_train_rmse[0].set_linewidth(2)
    bars_test_rmse[0].set_edgecolor("darkgoldenrod")
    bars_test_rmse[0].set_linewidth(2)

    ax_rmse.set_xlabel("Configuración", fontsize=12, fontweight="bold")
    ax_rmse.set_ylabel("RMSE", fontsize=12, fontweight="bold")
    ax_rmse.set_title(
        f"Top {top_n} por RMSE - Train vs Test", fontsize=14, fontweight="bold"
    )
    ax_rmse.set_xticks(x_pos_rmse)
    ax_rmse.set_xticklabels(labels_rmse, rotation=45, ha="right", fontsize=10)
    ax_rmse.legend()
    ax_rmse.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax_rmse.set_axisbelow(True)

    finalize_plot(filename)


def plot_learning_curve(
    evals_result: dict,
    metric: str = "rmse",
    figsize: tuple = (14, 5),
    filename: str = None,
):
    """
    Grafica la learning curve de un modelo XGBoost.

    Args:
        evals_result: Diccionario retornado por xgb_model.evals_result()
        metric: Métrica a graficar (default: 'rmse')
        figsize: Tamaño de la figura (default: (14, 5))
    """
    # Extraer métricas
    train_metric = evals_result["validation_0"][metric]
    val_metric = evals_result["validation_1"][metric]

    epochs = range(1, len(train_metric) + 1)

    # Crear figura con 2 subplots
    fig = plt.figure(figsize=figsize)

    plt.plot(epochs, train_metric, label="Train", linewidth=2, color="skyblue")
    plt.plot(epochs, val_metric, label="Validation", linewidth=2, color="coral")
    plt.xlabel("Number of Trees (Iterations)", fontweight="bold")
    plt.ylabel(metric.upper(), fontweight="bold")
    plt.title(f"Learning Curve - {metric.upper()}", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)

    finalize_plot(filename)


def plot_classifier_results(
    results_df: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple = (16, 18),
    filename: str = None,
):
    """
    Visualiza resultados de classifier_search_cv con 6 subplots (3x2).

    Args:
        results_df: DataFrame retornado por classifier_search_cv
        top_n: Número de mejores configuraciones a mostrar en bar charts
        figsize: Tamaño de la figura
    """
    # Detectar columna de ID
    id_col = None
    for potential_id in ["config_id", "id"]:
        if potential_id in results_df.columns:
            id_col = potential_id
            break

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Métricas para bar charts (cada una ordenada por sí misma)
    # Usar nombres con prefijo test_
    metrics_config = [
        ("test_recall", "Top N por Recall", "coral"),
        ("test_f1", "Top N por F1", "steelblue"),
        ("test_avg_precision", "Top N por Avg Precision", "mediumseagreen"),
        ("test_precision", "Top N por Precision", "goldenrod"),
    ]

    # ============================================================
    # Bar charts (primeras 4 posiciones)
    # ============================================================
    bar_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (metric, title, color) in enumerate(metrics_config):
        row, col = bar_positions[idx]
        ax = axes[row, col]

        # Nombre de columna test
        metric_name = metric.replace("test_", "")

        # Ordenar por esta métrica y tomar top N
        top_data = results_df.sort_values(metric, ascending=False).head(top_n)

        # Labels
        if id_col:
            labels = [f"ID {int(r[id_col])}" for _, r in top_data.iterrows()]
        else:
            labels = [f"#{i+1}" for i in range(len(top_data))]

        x_pos = np.arange(len(top_data))

        bars = ax.bar(
            x_pos,
            top_data[metric],
            color=color,
            edgecolor="black",
            linewidth=1,
            alpha=0.8,
        )

        # Destacar el mejor
        bars[0].set_edgecolor("darkred")
        bars[0].set_linewidth(2.5)

        ax.set_xlabel("Configuración", fontweight="bold")
        ax.set_ylabel(metric_name.upper(), fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        # Agregar valores sobre las barras
        for bar, val in zip(bars, top_data[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ============================================================
    # Scatter: Precision vs Recall (color = F1)
    # ============================================================
    ax = axes[2, 0]
    scatter = ax.scatter(
        results_df["test_precision"],
        results_df["test_recall"],
        c=results_df["test_f1"],
        cmap="RdYlGn",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="F1 Score")

    # Marcar el mejor F1
    best_f1_idx = results_df["test_f1"].idxmax()
    best = results_df.loc[best_f1_idx]
    ax.scatter(
        best["test_precision"],
        best["test_recall"],
        s=200,
        facecolors="none",
        edgecolors="red",
        linewidth=2.5,
        label="Best F1",
    )

    ax.set_xlabel("Precision", fontweight="bold")
    ax.set_ylabel("Recall", fontweight="bold")
    ax.set_title("Precision vs Recall (color=F1)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # ============================================================
    # Scatter: ROC-AUC vs F1
    # ============================================================
    ax = axes[2, 1]
    scatter = ax.scatter(
        results_df["test_roc_auc"],
        results_df["test_f1"],
        c=results_df["test_recall"],
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Recall")

    # Marcar el mejor F1
    ax.scatter(
        best["test_roc_auc"],
        best["test_f1"],
        s=200,
        facecolors="none",
        edgecolors="red",
        linewidth=2.5,
        label="Best F1",
    )

    ax.set_xlabel("ROC-AUC", fontweight="bold")
    ax.set_ylabel("F1", fontweight="bold")
    ax.set_title("ROC-AUC vs F1 (color=Recall)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    finalize_plot(filename)

    # Imprimir resumen del mejor modelo
    print("\n" + "=" * 60)
    print("🏆 MEJOR MODELO (por F1):")
    print("=" * 60)
    print(f"Config ID: {int(best[id_col]) if id_col else 'N/A'}")
    print(f"F1:        {best['test_f1']:.4f}")
    print(f"Precision: {best['test_precision']:.4f}")
    print(f"Recall:    {best['test_recall']:.4f}")
    print(f"ROC-AUC:   {best['test_roc_auc']:.4f}")
    print(f"Avg Prec:  {best['test_avg_precision']:.4f}")


def plot_classifier_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ["f1", "recall", "precision"],
    figsize: Tuple[int, int] = (18, 6),
    title: str = "Comparación de Modelos de Clasificación",
    filename: str = None,
):
    """
    Genera gráficos de barras para comparar clasificadores en múltiples métricas.
    Si hay métricas de train, muestra barras agrupadas (Train vs Test).

    Args:
        results_df: DataFrame retornado por evaluate_classifiers_cv
        metrics: Lista de métricas a plotear (ej: ['f1', 'recall'])
        figsize: Tamaño de la figura
        title: Título general
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    # Colores
    test_color = "coral"

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Ordenar (Mayor es mejor)
        data = results_df.sort_values(metric, ascending=False)

        x_pos = np.arange(len(data))
        model_names = data.index

        # Solo Test
        bars_test = ax.bar(
            x_pos,
            data[metric],
            width=0.6,
            yerr=data.get(f"{metric}_std"),
            color=test_color,
            edgecolor="black",
            alpha=0.9,
        )

        # Estética
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontweight="bold")
        ax.set_ylabel(metric.upper(), fontweight="bold")
        ax.set_title(f"Ranking por {metric.upper()}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        # Agregar valores
        for bar, val in zip(bars_test, data[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="darkred",
            )

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.05)
    finalize_plot(filename)


def plot_pca_scatter(
    X_pca_2d, y_target, sample_size=5000, figsize=(12, 8), filename=None
):
    """
    Crea un scatter plot de los dos primeros componentes principales coloreado por el target.

    Args:
        X_pca_2d: Array de componentes principales (n_samples, 2)
        y_target: Series con los valores del target (precio)
        sample_size: Número de puntos a mostrar (para evitar overplotting)
        figsize: Tamaño de la figura
        filename: Nombre del archivo para guardar
    """
    # Crear DataFrame temporal para facilitar sampling
    df_temp = pd.DataFrame(
        {"PC1": X_pca_2d[:, 0], "PC2": X_pca_2d[:, 1], "target": y_target.values}
    )

    # Sampling si hay muchos datos
    if sample_size and len(df_temp) > sample_size:
        df_temp = df_temp.sample(n=sample_size, random_state=42)

    # Crear figura
    plt.figure(figsize=figsize)

    # Detectar si es variable categórica o numérica
    if y_target.dtype == "object" or len(y_target.unique()) < 20:
        # Variable categórica
        unique_values = df_temp["target"].unique()
        colors = sns.color_palette("Set1", n_colors=len(unique_values))

        for i, value in enumerate(unique_values):
            mask = df_temp["target"] == value
            plt.scatter(
                df_temp.loc[mask, "PC1"],
                df_temp.loc[mask, "PC2"],
                label=value,
                alpha=0.6,
                s=30,
                edgecolors="none",
                color=colors[i],
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        # Variable numérica
        scatter = plt.scatter(
            df_temp["PC1"],
            df_temp["PC2"],
            c=df_temp["target"],
            cmap="viridis",
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

        # Colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Log(Precio)")

    # Configurar plot
    plt.xlabel("Primera Componente Principal (PC1)")
    plt.ylabel("Segunda Componente Principal (PC2)")
    plt.grid(True, alpha=0.3)

    finalize_plot(filename)


def plot_cluster_analysis(df_pca, cluster_analysis, amenities_analysis, filename=None):
    """
    Crea un análisis visual completo de los clusters.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Distribución de clusters (tamaño)
    cluster_counts = df_pca["cluster"].value_counts().sort_index()
    colors = sns.color_palette("Set2", len(cluster_counts))

    axes[0, 0].pie(
        cluster_counts.values,
        labels=[f"Cluster {i}" for i in cluster_counts.index],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    axes[0, 0].set_title("Distribución de Clusters")

    # 2. Precio promedio por cluster
    price_means = cluster_analysis["precio_pesos_constantes_mean"]
    axes[0, 1].bar(price_means.index, np.exp(price_means.values), color=colors)
    axes[0, 1].set_xlabel("Cluster")
    axes[0, 1].set_ylabel("Precio Promedio ($)")
    axes[0, 1].set_title("Precio Promedio por Cluster")
    axes[0, 1].tick_params(axis="y", rotation=45)

    # 3. Tamaño promedio por cluster
    size_means = cluster_analysis["STotalM2_mean"]
    axes[0, 2].bar(size_means.index, size_means.values, color=colors)
    axes[0, 2].set_xlabel("Cluster")
    axes[0, 2].set_ylabel("Superficie (m²) - Estandarizada")
    axes[0, 2].set_title("Tamaño Promedio por Cluster")

    # 4. Boxplot de precios por cluster
    df_pca.boxplot(column="precio_pesos_constantes", by="cluster", ax=axes[1, 0])
    axes[1, 0].set_xlabel("Cluster")
    axes[1, 0].set_ylabel("Log(Precio)")
    axes[1, 0].set_title("Distribución de Precios por Cluster")
    plt.suptitle("")  # Remove automatic suptitle

    # 5. Heatmap de amenities
    amenities_subset = amenities_analysis[
        ["Pileta", "Gimnasio", "Seguridad", "AireAC", "Laundry"]
    ]
    # Asegurar que los datos sean numéricos
    amenities_subset = amenities_subset.astype(float)
    im = axes[1, 1].imshow(amenities_subset.values, cmap="YlOrRd", aspect="auto")
    axes[1, 1].set_xticks(range(len(amenities_subset.columns)))
    axes[1, 1].set_xticklabels(amenities_subset.columns, rotation=45)
    axes[1, 1].set_yticks(range(len(amenities_subset.index)))
    axes[1, 1].set_yticklabels([f"Cluster {i}" for i in amenities_subset.index])
    axes[1, 1].set_title("Amenities por Cluster")

    # Añadir valores en el heatmap
    for i in range(len(amenities_subset.index)):
        for j in range(len(amenities_subset.columns)):
            axes[1, 1].text(
                j,
                i,
                f"{amenities_subset.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    # 6. Luxury properties por cluster
    luxury_pct = cluster_analysis["outlier_mean"] * 100
    axes[1, 2].bar(luxury_pct.index, luxury_pct.values, color=colors)
    axes[1, 2].set_xlabel("Cluster")
    axes[1, 2].set_ylabel("% Propiedades Luxury")
    axes[1, 2].set_title("Propiedades Luxury por Cluster")

    plt.tight_layout()
    finalize_plot(filename)


def plot_cluster_heatmaps(df_pca, numerical_cols, categorical_cols, filename=None):
    """
    Crea heatmaps de análisis de clusters con variables personalizables.

    Args:
        df_pca: DataFrame con columna 'cluster'
        numerical_cols: Lista de columnas numéricas para el primer heatmap
        categorical_cols: Lista de columnas categóricas/booleanas para el segundo heatmap
        filename: Nombre del archivo para guardar
    """
    # Análisis interno
    numerical_analysis = df_pca.groupby('cluster')[numerical_cols].mean()
    categorical_analysis = df_pca.groupby('cluster')[categorical_cols].mean()

    # Layout vertical con heatmaps cuadrados
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))

    # 1. Heatmap de variables numéricas
    im1 = axes[0].imshow(numerical_analysis.values.astype(float), cmap='RdYlBu_r', aspect='equal')
    axes[0].set_xticks(range(len(numerical_analysis.columns)))
    axes[0].set_xticklabels([COLUMN_NAMES_LEGIBLE.get(col, col) for col in numerical_analysis.columns],
                           rotation=45, ha='right')
    axes[0].set_yticks(range(len(numerical_analysis.index)))
    axes[0].set_yticklabels([f'Cluster {i}' for i in numerical_analysis.index])
    axes[0].set_title('Variables Numéricas por Cluster', fontsize=14, pad=20)

    # Añadir valores en el heatmap
    for i in range(len(numerical_analysis.index)):
        for j in range(len(numerical_analysis.columns)):
            axes[0].text(j, i, f'{numerical_analysis.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if abs(numerical_analysis.iloc[i, j]) > 1 else 'black')

    # 2. Heatmap de variables categóricas/amenities
    im2 = axes[1].imshow(categorical_analysis.values.astype(float), cmap='YlOrRd', aspect='equal')
    axes[1].set_xticks(range(len(categorical_analysis.columns)))
    axes[1].set_xticklabels([COLUMN_NAMES_LEGIBLE.get(col, col) for col in categorical_analysis.columns],
                           rotation=45, ha='right')
    axes[1].set_yticks(range(len(categorical_analysis.index)))
    axes[1].set_yticklabels([f'Cluster {i}' for i in categorical_analysis.index])
    axes[1].set_title('Amenities por Cluster', fontsize=14, pad=20)

    # Añadir valores en el heatmap
    for i in range(len(categorical_analysis.index)):
        for j in range(len(categorical_analysis.columns)):
            axes[1].text(j, i, f'{categorical_analysis.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if categorical_analysis.iloc[i, j] > 0.3 else 'black')

    # Colorbars
    plt.colorbar(im1, ax=axes[0], shrink=0.6)
    plt.colorbar(im2, ax=axes[1], shrink=0.6)

    plt.tight_layout()
    finalize_plot(filename)


def plot_cluster_geographic(df_pca, filename=None):
    """
    Análisis geográfico de clusters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Scatter geográfico coloreado por cluster
    colors = sns.color_palette("Set2", df_pca["cluster"].nunique())
    for i, cluster in enumerate(sorted(df_pca["cluster"].unique())):
        cluster_data = df_pca[df_pca["cluster"] == cluster].sample(
            min(1000, len(df_pca[df_pca["cluster"] == cluster]))
        )
        axes[0].scatter(
            cluster_data["LONGITUDE"],
            cluster_data["LATITUDE"],
            c=[colors[i]],
            label=f"Cluster {cluster}",
            alpha=0.6,
            s=20,
        )

    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")
    axes[0].set_title("Distribución Geográfica de Clusters")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Ciudades más frecuentes por cluster
    cluster_cities = []
    city_counts = []

    for cluster in sorted(df_pca["cluster"].unique()):
        cluster_data = df_pca[df_pca["cluster"] == cluster]
        top_city = cluster_data["ITE_ADD_CITY_NAME"].mode().iloc[0]
        city_count = (cluster_data["ITE_ADD_CITY_NAME"] == top_city).sum()
        cluster_cities.append(f"Cluster {cluster}\n({top_city})")
        city_counts.append(city_count)

    axes[1].bar(range(len(cluster_cities)), city_counts, color=colors)
    axes[1].set_xticks(range(len(cluster_cities)))
    axes[1].set_xticklabels(cluster_cities, rotation=45, ha="right")
    axes[1].set_ylabel("Cantidad de Propiedades")
    axes[1].set_title("Ciudad Principal por Cluster")

    plt.tight_layout()
    finalize_plot(filename)


def plot_boolean_percentage(
    df: pd.DataFrame,
    bool_cols: List[str],
    figsize: Tuple[int, int] = (12, 8),
    filename: str = None,
):
    """
    Genera un gráfico de barras mostrando el porcentaje de valores True para columnas booleanas.

    Args:
        df: DataFrame con los datos
        bool_cols: Lista de nombres de columnas booleanas
        figsize: Tamaño de la figura
        filename: Nombre del archivo para guardar (opcional)
    """
    # Calcular porcentajes de True para cada columna
    percentages = {}
    for col in bool_cols:
        if col in df.columns:
            # Calcular porcentaje de True (excluyendo valores nulos)
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                pct_true = (valid_data == True).mean() * 100
                percentages[col] = pct_true

    if not percentages:
        print("No se encontraron columnas booleanas válidas.")
        return

    # Crear DataFrame para el plot
    pct_df = pd.DataFrame.from_dict(percentages, orient="index", columns=["percentage"])
    pct_df = pct_df.sort_values("percentage", ascending=False)

    # Mapear nombres a versiones legibles
    legible_names = [COLUMN_NAMES_LEGIBLE.get(col, col) for col in pct_df.index]

    # Crear el plot
    plt.figure(figsize=figsize)
    bars = plt.bar(
        range(len(pct_df)),
        pct_df["percentage"],
        color=sns.color_palette("viridis", len(pct_df)),
        edgecolor="black",
        alpha=0.8,
    )

    # Configurar ejes y etiquetas
    plt.xticks(range(len(pct_df)), legible_names, rotation=45, ha="right")
    plt.ylabel("Porcentaje (%)", fontweight="bold")
    plt.ylim(0, 100)

    # Agregar valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars, pct_df["percentage"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Grid para mejor legibilidad
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    finalize_plot(filename)
