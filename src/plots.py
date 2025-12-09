import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import math
from typing import List, Tuple, Dict, Union
import contextily as cx
from sklearn.tree import plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


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
    impact_df = pd.DataFrame.from_dict(impacts, orient="index", columns=["impact_pct"])
    impact_df = impact_df.sort_values(by="impact_pct", ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=impact_df.index,
        y=impact_df["impact_pct"],
        hue=impact_df.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Percentage Impact on Median Price by Amenity", fontsize=16)
    plt.xlabel("Amenity")
    plt.ylabel("Median Price Impact (%) vs. Overall Median")
    plt.xticks(rotation=45)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_categorical_impact(
    df: pd.DataFrame, cat_cols: List[str], target_col: str, n_cols: int = 2
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
            ax.set_title(f'Column "{col}" not found')
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
        ax.set_title(f'Median Price Impact of "{col}"')
        ax.set_xlabel(col)
        ax.set_ylabel("Median Price Impact (%) vs. Overall")
        if len(impact_pct.index) > 20:
            ax.set_xticks([])
            ax.set_xlabel(f"{col} ({len(impact_pct.index)} categories)")
        else:
            ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_feature_impact_ranking(df: pd.DataFrame, cat_cols: List[str], target_col: str):
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
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=ranked_df.index,
        y=ranked_df["impact_score"],
        hue=ranked_df.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Ranking of Categorical Feature Impact", fontsize=16)
    plt.xlabel("Feature")
    plt.ylabel("Impact Score (Std. Dev. of Median Price Impact %)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str] = None,
    bool_cols: List[str] = None,
    cat_cols: List[str] = None,
):
    """
    Calculates and plots the percentage of missing values, coloring bars by column type.
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        print("No missing values found in the DataFrame.")
        return

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

    palette = {
        col: color_map.get(col_to_type.get(col, "other"), "gray")
        for col in missing_pct.index
    }

    plt.figure(figsize=(15, 8))
    sns.barplot(
        x=missing_pct.index,
        y=missing_pct.values,
        hue=missing_pct.index,
        palette=palette,
        legend=False,
    )

    plt.title("Percentage of Missing Values by Column", fontsize=16)
    plt.xlabel("Columns")
    plt.ylabel("% of Missing Values")
    plt.xticks(rotation=45, ha="right")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map.get(type_name), label=type_name.capitalize())
        for type_name in color_map
        if type_name in col_to_type.values()
    ]
    plt.legend(handles=legend_elements, title="Column Type")

    plt.show()

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

    plt.tight_layout()
    plt.show()


def plot_interaction(
    df: pd.DataFrame, target_col: str, x_cols: Union[str, List[str]], facet_col: str
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
        plt.show()


def plot_numeric_vs_target(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    n_cols: int = 2,
    sample_size: int = 2000,
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

    plt.tight_layout()
    plt.show()


def plot_decision_tree(tree_model, feature_names, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    plot_tree(
        tree_model.named_steps["regressor"],
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Reglas del Árbol de Decisión")
    plt.show()


def plot_feature_importance(xgb_model, feature_names):
    xgb_regressor = xgb_model.named_steps["regressor"]

    importances = xgb_regressor.feature_importances_

    feat_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feat_imp_df.head(20),
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Top 20 Features más Importantes (XGBoost)")
    plt.xlabel("Importancia (Gain)")
    plt.show()


def plot_model_comparison(models_predictions: Dict[str, List]):
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
    fig.suptitle("Impacto del Preprocesamiento en Modelos", fontsize=16)

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

        ax.set_title(title)

        # Formato de eje Y
        ax.ticklabel_format(style="plain", axis="y")

        # Etiquetas de valores sobre las barras
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=9)

        if i == 0:
            ax.legend(title=hue_col, loc="upper right")
        else:
            if ax.get_legend():
                ax.legend_.remove()

    plt.tight_layout()
    plt.show()

    return df_metrics


def plot_cv_results(
    results_df: pd.DataFrame,
    baseline_name: str = "Baseline",
    vertical_layout: bool = True,
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

    plt.tight_layout()
    plt.show()


def plot_grid_search_results(
    results_df: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple = (16, 7),
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

    plt.tight_layout()
    plt.show()


def plot_learning_curve(
    evals_result: dict,
    metric: str = "rmse",
    figsize: tuple = (14, 5),
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


def plot_classifier_results(
    results_df: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple = (16, 18),
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
        ('test_recall', 'Top N por Recall', 'coral'),
        ('test_f1', 'Top N por F1', 'steelblue'),
        ('test_avg_precision', 'Top N por Avg Precision', 'mediumseagreen'),
        ('test_precision', 'Top N por Precision', 'goldenrod'),
    ]
    
    # ============================================================
    # Bar charts (primeras 4 posiciones)
    # ============================================================
    bar_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (metric, title, color) in enumerate(metrics_config):
        row, col = bar_positions[idx]
        ax = axes[row, col]
        
        # Nombre de columna test
        metric_name = metric.replace('test_', '')
        
        # Ordenar por esta métrica y tomar top N
        top_data = results_df.sort_values(metric, ascending=False).head(top_n)
        
        # Labels
        if id_col:
            labels = [f"ID {int(r[id_col])}" for _, r in top_data.iterrows()]
        else:
            labels = [f"#{i+1}" for i in range(len(top_data))]
        
        x_pos = np.arange(len(top_data))
        
        bars = ax.bar(x_pos, top_data[metric], color=color, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Destacar el mejor
        bars[0].set_edgecolor('darkred')
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
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)


    
    # ============================================================
    # Scatter: Precision vs Recall (color = F1)
    # ============================================================
    ax = axes[2, 0]
    scatter = ax.scatter(
        results_df['test_precision'], 
        results_df['test_recall'],
        c=results_df['test_f1'],
        cmap='RdYlGn',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    # Marcar el mejor F1
    best_f1_idx = results_df['test_f1'].idxmax()
    best = results_df.loc[best_f1_idx]
    ax.scatter(best['test_precision'], best['test_recall'], 
               s=200, facecolors='none', edgecolors='red', linewidth=2.5, label='Best F1')
    
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
        results_df['test_roc_auc'], 
        results_df['test_f1'],
        c=results_df['test_recall'],
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, ax=ax, label='Recall')
    
    # Marcar el mejor F1
    ax.scatter(best['test_roc_auc'], best['test_f1'], 
               s=200, facecolors='none', edgecolors='red', linewidth=2.5, label='Best F1')
    
    ax.set_xlabel("ROC-AUC", fontweight="bold")
    ax.set_ylabel("F1", fontweight="bold")
    ax.set_title("ROC-AUC vs F1 (color=Recall)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen del mejor modelo
    print("\n" + "="*60)
    print("🏆 MEJOR MODELO (por F1):")
    print("="*60)
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
            edgecolor='black',
            alpha=0.9
        )

        # Estética
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontweight="bold")
        ax.set_ylabel(metric.upper(), fontweight="bold")
        ax.set_title(f"Ranking por {metric.upper()}", fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Agregar valores
        for bar, val in zip(bars_test, data[metric]):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                val + 0.005, 
                f"{val:.3f}", 
                ha='center', 
                va='bottom',
                fontsize=9, 
                fontweight='bold',
                color='darkred'
            )

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

