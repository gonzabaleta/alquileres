from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.constants import (
    DEV_SET_RAW_PATH,
    DEV_SET_CLEAN_PATH,
    DEV_SET_CLEAN_NORMAL_PATH,
    DEV_SET_CLEAN_OUTLIERS_PATH,
    DevSetType,
)


_DEV_SET_PATHS = {
    DevSetType.RAW: DEV_SET_RAW_PATH,
    DevSetType.CLEAN: DEV_SET_CLEAN_PATH,
    DevSetType.NORMAL: DEV_SET_CLEAN_NORMAL_PATH,
    DevSetType.OUTLIERS: DEV_SET_CLEAN_OUTLIERS_PATH,
}


def get_dev_set(set_type: DevSetType) -> pd.DataFrame:
    """
    Carga un tipo específico del dataset de desarrollo.

    Args:
        set_type: El tipo de dataset a cargar, especificado por el Enum DevSet.

    Returns:
        Un DataFrame de pandas con el dataset solicitado.
        
    Raises:
        KeyError: Si el set_type no es un miembro válido de DevSet.
    """
    try:
        path = _DEV_SET_PATHS[set_type]
        return pd.read_csv("../" + path, low_memory=False)
    except KeyError:
        raise ValueError(f"Invalid dataset type: {set_type}. Must be one of {list(DevSetType)}")


def analizar_columnas_categoricas(df: pd.DataFrame, columnas: list):
    """
    Analiza una lista de columnas categóricas de un DataFrame y muestra un resumen.
    """
    potentially_boolean_cols = []
    low_variance_boolean_cols = []

    for col in columnas:
        if col not in df.columns:
            print(f"--- Columna: {col} (NO ENCONTRADA) ---\n")
            continue

        print(f"--- Columna: {col} ---")
        total_count = len(df[col])
        value_counts = df[col].value_counts(dropna=False)
        value_percentages = df[col].value_counts(normalize=True, dropna=False) * 100

        print("Valores únicos, conteo y porcentaje:")
        for value, count in value_counts.items():
            percentage = value_percentages[value]
            print(f"  - {value}: {count} ({percentage:.2f}%)")

        nan_count = df[col].isnull().sum()
        nan_percentage = (nan_count / total_count) * 100
        print(f"\nCantidad de NaN: {nan_count} ({nan_percentage:.2f}%)")

        unique_values_str = set(str(v).lower() for v in df[col].unique() if pd.notna(v))
        boolean_markers = {"si", "sí", "no"}
        is_potentially_boolean = any(
            marker in unique_values_str for marker in boolean_markers
        )

        print(f"Potencialmente booleana?: {'Sí' if is_potentially_boolean else 'No'}")

        if is_potentially_boolean:
            potentially_boolean_cols.append(col)
            bool_map = {"si": True, "sí": True, "yes": True, "no": False}
            bool_series = df[col].str.lower().map(bool_map)
            if bool_series.count() > 0:
                true_percentage = (
                    bool_series.value_counts(normalize=True).get(True, 0) * 100
                )
                if true_percentage < 10 or (100 - true_percentage) < 10:
                    low_variance_boolean_cols.append(col)

        print("-" * (len(col) + 16) + "\n")

    print("\n--- Resumen Final ---")
    print("Columnas potencialmente booleanas:")
    print(
        f"  {', '.join(potentially_boolean_cols) if potentially_boolean_cols else '(Ninguna)'}"
    )
    print("\nColumnas booleanas con menos de 10% en alguna categoría (baja varianza):")
    print(
        f"  {', '.join(low_variance_boolean_cols) if low_variance_boolean_cols else '(Ninguna)'}"
    )


def get_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Filters a list of columns, returning only those that exist in the DataFrame.
    """
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = set(columns) - set(existing_cols)
    if missing_cols:
        print(
            f"Warning: The following columns were not found in the DataFrame and will be ignored: {', '.join(missing_cols)}"
        )
    return existing_cols


def print_model_metrics(y_true, y_pred, model_name):
    """Imprime métricas de un modelo"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"=== Modelo: {model_name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print("-" * 50)
