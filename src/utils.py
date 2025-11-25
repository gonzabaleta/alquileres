import pandas as pd
from typing import List

DEV_SET_RAW_PATH = "data/raw/alquiler_AMBA_dev.csv"
DEV_SET_CLEAN_PATH = "data/processed/dev_set_clean.csv"
TEST_SET_RAW_PATH = "data/alquiler_AMBA_test.csv"
TARGET = "precio_pesos_constantes"

CATEGORICAL_COLS = [
    "TIPOPROPIEDAD",
    "MesListing",
    "SitioOrigen",
    "Amoblado",
    "Cisterna",
    "AccesoInternet",
    "BusinessCenter",
    "Gimnasio",
    "Laundry",
    "Calefaccion",
    "SalonDeUsosMul",
    "AireAC",
    "Recepcion",
    "Estacionamiento",
    "Jacuzzi",
    "AreaJuegosInfantiles",
    "Chimenea",
    "Ascensor",
    "SalonFiestas",
    "Seguridad",
    "Pileta",
    "EstacionamientoVisitas",
    "SistContraIncendios",
    "CanchaTennis",
    "AreaCine",
    "ITE_ADD_CITY_NAME",
    "ITE_ADD_STATE_NAME",
    "ITE_ADD_NEIGHBORHOOD_NAME",
    "ITE_TIPO_PROD",
]


def analizar_columnas_categoricas(df: pd.DataFrame, columnas: list):
    """
    Analiza una lista de columnas categóricas de un DataFrame y muestra un resumen.

    Para cada columna, muestra:
    - Valores únicos con conteos y porcentajes.
    - Cantidad y porcentaje de valores NaN.
    - Si es potencialmente booleana.

    Al final, imprime un resumen de columnas potencialmente booleanas y de baja varianza.
    """
    potentially_boolean_cols = []
    low_variance_boolean_cols = []

    for col in columnas:
        if col not in df.columns:
            print(f"--- Columna: {col} (NO ENCONTRADA) ---\n")
            continue

        print(f"--- Columna: {col} ---")

        # 1. Valores únicos, conteos y porcentajes
        total_count = len(df[col])
        value_counts = df[col].value_counts(dropna=False)
        value_percentages = df[col].value_counts(normalize=True, dropna=False) * 100

        print("Valores únicos, conteo y porcentaje:")
        for value, count in value_counts.items():
            percentage = value_percentages[value]
            print(f"  - {value}: {count} ({percentage:.2f}%)")

        # 2. Cantidad y porcentaje de NaN
        nan_count = df[col].isnull().sum()
        nan_percentage = (nan_count / total_count) * 100
        # La información de NaN ya está en el bucle anterior si dropna=False,
        # pero lo dejamos explícito para mayor claridad.
        print(f"\nCantidad de NaN: {nan_count} ({nan_percentage:.2f}%)")

        # 3. Potencialmente booleana
        unique_values_str = set(str(v).lower() for v in df[col].unique() if pd.notna(v))
        boolean_markers = {"si", "sí", "no"}
        is_potentially_boolean = any(
            marker in unique_values_str for marker in boolean_markers
        )

        print(f"Potencialmente booleana?: {'Sí' if is_potentially_boolean else 'No'}")

        if is_potentially_boolean:
            potentially_boolean_cols.append(col)

            # Mapeo simple para identificar 'Sí' y 'No'
            bool_map = {
                "si": True,
                "sí": True,
                "yes": True,
                "no": False,
            }
            # Convertir a booleano de forma segura
            bool_series = df[col].str.lower().map(bool_map)

            # Calcular porcentajes de True/False ignorando nulos
            if bool_series.count() > 0:  # Evitar división por cero si solo hay NaNs
                true_percentage = (
                    bool_series.value_counts(normalize=True).get(True, 0) * 100
                )
                false_percentage = (
                    bool_series.value_counts(normalize=True).get(False, 0) * 100
                )

                if true_percentage < 10 or false_percentage < 10:
                    low_variance_boolean_cols.append(col)

        print("-" * (len(col) + 16) + "\n")

    # Resumen final
    print("\n--- Resumen Final ---")
    print("Columnas potencialmente booleanas:")
    if potentially_boolean_cols:
        for col in potentially_boolean_cols:
            print(f"  - {col}")
    else:
        print("  (Ninguna)")

    print("\nColumnas booleanas con menos de 10% de 'Sí' o 'No' (baja varianza):")
    if low_variance_boolean_cols:
        for col in low_variance_boolean_cols:
            print(f"  - {col}")
    else:
        print("  (Ninguna)")


def get_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Filters a list of columns, returning only those that exist in the DataFrame.

    Args:
        df: The DataFrame to check against.
        columns: A list of column names to validate.

    Returns:
        A list of column names that are present in the DataFrame.
    """
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = set(columns) - set(existing_cols)
    if missing_cols:
        print(
            f"Warning: The following columns were not found in the DataFrame and will be ignored: {', '.join(missing_cols)}"
        )
    return existing_cols
