import pandas as pd
from typing import List

DEV_SET_RAW_PATH = "data/raw/alquiler_AMBA_dev.csv"
DEV_SET_CLEAN_PATH = "data/processed/dev_set_clean.csv"
TEST_SET_RAW_PATH = "data/alquiler_AMBA_test.csv"
TARGET = "precio_pesos_constantes"


class COLS:
    """
    Clase contenedora para los nombres de las columnas del dataset.
    Permite un acceso centralizado y previene errores de tipeo.
    """

    # --- Identificadores y Target ---
    ID_GRID = "id_grid"
    TARGET = TARGET

    # --- Features Principales ---
    MES_LISTING = "MesListing"
    TIPO_PROPIEDAD = "TIPOPROPIEDAD"
    SUP_TOTAL = "STotalM2"
    SUP_CONSTR = "SConstrM2"
    SUP_DESCUBIERTA = "SDescubiertaM2"
    DORMITORIOS = "Dormitorios"
    BANOS = "Banos"
    AMBIENTES = "Ambientes"
    ANTIGUEDAD = "Antiguedad"
    COCHERAS = "Cocheras"

    # --- Features de Ubicación ---
    CIUDAD = "ITE_ADD_CITY_NAME"
    PROVINCIA = "ITE_ADD_STATE_NAME"
    BARRIO = "ITE_ADD_NEIGHBORHOOD_NAME"
    LONGITUD = "LONGITUDE"
    LATITUD = "LATITUDE"

    # --- Features Booleanas (Amenities) ---
    AMOBLADO = "Amoblado"
    CISTERNA = "Cisterna"
    INTERNET = "AccesoInternet"
    BUSINESS = "BusinessCenter"
    GIMNASIO = "Gimnasio"
    LAUNDRY = "Laundry"
    CALEFACCION = "Calefaccion"
    SUM = "SalonDeUsosMul"
    AIRE = "AireAC"
    RECEPCION = "Recepcion"
    ESTACIONAMIENTO = "Estacionamiento"
    JACUZZI = "Jacuzzi"
    JUEGOS = "AreaJuegosInfantiles"
    CHIMENEA = "Chimenea"
    ASCENSOR = "Ascensor"
    SALON_FIESTAS = "SalonFiestas"
    SEGURIDAD = "Seguridad"
    PILETA = "Pileta"
    PISTA_JOGGING = "PistaJogging"
    ESTACIONAMIENTO_VISITAS = "EstacionamientoVisitas"
    LOBBY = "Lobby"
    LOCALES = "LocalesComerciales"
    SIST_INCENDIOS = "SistContraIncendios"
    PARRILLAS = "AreaParrillas"
    TENNIS = "CanchaTennis"
    CINE = "AreaCine"
    LUXURY = "LuxuryAmenities"

    # --- Otras ---
    SITIO_ORIGEN = "SitioOrigen"
    CONDICION = "ITE_TIPO_PROD"  # Usado, Nuevo, Sin Clasificar
    ANIO = "year"
    MES = "mes_listing"


CATEGORICAL_COLS = [
    COLS.TIPO_PROPIEDAD,
    COLS.MES_LISTING,
    COLS.SITIO_ORIGEN,
    COLS.AMOBLADO,
    COLS.CISTERNA,
    COLS.INTERNET,
    COLS.BUSINESS,
    COLS.GIMNASIO,
    COLS.LAUNDRY,
    COLS.CALEFACCION,
    COLS.SUM,
    COLS.AIRE,
    COLS.RECEPCION,
    COLS.ESTACIONAMIENTO,
    COLS.JACUZZI,
    COLS.JUEGOS,
    COLS.CHIMENEA,
    COLS.ASCENSOR,
    COLS.SALON_FIESTAS,
    COLS.SEGURIDAD,
    COLS.PILETA,
    COLS.ESTACIONAMIENTO_VISITAS,
    COLS.SIST_INCENDIOS,
    COLS.TENNIS,
    COLS.CINE,
    COLS.CIUDAD,
    COLS.PROVINCIA,
    COLS.BARRIO,
    COLS.CONDICION,
]


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
