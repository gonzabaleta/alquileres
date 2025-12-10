from enum import Enum

DEV_SET_RAW_PATH = "data/raw/alquiler_AMBA_dev.csv"
DEV_SET_CLEAN_PATH = "data/processed/dev_set_clean.csv"
DEV_SET_CLEAN_NORMAL_PATH = "data/processed/dev_set_clean_normal.csv"
DEV_SET_CLEAN_OUTLIERS_PATH = "data/processed/dev_set_clean_outliers.csv"
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
    SUP_DESCUBIERTA_PCT = "pct_descubierto"
    BANOS_POR_DORMITORIO = "Banos_por_dormitorio"
    M2_POR_AMBIENTE = "M2_por_ambiente"
    DORMITORIOS = "Dormitorios"
    BANOS = "Banos"
    AMBIENTES = "Ambientes"
    ANTIGUEDAD = "Antiguedad"
    COCHERAS = "Cocheras"
    AMENITIES_SCORE = "amenities_score"

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
    SALON_USOS_MULTIPLES = "SalonDeUsosMul"
    SUM = "SUM"
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


class DevSetType(Enum):
    RAW = "raw"
    CLEAN = "clean"
    NORMAL = "normal"
    OUTLIERS = "outliers"


# --- Mapeo de Nombres de Columnas a Nombres Legibles ---

COLUMN_NAMES_LEGIBLE = {
    # --- Identificadores y Target ---
    COLS.ID_GRID: "ID de Grilla",
    COLS.TARGET: "Precio (Pesos)",
    # --- Features Principales ---
    COLS.MES_LISTING: "Mes de Publicación",
    COLS.TIPO_PROPIEDAD: "Tipo de Propiedad",
    COLS.SUP_TOTAL: "Sup. Total (m²)",
    COLS.SUP_CONSTR: "Sup. Construida (m²)",
    COLS.SUP_DESCUBIERTA: "Sup. Descubierta (m²)",
    COLS.SUP_DESCUBIERTA_PCT: "Sup. Descubierta (%)",
    COLS.BANOS_POR_DORMITORIO: "Baños por Dormitorio",
    COLS.M2_POR_AMBIENTE: "m² por Ambiente",
    COLS.DORMITORIOS: "Dormitorios",
    COLS.BANOS: "Baños",
    COLS.AMBIENTES: "Ambientes",
    COLS.ANTIGUEDAD: "Antigüedad (años)",
    COLS.COCHERAS: "Cocheras",
    COLS.AMENITIES_SCORE: "Amenities Score",
    # --- Features de Ubicación ---
    COLS.CIUDAD: "Localidad",
    COLS.PROVINCIA: "Ciudad",
    COLS.BARRIO: "Barrio",
    COLS.LONGITUD: "Longitud",
    COLS.LATITUD: "Latitud",
    # --- Features Booleanas (Amenities) ---
    COLS.AMOBLADO: "Amoblado",
    COLS.CISTERNA: "Cisterna",
    COLS.INTERNET: "Acceso a Internet",
    COLS.BUSINESS: "Business Center",
    COLS.GIMNASIO: "Gimnasio",
    COLS.LAUNDRY: "Laundry",
    COLS.CALEFACCION: "Calefacción",
    COLS.SALON_USOS_MULTIPLES: "Salón de Usos Múltiples",
    COLS.SUM: "SUM",
    COLS.AIRE: "Aire Acondicionado",
    COLS.RECEPCION: "Recepción",
    COLS.ESTACIONAMIENTO: "Estacionamiento",
    COLS.JACUZZI: "Jacuzzi",
    COLS.JUEGOS: "Área de Juegos Infantiles",
    COLS.CHIMENEA: "Chimenea",
    COLS.ASCENSOR: "Ascensor",
    COLS.SALON_FIESTAS: "Salón de Fiestas",
    COLS.SEGURIDAD: "Seguridad",
    COLS.PILETA: "Pileta",
    COLS.PISTA_JOGGING: "Pista de Jogging",
    COLS.ESTACIONAMIENTO_VISITAS: "Estacionamiento para Visitas",
    COLS.LOBBY: "Lobby",
    COLS.LOCALES: "Locales Comerciales",
    COLS.SIST_INCENDIOS: "Sistema Contra Incendios",
    COLS.PARRILLAS: "Área de Parrillas",
    COLS.TENNIS: "Cancha de Tennis",
    COLS.CINE: "Área de Cine",
    COLS.LUXURY: "Amenities de Lujo",
    # --- Otras ---
    COLS.SITIO_ORIGEN: "Sitio de Origen",
    COLS.CONDICION: "Condición",
    COLS.ANIO: "Año",
    COLS.MES: "Mes",
}
