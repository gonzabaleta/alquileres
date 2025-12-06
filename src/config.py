from src.pipeline import (
    PipelineConfig,
    FeatureCreatorParams,
    OutlierClipperParams,
    TargetParams,
)
from src.utils import COLS

# Parámetros compartidos en varias configuraciones
COLS_TO_DROP = [COLS.SUP_TOTAL]
FEATURE_CREATOR_PARAMS = FeatureCreatorParams(
    add_amenities_score=True,
    add_room_density=True,
    add_bath_bed_ratio=True,
    add_uncovered_pct=True,
)
OUTLIER_CLIPPER_PARAMS = OutlierClipperParams(
    cols_to_clip=[
        COLS.SUP_CONSTR,
        COLS.SUP_DESCUBIERTA,
        COLS.ANTIGUEDAD,
        COLS.DORMITORIOS,
        COLS.BANOS,
        COLS.AMBIENTES,
        COLS.COCHERAS,
    ],
    upper_pct=0.98,
)
MEDIAN_IMPUTER_COLS = [
    COLS.ANTIGUEDAD,
    COLS.SUP_DESCUBIERTA,
    COLS.SUP_CONSTR,
    COLS.SUP_DESCUBIERTA_PCT,
    COLS.M2_POR_AMBIENTE,
    COLS.BANOS_POR_DORMITORIO,
    COLS.AMENITIES_SCORE,
]
MODE_IMPUTATION_COLS = [COLS.AMBIENTES, COLS.DORMITORIOS, COLS.BANOS, COLS.COCHERAS]
LOG_STD_COLS = [COLS.SUP_CONSTR, COLS.SUP_DESCUBIERTA, COLS.ANTIGUEDAD]
STD_ONLY_COLS = [COLS.LONGITUD, COLS.LATITUD]
BOOLEAN_IMPUTER_COLS = [
    COLS.AMOBLADO,
    COLS.GIMNASIO,
    COLS.LAUNDRY,
    COLS.CALEFACCION,
    COLS.AIRE,
    COLS.SEGURIDAD,
    COLS.PILETA,
    COLS.TENNIS,
    COLS.RECEPCION,
    COLS.BUSINESS,
    COLS.ESTACIONAMIENTO,
    "SUM",
    COLS.JACUZZI,
]
ONE_HOT_COLS = [COLS.PROVINCIA, COLS.CONDICION, COLS.ANIO]

# Pipeline de preprocesamiento para decision trees
decision_trees_pipeline_config = PipelineConfig(
    cols_to_drop=COLS_TO_DROP,
    feature_creator_params=FEATURE_CREATOR_PARAMS,
    median_imputer_cols=MEDIAN_IMPUTER_COLS,
    outlier_clipper_params=OUTLIER_CLIPPER_PARAMS,
    mode_imputation_cols=MODE_IMPUTATION_COLS,
    log_std_cols=[],  # Trees no necesitan estandarización
    std_only_cols=[],
    boolean_imputer_cols=BOOLEAN_IMPUTER_COLS,
    one_hot_cols=ONE_HOT_COLS,
    target_params=TargetParams(
        target_clipper_params=OutlierClipperParams(
            cols_to_clip=[COLS.TARGET], upper_pct=1  # no clipeamos target
        ),
        log_transform=False,
    ),
    ordinal_cols=[COLS.BARRIO, COLS.CIUDAD],
    target_encode_cols=[COLS.ID_GRID],
)

# Definimos las columnas que necesitan escalado para modelos lineales (Todas las numéricas)
LINEAR_STD_COLS = [
    COLS.LONGITUD,
    COLS.LATITUD,
    COLS.SUP_DESCUBIERTA_PCT,
    COLS.M2_POR_AMBIENTE,
    COLS.BANOS_POR_DORMITORIO,
    COLS.AMENITIES_SCORE,
    COLS.AMBIENTES,
    COLS.DORMITORIOS,
    COLS.BANOS,
    COLS.COCHERAS,
]

# Modelos lineales van a tener el target transformado y clipeado
linear_models_pipeline_config = PipelineConfig(
    cols_to_drop=COLS_TO_DROP,
    feature_creator_params=FEATURE_CREATOR_PARAMS,
    median_imputer_cols=MEDIAN_IMPUTER_COLS,
    outlier_clipper_params=OUTLIER_CLIPPER_PARAMS,
    mode_imputation_cols=MODE_IMPUTATION_COLS,
    log_std_cols=LOG_STD_COLS,
    std_only_cols=LINEAR_STD_COLS,
    boolean_imputer_cols=BOOLEAN_IMPUTER_COLS,
    one_hot_cols=ONE_HOT_COLS,
    target_params=TargetParams(
        target_clipper_params=OutlierClipperParams(
            cols_to_clip=[], upper_pct=0.98  # clipeamos outliers en target
        ),
        log_transform=True,
    ),
    ordinal_cols=[],
    target_encode_cols=[COLS.BARRIO, COLS.CIUDAD, COLS.ID_GRID],
)
