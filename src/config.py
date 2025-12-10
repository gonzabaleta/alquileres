"""
Configuraciones de Pipelines - Ganadoras del Análisis de Preprocessing

Este archivo contiene las 4 configuraciones óptimas identificadas durante
la experimentación, listas para uso en producción.
"""

"""
Uso recomendado:

1. DATASET NORMAL (98.5% del mercado, sin outliers extremos):
   - XGBoost / Random Forest → TREE_BASED_CONFIG_NORMAL
   - Ridge / Lasso / ElasticNet → LINEAR_DEEP_LEARNING_CONFIG_NORMAL
   - Redes Neuronales → LINEAR_DEEP_LEARNING_CONFIG_NORMAL

2. DATASET CON OUTLIERS (dataset original completo):
   - XGBoost / Random Forest → TREE_BASED_CONFIG_OUTLIERS
   - Ridge / Lasso / ElasticNet → LINEAR_DEEP_LEARNING_CONFIG_OUTLIERS
   - Redes Neuronales → LINEAR_DEEP_LEARNING_CONFIG_OUTLIERS

Ejemplo de uso:
```python
from src.pipeline import build_feature_pipeline, build_target_pipeline
from src.config import TREE_BASED_CONFIG_NORMAL

feature_pipeline = build_feature_pipeline(TREE_BASED_CONFIG_NORMAL)
target_pipeline = build_target_pipeline(TREE_BASED_CONFIG_NORMAL)
```
"""


from src.pipeline import (
    PipelineConfig,
    FeatureCreatorParams,
    OutlierClipperParams,
    TargetParams,
)
from src.constants import COLS

# ============================================================
# CONFIGURACIONES PARA DATASET NORMAL (98.5% sin outliers extremos)
# ============================================================

# Para modelos basados en árboles (XGBoost, Random Forest, etc.)
TREE_BASED_CONFIG_NORMAL = PipelineConfig(
    cols_to_drop=[COLS.SUP_TOTAL],
    feature_creator_params=FeatureCreatorParams(
        add_amenities_score=True,
        add_room_density=True,
        add_bath_bed_ratio=True,
        add_uncovered_pct=True,
    ),
    median_imputer_cols=[
        COLS.ID_GRID,
        COLS.SUP_CONSTR,
        COLS.DORMITORIOS,
        COLS.BANOS,
        COLS.AMBIENTES,
        COLS.COCHERAS,
        COLS.LONGITUD,
        COLS.LATITUD,
        COLS.ANIO,
        COLS.MES,
        COLS.SUP_DESCUBIERTA,
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
        COLS.ANTIGUEDAD,
    ],
    outlier_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
    mode_imputation_cols=[COLS.CIUDAD, COLS.PROVINCIA, COLS.BARRIO, COLS.CONDICION],
    log_cols=[],  # Árboles no necesitan transformación de features
    std_cols=[],  # Árboles no necesitan escalado
    boolean_imputer_cols=[
        COLS.AMOBLADO,
        COLS.BUSINESS,
        COLS.GIMNASIO,
        COLS.LAUNDRY,
        COLS.CALEFACCION,
        COLS.AIRE,
        COLS.RECEPCION,
        COLS.ESTACIONAMIENTO,
        COLS.JACUZZI,
        COLS.SEGURIDAD,
        COLS.PILETA,
        COLS.TENNIS,
        "SUM",
    ],
    one_hot_cols=[COLS.PROVINCIA, COLS.CONDICION, COLS.ANIO],
    target_params=TargetParams(
        target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
        log_transform=False,  # Sin log target
    ),
    ordinal_cols=[COLS.BARRIO, COLS.CIUDAD, COLS.MES],
    target_encode_cols=[],
)


# Columnas numéricas para escalar (requeridas por modelos lineales y redes neuronales)
NUMERIC_COLS_FOR_SCALING = [
    COLS.SUP_CONSTR,
    COLS.ANTIGUEDAD,
    COLS.DORMITORIOS,
    COLS.BANOS,
    COLS.AMBIENTES,
    COLS.COCHERAS,
    COLS.LONGITUD,
    COLS.LATITUD,
    COLS.ANIO,
    COLS.MES,
    COLS.ID_GRID,
]


# Para modelos lineales y Deep Learning (Regresión Lineal, Ridge, Redes Neuronales)
LINEAR_DEEP_LEARNING_CONFIG_NORMAL = PipelineConfig(
    cols_to_drop=[COLS.SUP_TOTAL],
    feature_creator_params=FeatureCreatorParams(
        add_amenities_score=True,
        add_room_density=True,
        add_bath_bed_ratio=True,
        add_uncovered_pct=True,
    ),
    median_imputer_cols=[
        COLS.ID_GRID,
        COLS.SUP_CONSTR,
        COLS.ANTIGUEDAD,
        COLS.DORMITORIOS,
        COLS.BANOS,
        COLS.AMBIENTES,
        COLS.COCHERAS,
        COLS.LONGITUD,
        COLS.LATITUD,
        COLS.ANIO,
        COLS.MES,
        COLS.SUP_DESCUBIERTA,
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
    ],
    outlier_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
    mode_imputation_cols=[COLS.CIUDAD, COLS.PROVINCIA, COLS.CONDICION],
    log_cols=[COLS.SUP_CONSTR, COLS.ANTIGUEDAD],  # Normalizar distribuciones sesgadas
    std_cols=[
        col
        for col in NUMERIC_COLS_FOR_SCALING
        if col not in [COLS.SUP_CONSTR, COLS.ANTIGUEDAD]
    ]
    + [
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
    ],
    boolean_imputer_cols=[
        COLS.AMOBLADO,
        COLS.BUSINESS,
        COLS.GIMNASIO,
        COLS.LAUNDRY,
        COLS.CALEFACCION,
        COLS.AIRE,
        COLS.RECEPCION,
        COLS.ESTACIONAMIENTO,
        COLS.JACUZZI,
        COLS.SEGURIDAD,
        COLS.PILETA,
        COLS.TENNIS,
        "SUM",
    ],
    one_hot_cols=[COLS.PROVINCIA, COLS.CONDICION],  # One-hot para baja cardinalidad
    target_params=TargetParams(
        target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
        log_transform=False,  # Sin log target en dataset normal
    ),
    ordinal_cols=[COLS.MES],
    target_encode_cols=[
        COLS.ID_GRID,
        COLS.BARRIO,
        COLS.CIUDAD,
    ],  # Para alta cardinalidad
)


# ============================================================
# CONFIGURACIONES PARA DATASET CON OUTLIERS (original completo)
# ============================================================

# Para modelos basados en árboles con outliers extremos
TREE_BASED_CONFIG_OUTLIERS = PipelineConfig(
    cols_to_drop=[COLS.SUP_TOTAL],
    feature_creator_params=FeatureCreatorParams(
        add_amenities_score=True,
        add_room_density=True,
        add_bath_bed_ratio=True,
        add_uncovered_pct=True,
    ),
    median_imputer_cols=[
        COLS.ID_GRID,
        COLS.SUP_CONSTR,
        COLS.DORMITORIOS,
        COLS.BANOS,
        COLS.AMBIENTES,
        COLS.COCHERAS,
        COLS.LONGITUD,
        COLS.LATITUD,
        COLS.ANIO,
        COLS.MES,
        COLS.SUP_DESCUBIERTA,
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
        COLS.ANTIGUEDAD,
    ],
    outlier_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
    mode_imputation_cols=[COLS.CIUDAD, COLS.PROVINCIA, COLS.BARRIO, COLS.CONDICION],
    log_cols=[],
    std_cols=[],
    boolean_imputer_cols=[
        COLS.AMOBLADO,
        COLS.BUSINESS,
        COLS.GIMNASIO,
        COLS.LAUNDRY,
        COLS.CALEFACCION,
        COLS.AIRE,
        COLS.RECEPCION,
        COLS.ESTACIONAMIENTO,
        COLS.JACUZZI,
        COLS.SEGURIDAD,
        COLS.PILETA,
        COLS.TENNIS,
        "SUM",
    ],
    one_hot_cols=[COLS.PROVINCIA, COLS.CONDICION, COLS.ANIO],
    target_params=TargetParams(
        target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
        log_transform=True,  # Log target para comprimir escala
    ),
    ordinal_cols=[COLS.BARRIO, COLS.CIUDAD, COLS.MES],
    target_encode_cols=[],
)


# Para modelos lineales y Deep Learning con outliers extremos
LINEAR_DEEP_LEARNING_CONFIG_OUTLIERS = PipelineConfig(
    cols_to_drop=[COLS.SUP_TOTAL],
    feature_creator_params=FeatureCreatorParams(
        add_amenities_score=True,
        add_room_density=True,
        add_bath_bed_ratio=True,
        add_uncovered_pct=True,
    ),
    median_imputer_cols=[
        COLS.ID_GRID,
        COLS.SUP_CONSTR,
        COLS.ANTIGUEDAD,
        COLS.DORMITORIOS,
        COLS.BANOS,
        COLS.AMBIENTES,
        COLS.COCHERAS,
        COLS.LONGITUD,
        COLS.LATITUD,
        COLS.ANIO,
        COLS.MES,
        COLS.SUP_DESCUBIERTA,
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
    ],
    outlier_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
    mode_imputation_cols=[COLS.CIUDAD, COLS.PROVINCIA, COLS.CONDICION],
    log_cols=[COLS.SUP_CONSTR, COLS.ANTIGUEDAD],
    std_cols=[
        col
        for col in NUMERIC_COLS_FOR_SCALING
        if col not in [COLS.SUP_CONSTR, COLS.ANTIGUEDAD]
    ]
    + [
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
    ],
    boolean_imputer_cols=[
        COLS.AMOBLADO,
        COLS.BUSINESS,
        COLS.GIMNASIO,
        COLS.LAUNDRY,
        COLS.CALEFACCION,
        COLS.AIRE,
        COLS.RECEPCION,
        COLS.ESTACIONAMIENTO,
        COLS.JACUZZI,
        COLS.SEGURIDAD,
        COLS.PILETA,
        COLS.TENNIS,
        "SUM",
    ],
    one_hot_cols=[COLS.PROVINCIA, COLS.CONDICION],
    target_params=TargetParams(
        target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
        log_transform=True,  # Log target para comprimir escala
    ),
    ordinal_cols=[COLS.MES],
    target_encode_cols=[COLS.ID_GRID, COLS.BARRIO, COLS.CIUDAD],
)
