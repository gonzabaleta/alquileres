"""
Configuraciones de Preprocessing para Experimentos

Este archivo contiene todas las configuraciones de pipeline probadas
durante la fase de experimentación de preprocessing.

LAS CONFIGURACIONES DE ESTE ARCHIVO SON PARA EXPERIMENTOS Y NO DEBERÍAN SER USADAS
PARA EL MODELO FINAL!!!!
"""

import copy
import sys

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate
from tqdm import tqdm

from src.pipeline import build_feature_pipeline, build_target_pipeline

sys.path.append("..")

from src.config import (
    PipelineConfig,
    FeatureCreatorParams,
    OutlierClipperParams,
    TargetParams,
)
from src.constants import COLS


def evaluate_pipeline_configs(
    model_factory,
    pipeline_configs: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

    results = {}

    for config_name, config in tqdm(
        pipeline_configs.items(), desc="Evaluando Configuraciones"
    ):
        # Construir pipelines
        feature_pipeline = build_feature_pipeline(config)
        target_pipeline = build_target_pipeline(config)

        # Crear modelo usando la factory
        model = model_factory(feature_pipeline)

        # Envolver en TransformedTargetRegressor
        final_regressor = TransformedTargetRegressor(
            regressor=model, transformer=target_pipeline, check_inverse=False
        )

        # Cross-Validation
        cv_results = cross_validate(
            final_regressor,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        # Guardar resultados
        results[config_name] = {
            "rmse_mean": -np.mean(cv_results["test_rmse"]),
            "rmse_std": np.std(cv_results["test_rmse"]),
            "mae_mean": -np.mean(cv_results["test_mae"]),
            "mae_std": np.std(cv_results["test_mae"]),
        }

    # Convertir a DataFrame para facilitar análisis
    results_df = pd.DataFrame(results).T
    return results_df


# ============================================================
# DEFINICIÓN DE COLUMNAS BASE
# ============================================================

numeric_cols = [
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
    COLS.ANTIGUEDAD,
]

categorical_cols = [
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
    COLS.CIUDAD,
    COLS.PROVINCIA,
    COLS.BARRIO,
    COLS.CONDICION,
    COLS.SUM,
    COLS.JUEGOS,
    COLS.CISTERNA,
    COLS.ESTACIONAMIENTO_VISITAS,
]

# Columnas numéricas para escalar (crítico para modelos lineales)
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


# ============================================================
# CONFIGURACIÓN BASE
# ============================================================

base_pipeline_config = PipelineConfig(
    cols_to_drop=[COLS.SUP_TOTAL],
    feature_creator_params=FeatureCreatorParams(
        add_amenities_score=False,
        add_room_density=False,
        add_bath_bed_ratio=False,
        add_uncovered_pct=False,
    ),
    median_imputer_cols=numeric_cols,
    outlier_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
    mode_imputation_cols=categorical_cols,
    log_cols=[],
    std_cols=[],
    boolean_imputer_cols=[],
    one_hot_cols=[],
    target_params=TargetParams(
        target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
        log_transform=False,
    ),
    ordinal_cols=categorical_cols,
    target_encode_cols=[],
)


# ============================================================
# CONFIGURACIONES PARA XGBOOST (Exploratorias)
# ============================================================


def get_xgboost_exploratory_configs():
    """Configuraciones exploratorias para XGBoost (primeros experimentos)"""
    configs = {"01_baseline": base_pipeline_config}

    # 1. Log target
    config_log_target = copy.deepcopy(base_pipeline_config)
    config_log_target["target_params"]["log_transform"] = True
    configs["02_log_target"] = config_log_target

    # 2. Clip target
    config_clip_target = copy.deepcopy(base_pipeline_config)
    config_clip_target["target_params"]["target_clipper_params"]["cols_to_clip"] = [
        COLS.TARGET
    ]
    config_clip_target["target_params"]["target_clipper_params"]["upper_pct"] = 0.98
    configs["03_clip_target"] = config_clip_target

    # 3. Feature engineering
    config_feat_eng = copy.deepcopy(base_pipeline_config)
    config_feat_eng["feature_creator_params"]["add_amenities_score"] = True
    config_feat_eng["feature_creator_params"]["add_room_density"] = True
    config_feat_eng["feature_creator_params"]["add_bath_bed_ratio"] = True
    config_feat_eng["feature_creator_params"]["add_uncovered_pct"] = True
    config_feat_eng["median_imputer_cols"] = numeric_cols + [
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.SUP_DESCUBIERTA,
    ]
    configs["04_feat_eng"] = config_feat_eng

    # 4. Target encoding - ID Grid
    config_target_enc = copy.deepcopy(base_pipeline_config)
    config_target_enc["target_encode_cols"] = [COLS.ID_GRID]
    configs["05_target_enc_id_grid"] = config_target_enc

    # 5. Target encoding - ID Grid + Barrio + Ciudad
    target_enc2_cols = [COLS.ID_GRID, COLS.BARRIO, COLS.CIUDAD]
    config_target_enc2 = copy.deepcopy(base_pipeline_config)
    config_target_enc2["target_encode_cols"] = target_enc2_cols
    config_target_enc2["ordinal_cols"] = [
        c for c in categorical_cols if c not in target_enc2_cols
    ]
    configs["06_target_enc_full"] = config_target_enc2

    # 6. One hot encoding
    one_hot_cols = [COLS.PROVINCIA, COLS.CONDICION, COLS.ANIO]
    config_one_hot = copy.deepcopy(base_pipeline_config)
    config_one_hot["one_hot_cols"] = one_hot_cols
    config_one_hot["ordinal_cols"] = [
        c for c in categorical_cols if c not in one_hot_cols
    ]
    configs["07_one_hot"] = config_one_hot

    # 7. Sin imputación (solo para verificar)
    config_no_impute = copy.deepcopy(base_pipeline_config)
    config_no_impute["median_imputer_cols"] = []
    config_no_impute["mode_imputation_cols"] = []
    config_no_impute["boolean_imputer_cols"] = []
    configs["08_no_impute"] = config_no_impute

    # 8. Scaling + Log features
    config_std = copy.deepcopy(base_pipeline_config)
    config_std["std_cols"] = [
        COLS.SUP_CONSTR,
        COLS.ANTIGUEDAD,
        COLS.LONGITUD,
        COLS.LATITUD,
    ]
    config_std["log_cols"] = [COLS.SUP_CONSTR, COLS.ANTIGUEDAD]
    configs["09_std_log"] = config_std

    # 9. Scaling completo
    config_std_full = copy.deepcopy(base_pipeline_config)
    config_std_full["std_cols"] = [
        COLS.SUP_CONSTR,
        COLS.ANTIGUEDAD,
        COLS.DORMITORIOS,
        COLS.BANOS,
        COLS.AMBIENTES,
        COLS.COCHERAS,
        COLS.LATITUD,
        COLS.LONGITUD,
    ]
    config_std_full["log_cols"] = [
        COLS.SUP_CONSTR,
        COLS.ANTIGUEDAD,
        COLS.DORMITORIOS,
        COLS.BANOS,
        COLS.AMBIENTES,
        COLS.COCHERAS,
    ]
    configs["10_std_full"] = config_std_full

    # 10. Clip features
    config_clip_features = copy.deepcopy(base_pipeline_config)
    config_clip_features["outlier_clipper_params"] = OutlierClipperParams(
        cols_to_clip=[
            COLS.SUP_CONSTR,
            COLS.ANTIGUEDAD,
            COLS.DORMITORIOS,
            COLS.BANOS,
            COLS.AMBIENTES,
            COLS.COCHERAS,
        ],
        upper_pct=0.98,
    )
    configs["11_clip_features"] = config_clip_features

    return configs


def get_xgboost_final_configs():
    """Configuraciones finales para XGBoost (después de análisis)"""

    # Config 1: Full preprocessing (sin log target)
    xgb_config1 = PipelineConfig(
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
            COLS.SUM,
            COLS.ESTACIONAMIENTO_VISITAS,
            COLS.CISTERNA,
            COLS.JUEGOS,
        ],
        one_hot_cols=[COLS.PROVINCIA, COLS.CONDICION, COLS.ANIO],
        target_params=TargetParams(
            target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
            log_transform=False,
        ),
        ordinal_cols=[COLS.BARRIO, COLS.CIUDAD, COLS.MES],
        target_encode_cols=[],
    )

    # Config 2: Con target encoding
    xgb_config2 = copy.deepcopy(xgb_config1)
    xgb_config2["target_encode_cols"] = [COLS.ID_GRID]

    # Config 3: Con log target
    xgb_config3 = copy.deepcopy(xgb_config1)
    xgb_config3["target_params"]["target_clipper_params"]["cols_to_clip"] = []
    xgb_config3["target_params"]["target_clipper_params"]["upper_pct"] = 1
    xgb_config3["target_params"]["log_transform"] = True

    return {
        "01_baseline": base_pipeline_config,
        "02_full": xgb_config1,
        "03_target_encode_id": xgb_config2,
        "04_log_target": xgb_config3,
    }


# ============================================================
# CONFIGURACIONES PARA RIDGE (Modelos Lineales)
# ============================================================


def get_ridge_configs():
    """Configuraciones para Ridge (requieren scaling obligatorio)"""

    # Config 1: Baseline con scaling
    ridge_config1 = PipelineConfig(
        cols_to_drop=[COLS.SUP_TOTAL],
        feature_creator_params=FeatureCreatorParams(
            add_amenities_score=False,
            add_room_density=False,
            add_bath_bed_ratio=False,
            add_uncovered_pct=False,
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
        ],
        outlier_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
        mode_imputation_cols=[COLS.CIUDAD, COLS.PROVINCIA, COLS.BARRIO, COLS.CONDICION],
        log_cols=[],
        std_cols=NUMERIC_COLS_FOR_SCALING,  # CRÍTICO para Ridge
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
            COLS.SUM,
            COLS.ESTACIONAMIENTO_VISITAS,
            COLS.CISTERNA,
            COLS.JUEGOS,
        ],
        one_hot_cols=[],
        target_params=TargetParams(
            target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
            log_transform=False,
        ),
        ordinal_cols=[COLS.BARRIO, COLS.CIUDAD, COLS.PROVINCIA, COLS.CONDICION],
        target_encode_cols=[],
    )

    # Config 2: Feature Engineering + Scaling
    ridge_config2 = PipelineConfig(
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
        mode_imputation_cols=[COLS.CIUDAD, COLS.PROVINCIA, COLS.BARRIO, COLS.CONDICION],
        log_cols=[],
        std_cols=NUMERIC_COLS_FOR_SCALING
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
            COLS.SUM,
            COLS.ESTACIONAMIENTO_VISITAS,
            COLS.CISTERNA,
            COLS.JUEGOS,
        ],
        one_hot_cols=[],
        target_params=TargetParams(
            target_clipper_params=OutlierClipperParams(cols_to_clip=[], upper_pct=1),
            log_transform=False,
        ),
        ordinal_cols=[COLS.BARRIO, COLS.CIUDAD, COLS.PROVINCIA, COLS.CONDICION],
        target_encode_cols=[],
    )

    # Config 3: Log features + Scaling
    ridge_config3 = copy.deepcopy(ridge_config2)
    ridge_config3["log_cols"] = [COLS.SUP_CONSTR, COLS.ANTIGUEDAD]
    ridge_config3["std_cols"] = [
        col
        for col in NUMERIC_COLS_FOR_SCALING
        if col not in [COLS.SUP_CONSTR, COLS.ANTIGUEDAD]
    ] + [
        COLS.SUP_DESCUBIERTA_PCT,
        COLS.AMENITIES_SCORE,
        COLS.M2_POR_AMBIENTE,
        COLS.BANOS_POR_DORMITORIO,
    ]

    # Config 4: Target Encoding
    ridge_config4 = copy.deepcopy(ridge_config3)
    ridge_config4["mode_imputation_cols"] = [
        COLS.CIUDAD,
        COLS.PROVINCIA,
        COLS.CONDICION,
    ]
    ridge_config4["one_hot_cols"] = [COLS.PROVINCIA, COLS.CONDICION]
    ridge_config4["ordinal_cols"] = [COLS.MES]
    ridge_config4["target_encode_cols"] = [COLS.ID_GRID, COLS.BARRIO, COLS.CIUDAD]

    # Config 5: Log Target
    ridge_config5 = copy.deepcopy(ridge_config4)
    ridge_config5["target_params"]["log_transform"] = True

    # Config 6: Clip + Log Target
    ridge_config6 = copy.deepcopy(ridge_config5)
    ridge_config6["target_params"]["target_clipper_params"]["cols_to_clip"] = [
        COLS.TARGET
    ]
    ridge_config6["target_params"]["target_clipper_params"]["upper_pct"] = 0.95

    return {
        "01_baseline_scaled": ridge_config1,
        "02_feat_eng_scaled": ridge_config2,
        "03_log_features_scaled": ridge_config3,
        "04_target_encoding": ridge_config4,
        "05_log_target_scaled": ridge_config5,
        "06_clip_log_target": ridge_config6,
    }
