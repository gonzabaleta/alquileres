"""
Configuraciones optimizadas de XGBoost para diferentes datasets.

Cada configuración fue obtenida mediante RandomizedSearchCV (ver 08-arboles-analisis.ipynb)
"""

from xgboost import XGBRegressor
from src.pipeline import build_feature_pipeline, build_target_pipeline
from src.config import TREE_BASED_CONFIG_NORMAL, TREE_BASED_CONFIG_OUTLIERS
from src.model_evaluation import _build_full_regressor


XGBOOST_NORMAL_CONFIG = {
    "colsample_bytree": 0.7545843364636589,
    "gamma": 0.2822205453394412,
    "learning_rate": 0.0675347366775581,
    "max_depth": 14,
    "min_child_weight": 2,
    "n_estimators": 700,
    "reg_lambda": 4.9743315858488355,
    "subsample": 0.6231122243985466,
    "random_state": 42,
}

XGBOOST_OUTLIERS_CONFIG = {
    "colsample_bytree": 0.8134217734214317,
    "gamma": 0.00012050234256727466,
    "learning_rate": 0.08237330081140835,
    "max_depth": 9,
    "min_child_weight": 5,
    "n_estimators": 600,
    "reg_lambda": 0.07749120913095153,
    "subsample": 0.6667057075490768,
    "random_state": 42,
}

XGBOOST_FULL_CONFIG = {
    "colsample_bytree": 0.7815158165534557,
    "gamma": 0.11802523167323198,
    "learning_rate": 0.032049024199003795,
    "max_depth": 14,
    "min_child_weight": 1,
    "n_estimators": 900,
    "reg_lambda": 1.2435713629381495,
    "subsample": 0.846857994641612,
    "random_state": 42,
}


def get_xgboost_normal():
    """
    Retorna el modelo XGBoost optimizado para el dataset normal.
    Incluye preprocessing moderado (feature + target pipeline).

    Returns:
        TransformedTargetRegressor con el modelo completo
    """
    feature_pipeline = build_feature_pipeline(TREE_BASED_CONFIG_NORMAL)
    target_pipeline = build_target_pipeline(TREE_BASED_CONFIG_NORMAL)

    model = XGBRegressor(**XGBOOST_NORMAL_CONFIG)

    return _build_full_regressor(model, feature_pipeline, target_pipeline)


def get_xgboost_outliers():
    """
    Retorna el modelo XGBoost optimizado para el dataset de outliers.
    Incluye preprocessing específico para valores extremos.

    Returns:
        TransformedTargetRegressor con el modelo completo
    """
    feature_pipeline = build_feature_pipeline(TREE_BASED_CONFIG_OUTLIERS)
    target_pipeline = build_target_pipeline(TREE_BASED_CONFIG_OUTLIERS)

    model = XGBRegressor(**XGBOOST_OUTLIERS_CONFIG)

    return _build_full_regressor(model, feature_pipeline, target_pipeline)


def get_xgboost_full():
    """
    Retorna el modelo XGBoost optimizado para el dataset completo (normal + outliers).
    Usa el pipeline normal (más robusto para mix de datos).

    Returns:
        TransformedTargetRegressor con el modelo completo
    """
    feature_pipeline = build_feature_pipeline(TREE_BASED_CONFIG_NORMAL)
    target_pipeline = build_target_pipeline(TREE_BASED_CONFIG_NORMAL)

    model = XGBRegressor(**XGBOOST_FULL_CONFIG)

    return _build_full_regressor(model, feature_pipeline, target_pipeline)
