"""
Este módulo contiene las funciones para construir pipelines de preprocesamiento
modulares y configurables para el proyecto de predicción de precios de alquileres.

Funciones principales:
- build_feature_pipeline(): Construye el pipeline de transformación de features (X)
- build_target_pipeline(): Construye el pipeline de transformación del target (y)
- get_base_xgboost(): Crea un modelo XGBoost con el feature pipeline integrado

Los pipelines se configuran mediante objetos PipelineConfig (definidos más abajo),
que especifican qué transformaciones aplicar a qué columnas.

Uso típico:
    from src.config import TREE_BASED_CONFIG_NORMAL
    from src.pipeline import build_feature_pipeline, build_target_pipeline

    feature_pipeline = build_feature_pipeline(TREE_BASED_CONFIG_NORMAL)
    target_pipeline = build_target_pipeline(TREE_BASED_CONFIG_NORMAL)
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from typing import List, TypedDict, Any, Tuple

from src.preprocessing import (
    ColumnDropper,
    FeatureCreator,
    OutlierClipper,
    get_log_transformer,
)


# Configuración del pipeline:


# Config para las nuevas columnas de feature engineering
# Permite configurar cuáles queremos agregar y cuáles no.
class FeatureCreatorParams(TypedDict):
    add_amenities_score: bool
    add_room_density: bool
    add_bath_bed_ratio: bool
    add_uncovered_pct: bool


# Config para clippear columnas
class OutlierClipperParams(TypedDict):
    cols_to_clip: List[str]  # columnas a clipear
    upper_pct: float  # percentil superior a clipear


class TargetParams(TypedDict):
    target_clipper_params: OutlierClipperParams
    log_transform: bool


class PipelineConfig(TypedDict):
    # Imputación:
    cols_to_drop: List[str]  # columnas que se van a eliminar
    mode_imputation_cols: List[str]  # columnas que se van a imputar con la moda
    median_imputer_cols: List[str]  # columnas que se van a imputar con la mediana
    boolean_imputer_cols: List[str]  # columnas booleanas (imputar con False)

    # Encoding
    target_encode_cols: List[str]  # columnas para target encoding
    ordinal_cols: List[str]  # columnas para ordinal encoding
    one_hot_cols: List[str]  # columnas para one hot encoding

    # Transformaciones
    feature_creator_params: FeatureCreatorParams
    outlier_clipper_params: OutlierClipperParams
    log_cols: List[str]  # log transform
    std_cols: List[str]  # estandarización
    target_params: TargetParams


def build_feature_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Construye el pipeline de preprocesamiento para las features (X).
    """

    # ColumnTransformer permite aplicar los pipelines solo a determinadas columnas
    # Acá vamos a definir las transformaciones que sólo aplican a algunas columnas
    # Obs: todas se aplican en paralelo. Idealmente, cada transformer debería tener columnas completamente diferentes
    imputer = ColumnTransformer(
        transformers=[
            (
                "mode",
                SimpleImputer(strategy="most_frequent"),
                config["mode_imputation_cols"],
            ),
            ("median", SimpleImputer(strategy="median"), config["median_imputer_cols"]),
            (
                "boolean",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value=False),
                        ),
                        (
                            "to_int",
                            FunctionTransformer(
                                lambda X: X.astype(int), feature_names_out="one-to-one"
                            ),
                        ),
                    ]
                ),
                config["boolean_imputer_cols"],
            ),
        ],
        remainder="passthrough",  # importante: si nos olvidamos de pasarle una columna, no la elimina
        verbose_feature_names_out=False,
    )

    encoder = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config["one_hot_cols"],
            ),
            (
                "target",
                TargetEncoder(target_type="continuous", smooth="auto", random_state=42),
                config["target_encode_cols"],
            ),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                config.get("ordinal_cols", []),
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    column_scaler = ColumnTransformer(
        transformers=[
            ("standard_scaler", StandardScaler(), config["std_cols"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    column_log_transformer = ColumnTransformer(
        transformers=[
            (
                "log_transformer",
                get_log_transformer(),
                config["log_cols"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # Pipeline completo
    feature_pipeline = Pipeline(
        steps=[
            ("feature_creator", FeatureCreator(**config["feature_creator_params"])),
            ("column_dropper", ColumnDropper(columns=config["cols_to_drop"])),
            ("imputer", imputer),
            ("encoder", encoder),
            ("outlier_clipper", OutlierClipper(**config["outlier_clipper_params"])),
            ("log_transformer", column_log_transformer),
            ("scaler", column_scaler),
        ]
    )

    return feature_pipeline


def build_target_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Construye el pipeline de preprocesamiento para el target (y).
    Clipea outliers y aplica transformación logarítmica.
    Las predicciones finales son clippeadas en 0.
    Ambas transformaciones son opcionales y configurables en PipelineConfig
    """
    target_config = config["target_params"]

    # Clipeamos predicciones negativas (a veces XGBoost puede tirar negativo)
    zero_clipper = (
        "zero_clipper",
        FunctionTransformer(
            func=lambda x: x,
            inverse_func=lambda x: np.maximum(0, x),
            feature_names_out="one-to-one",
        ),
    )

    steps: List[Tuple[str, Any]] = [
        zero_clipper,
        ("clipper", OutlierClipper(**target_config["target_clipper_params"])),
    ]

    if target_config["log_transform"]:
        steps.append(
            (
                "log_transformer",
                get_log_transformer(),
            )
        )

    target_pipeline = Pipeline(steps=steps)
    return target_pipeline


def build_base_pipeline(numeric_cols, categorical_cols):
    """
    Pipeline para los modelos base (hace lo mínimo para que puedan funcionar
    """

    # Numéricas: Imputar mediana
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categóricas: Imputar moda + Ordinal
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough",
    )
