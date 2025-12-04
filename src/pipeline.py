import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from typing import List, TypedDict

from src.preprocessing import (
    ColumnDropper,
    FeatureCreator,
    MedianImputer,
    OutlierClipper,
)


# Configuación del pipeline:


# Config para la nueva columna de superficie descubierta
class FeatureCreatorParams(TypedDict):
    total_col: str
    constr_col: str
    new_col_name: str


# Config para clippear columnas
class OutlierClipperParams(TypedDict):
    cols_to_clip: List[str]
    upper_pct: float  # percentil a clipear


class PipelineConfig(TypedDict):
    cols_to_drop: List[str]  # columnas que se van a eliminar
    feature_creator_params: FeatureCreatorParams
    median_imputer_cols: List[str]  # columnas que se van a imputar con la mediana
    outlier_clipper_params: OutlierClipperParams
    discrete_numeric_cols: List[str]  # columnas que se van a imputar con la moda
    continuous_numeric_cols: List[str]  # log transform y estandarización
    standard_scale_cols: List[str]  # Solo estandarización (sin log) - para coordenadas
    ohe_cols: List[str]  # columnas para one hot encoding
    boolean_imputer_cols: List[str]  # columnas booleanas (imputar con False)
    target_clipper_params: OutlierClipperParams  # al precio lo clipeamos


def build_feature_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Construye el pipeline de preprocesamiento para las features (X).
    """
    # Sub-pipeline para numéricas discretas (ej: Dormitorios, Banos)
    # Solo se imputan con la moda (el valor más frecuente)
    discrete_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # Sub-pipeline para numéricas continuas (ej: Antiguedad, Superficie)
    # Transformación log y estandarización (la imputacion se hace antes)
    continuous_numeric_transformer = Pipeline(
        steps=[
            (
                "log_transformer",
                FunctionTransformer(
                    np.log1p, validate=False, feature_names_out="one-to-one"
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    # Sub-pipeline para numéricas que solo necesitan estandarización (ej: Latitud, Longitud)
    standard_scaler_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    # Sub-pipeline para categóricas que necesitan One-Hot Encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Sub-pipeline para amenities booleanas (imputar con false)
    boolean_transformer = SimpleImputer(strategy="constant", fill_value=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "discrete_numeric",
                discrete_numeric_transformer,
                config["discrete_numeric_cols"],
            ),
            (
                "continuous_numeric",
                continuous_numeric_transformer,
                config["continuous_numeric_cols"],
            ),
            (
                "standard_scale",
                standard_scaler_transformer,
                config["standard_scale_cols"],
            ),
            ("categorical", categorical_transformer, config["ohe_cols"]),
            ("boolean", boolean_transformer, config["boolean_imputer_cols"]),
        ],
        remainder="passthrough",
    )

    # Pipeline completo
    feature_pipeline = Pipeline(
        steps=[
            ("feature_creator", FeatureCreator(**config["feature_creator_params"])),
            ("column_dropper", ColumnDropper(columns=config["cols_to_drop"])),
            (
                "median_imputer",
                MedianImputer(cols_to_impute=config["median_imputer_cols"]),
            ),
            ("outlier_clipper", OutlierClipper(**config["outlier_clipper_params"])),
            ("preprocessor", preprocessor),
        ]
    )

    return feature_pipeline


def build_target_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Construye el pipeline de preprocesamiento para el target (y).
    Clipea outliers y aplica transformación logarítmica.
    """
    target_pipeline = Pipeline(
        steps=[
            ("clipper", OutlierClipper(**config["target_clipper_params"])),
            (
                "log_transformer",
                FunctionTransformer(
                    np.log1p, validate=False, feature_names_out="one-to-one"
                ),
            ),
        ]
    )
    return target_pipeline
