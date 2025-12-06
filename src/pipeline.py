import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
    OrdinalEncoder,
)
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


class TargetParams(TypedDict):
    target_clipper_params: OutlierClipperParams
    log_transform: bool


class PipelineConfig(TypedDict):
    cols_to_drop: List[str]  # columnas que se van a eliminar
    feature_creator_params: FeatureCreatorParams
    median_imputer_cols: List[str]  # columnas que se van a imputar con la mediana
    outlier_clipper_params: OutlierClipperParams
    mode_imputation_cols: List[str]  # columnas que se van a imputar con la moda
    log_std_cols: List[str]  # log transform y estandarización
    std_only_cols: List[str]  # Solo estandarización (sin log) - para coordenadas
    one_hot_cols: List[str]  # columnas para one hot encoding
    boolean_imputer_cols: List[str]  # columnas booleanas (imputar con False)
    target_encode_cols: List[str]  # columnas para target encoding
    ordinal_cols: List[str]  # columnas para ordinal encoding
    target_params: TargetParams


def build_feature_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Construye el pipeline de preprocesamiento para las features (X).
    """
    # Sub-pipeline para numéricas discretas (ej: Dormitorios, Banos)
    # Solo se imputan con la moda (el valor más frecuente)
    mode_imputator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # Sub-pipeline para numéricas continuas (ej: Antiguedad, Superficie)
    # Transformación log y estandarización (la imputacion se hace antes)
    std_log_transformer = Pipeline(
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
    std_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    # Sub-pipeline para categóricas con Alta Cardinalidad (Barrios, Ciudades)
    target_encoding_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "target_encoder",
                TargetEncoder(target_type="continuous", smooth="auto", random_state=42),
            ),
        ]
    )

    # Sub-pipeline para categóricas que necesitan One-Hot Encoding
    one_hot_encoder = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Sub-pipeline para categóricas ordinales
    ordinal_encoder = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    # Sub-pipeline para amenities booleanas (imputar con false)
    boolean_transformer = SimpleImputer(strategy="constant", fill_value=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "mode_imputator",
                mode_imputator,
                config["mode_imputation_cols"],
            ),
            (
                "std_log_transformer",
                std_log_transformer,
                config["log_std_cols"],
            ),
            (
                "standard_scale",
                std_transformer,
                config["std_only_cols"],
            ),
            ("one_hot_encoder", one_hot_encoder, config["one_hot_cols"]),
            (
                "target_encoding",
                target_encoding_transformer,
                config.get("target_encode_cols", []),
            ),
            (
                "ordinal_encoder",
                ordinal_encoder,
                config.get("ordinal_cols", []),
            ),
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
    steps = [
        (
            "clipper",
            OutlierClipper(
                cols_to_clip=config["target_params"]["target_clipper_params"][
                    "cols_to_clip"
                ],
                upper_pct=config["target_params"]["target_clipper_params"]["upper_pct"],
            ),
        )
    ]

    if config["target_params"]["log_transform"]:
        steps.append(
            (
                "log_transformer",
                FunctionTransformer(
                    np.log1p,
                    inverse_func=np.expm1,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
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
        ]
    )
