"""

Este módulo contiene transformadores personalizados para preprocesamiento

Transformadores principales:
- ColumnDropper: Elimina columnas específicas del DataFrame
- FeatureCreator: Crea features derivadas (amenities_score, ratios, densidades, etc.)
- OutlierClipper: Clipea outliers usando percentiles
- get_log_transformer(): Factory function para transformación logarítmica

Estos transformadores son compatibles con sklearn.pipeline.Pipeline y se usan
en las funciones build_feature_pipeline() y build_target_pipeline() del módulo
src.pipeline.
"""

import pandas as pd
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from typing import List
from src.utils import COLS


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Elimina columnas del DataFrame
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns, errors="ignore")


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Crea features nuevas a partir de las existentes según lo experimentado en 04-feature-engineering.ipynb
    """

    def __init__(
        self,
        add_amenities_score: bool = True,
        add_room_density: bool = True,
        add_bath_bed_ratio: bool = True,
        add_uncovered_pct: bool = True,
    ):
        self.add_amenities_score = add_amenities_score
        self.add_room_density = add_room_density
        self.add_bath_bed_ratio = add_bath_bed_ratio
        self.add_uncovered_pct = add_uncovered_pct

        self.amenities_list = [
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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Silenciamos el FutureWarning de pandas sobre downcasting en los fillna
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

            # 1. Porcentaje de superficie descubierta
            if self.add_uncovered_pct:
                # Creamos la col descubierta primero
                col_descubierta = COLS.SUP_DESCUBIERTA
                X_df[col_descubierta] = (
                    X_df[COLS.SUP_TOTAL] - X_df[COLS.SUP_CONSTR]
                ).clip(lower=0)

                # Porcentaje de superficie descubierta:
                total_safe = (
                    X_df[COLS.SUP_TOTAL]
                    .replace(0, pd.NA)
                    .fillna(1)
                    .infer_objects(copy=False)
                )
                X_df[COLS.SUP_DESCUBIERTA_PCT] = X_df[col_descubierta] / total_safe

            # 2. Densidad (M2 por Ambiente)
            if self.add_room_density and COLS.AMBIENTES in X_df.columns:
                amb_safe = (
                    X_df[COLS.AMBIENTES]
                    .replace(0, pd.NA)
                    .fillna(1)
                    .infer_objects(copy=False)
                )
                X_df[COLS.M2_POR_AMBIENTE] = X_df[COLS.SUP_CONSTR] / amb_safe

            # 3. Baños por Dormitorio
            if (
                self.add_bath_bed_ratio
                and COLS.BANOS in X_df.columns
                and COLS.DORMITORIOS in X_df.columns
            ):
                dorm_safe = (
                    X_df[COLS.DORMITORIOS]
                    .replace(0, pd.NA)
                    .fillna(1)
                    .infer_objects(copy=False)
                )
                X_df[COLS.BANOS_POR_DORMITORIO] = X_df[COLS.BANOS] / dorm_safe

            # 4. Amenities Score
            if self.add_amenities_score:
                valid_amenities = [c for c in self.amenities_list if c in X_df.columns]
                if valid_amenities:
                    # Asegurar que sean numéricos (True=1, False=0), tratando errores como NaN
                    amenities_numeric = X_df[valid_amenities].apply(
                        pd.to_numeric, errors="coerce"
                    )
                    X_df[COLS.AMENITIES_SCORE] = amenities_numeric.sum(
                        axis=1, min_count=0
                    ).astype(float)

        # Safety Check: features numéricas deben ser float
        if COLS.AMENITIES_SCORE in X_df.columns:
            X_df[COLS.AMENITIES_SCORE] = X_df[COLS.AMENITIES_SCORE].astype(float)

        return X_df


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Recorta outliers basados en percentiles aprendidos del set de entrenamiento.
    Como en todos los casos los outliers son muy grandes, cortamos solo el percentil más alto
    """

    def __init__(self, cols_to_clip: List[str], upper_pct: float):
        self.cols_to_clip = cols_to_clip
        self.upper_pct = upper_pct
        self.limits_ = {}  # limites aprendidos en train a partir de los cuales clipear

    def fit(self, X, y=None):
        # Si es array (viene de TransformedTargetRegressor), lo convertimos a DF
        if not isinstance(X, pd.DataFrame):
            # Asumimos que si es array y hay 1 columna en cols_to_clip, es esa.
            if len(self.cols_to_clip) == 1:
                X = pd.DataFrame(X, columns=self.cols_to_clip)
            else:
                # Si no podemos inferir nombres, no hacemos nada en fit
                return self

        # aprender el límite de los percentiles (el valor a partir del cual cortar)
        for col in self.cols_to_clip:
            if col in X.columns:
                upper_limit = X[col].quantile(self.upper_pct)
                self.limits_[col] = upper_limit
        return self

    def transform(self, X) -> pd.DataFrame:
        # Manejo de array a DF
        is_array = not isinstance(X, pd.DataFrame)
        if is_array:
            if len(self.cols_to_clip) == 1:
                X_df = pd.DataFrame(X, columns=self.cols_to_clip)
            else:
                return X  # No podemos transformar sin nombres
        else:
            X_df = X.copy()

        # clipeamos los datos del percentil más alto al valor del límite
        for col in self.cols_to_clip:
            if col in self.limits_ and col in X_df.columns:
                limit = self.limits_[col]
                X_df[col] = X_df[col].clip(upper=limit)

        # Si entró como array, devolvemos array (para que sklearn no se queje)
        if is_array:
            return X_df.to_numpy()
        return X_df

    def inverse_transform(self, X, y=None):
        return X


class MedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores faltantes con la mediana y crea una columna flag para indicar la imputación.
    """

    def __init__(self, cols_to_impute: List[str]):
        self.cols_to_impute = cols_to_impute
        self.medians_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # nos guardamos la media
        for col in self.cols_to_impute:
            if col in X.columns:
                median = X[col].median()
                self.medians_[col] = median
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        # Silenciamos el FutureWarning de pandas sobre downcasting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

            for col in self.cols_to_impute:
                if col in X_copy.columns:
                    # creamos flag indicando que la columna fue imputada
                    flag_col_name = f"{col}_is_missing"
                    X_copy[flag_col_name] = X_copy[col].isnull()

                    # imputamos la mediana.
                    median_val = self.medians_.get(col)
                    if median_val is not None:
                        X_copy[col] = (
                            X_copy[col].fillna(median_val).infer_objects(copy=False)
                        )
        return X_copy


def get_log_transformer() -> FunctionTransformer:
    """Retorna un FunctionTransformer log1p/expm1"""
    return FunctionTransformer(
        np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
        validate=False,
        feature_names_out="one-to-one",
    )
