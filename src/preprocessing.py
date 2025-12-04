import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

from src.utils import COLS


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Elimina las columnas especificadas de un DataFrame.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns, errors="ignore")


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Crea la feature 'SDescubiertaM2' como la diferencia entre 'STotalM2' y 'SConstrM2'.
    """

    def __init__(
        self,
        total_col,
        constr_col,
        new_col_name,
    ):
        self.total = total_col
        self.constr = constr_col
        self.new = new_col_name

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.total in X_copy.columns and self.constr in X_copy.columns:
            # agregar columna descubierta como diferencia entre total y constr
            X_copy[self.new] = X_copy[self.total] - X_copy[self.constr]
            X_copy[self.new] = X_copy[self.new].clip(lower=0)  # eliminar negativos
        return X_copy


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Recorta outliers basados en percentiles aprendidos del set de entrenamiento.
    Como en todos los casos los outliers son muy grandes, cortamos solo el percentil más alto
    """

    def __init__(self, cols_to_clip: List[str], upper_pct: float):
        self.cols_to_clip = cols_to_clip
        self.upper_pct = upper_pct
        self.limits_ = {}

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
                return X # No podemos transformar sin nombres
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
        for col in self.cols_to_impute:
            if col in X_copy.columns:
                # creamos flag indicando que la columna fue imputada
                flag_col_name = f"{col}_is_missing"
                X_copy[flag_col_name] = X_copy[col].isnull()

                # imputamos la mediana.
                median_val = self.medians_.get(col)
                if median_val is not None:
                    X_copy[col] = X_copy[col].fillna(median_val)
        return X_copy
