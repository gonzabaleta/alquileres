from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd
import numpy as np


class OutlierAwareRegressor(BaseEstimator, RegressorMixin):
    """
    Modelo híbrido que usa un clasificador para detectar outliers y
    redirige la predicción a dos regresores especializados (normal vs outlier).
    """

    def __init__(self, classifier, normal_regressor, outlier_regressor):
        self.classifier = classifier
        self.normal_regressor = normal_regressor
        self.outlier_regressor = outlier_regressor

    def fit(self, X, y):
        # Clonar modelos para no afectar instancias originales
        self.classifier_ = clone(self.classifier)
        self.normal_regressor_ = clone(self.normal_regressor)
        self.outlier_regressor_ = clone(self.outlier_regressor)

        # 1. Entrenar clasificador
        # Definimos etiquetas de los outliers
        threshold = np.quantile(y, 0.985)
        is_outlier = (y >= threshold).astype(int)

        self.classifier_.fit(X, is_outlier)

        # 2. Dividir datos
        mask_outlier = is_outlier == 1
        X_out = X[mask_outlier]
        y_out = y[mask_outlier]

        X_norm = X[~mask_outlier]
        y_norm = y[~mask_outlier]

        # 3. Entrenar expertos
        self.outlier_regressor_.fit(X_out, y_out)
        self.normal_regressor_.fit(X_norm, y_norm)

        # Guardar threshold para referencia
        self.threshold_ = threshold

        return self

    def predict(self, X):
        check_is_fitted(self)

        # 1. Predecir probabilidad de outlier
        # Tip: Usar probabilidad permite ajustar el threshold luego
        probs = self.classifier_.predict_proba(X)[:, 1]

        # Usar threshold 0.5 por defecto (o ajustar si encontraste uno mejor)
        pred_is_outlier = probs >= 0.5

        # 2. Inicializar array de resultados
        final_preds = np.zeros(X.shape[0])

        # 3. Router
        mask_out = pred_is_outlier
        mask_norm = ~pred_is_outlier

        if np.any(mask_out):
            final_preds[mask_out] = self.outlier_regressor_.predict(X[mask_out])

        if np.any(mask_norm):
            final_preds[mask_norm] = self.normal_regressor_.predict(X[mask_norm])

        return final_preds
