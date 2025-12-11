"""
Configuración para el Multi-Layer Perceptron (MLP) Regressor.

La configuración fue obtenida mediante RandomizedSearchCV.
"""

from sklearn.neural_network import MLPRegressor
from src.pipeline import build_feature_pipeline, build_target_pipeline
from src.config import LINEAR_DEEP_LEARNING_CONFIG_NORMAL
from src.model_evaluation import _build_full_regressor

BEST_MLP_CONFIG = {
    "activation": "relu",
    "alpha": 0.000198,
    "batch_size": 128,
    "early_stopping": True,
    "hidden_layer_sizes": (64, 32),
    "learning_rate": "adaptive",
    "learning_rate_init": 0.002046,
    "max_iter": 120,
    "random_state": 42,
    "solver": "adam",
    "validation_fraction": 0.1,
}


def get_best_mlp():
    """
    Retorna el modelo MLP Regressor con la mejor configuración encontrada.
    Incluye el pipeline de preprocesamiento para modelos lineales/deep learning.

    Returns:
        TransformedTargetRegressor con el modelo MLP completo.
    """
    feature_pipeline = build_feature_pipeline(LINEAR_DEEP_LEARNING_CONFIG_NORMAL)
    target_pipeline = build_target_pipeline(LINEAR_DEEP_LEARNING_CONFIG_NORMAL)

    model = MLPRegressor(**BEST_MLP_CONFIG)

    return _build_full_regressor(model, feature_pipeline, target_pipeline)
