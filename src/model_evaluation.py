from contextlib import contextmanager

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_validate,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from typing import Type

from tqdm.auto import tqdm


def _build_full_regressor(model, feature_pipeline, target_pipeline):
    """Helper: construye el regressor completo con pipelines."""
    full_model = Pipeline([("features", feature_pipeline), ("model", model)])
    return TransformedTargetRegressor(
        regressor=full_model, transformer=target_pipeline, check_inverse=False
    )


def _extract_cv_metrics(cv_results: dict) -> dict:
    """Helper: extrae métricas de resultados de cross_validate."""
    test_rmse = -np.mean(cv_results["test_rmse"])
    test_mae = -np.mean(cv_results["test_mae"])

    # Train metrics (si están disponibles)
    if "train_rmse" in cv_results:
        train_rmse = -np.mean(cv_results["train_rmse"])
        train_mae = -np.mean(cv_results["train_mae"])
        overfit_gap = ((test_mae / train_mae) - 1) * 100 if train_mae > 0 else 0
        return {
            "mae_mean": test_mae,
            "mae_std": np.std(cv_results["test_mae"]),
            "rmse_mean": test_rmse,
            "rmse_std": np.std(cv_results["test_rmse"]),
            "train_mae_mean": train_mae,
            "train_mae_std": np.std(cv_results["train_mae"]),
            "train_rmse_mean": train_rmse,
            "train_rmse_std": np.std(cv_results["train_rmse"]),
            "overfit_gap_%": overfit_gap,
        }
    else:
        return {
            "mae_mean": test_mae,
            "mae_std": np.std(cv_results["test_mae"]),
            "rmse_mean": test_rmse,
            "rmse_std": np.std(cv_results["test_rmse"]),
        }


def evaluate_models_cv(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_pipeline,
    target_pipeline,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evalúa múltiples modelos usando K-Fold cross-validation.

    Args:
        models: Dict {nombre: modelo_sklearn}
        X, y: Features y target sin transformar
        feature_pipeline, target_pipeline: Pipelines de transformación
        n_splits: Folds para CV (default: 5)
        random_state: Seed (default: 42)
        verbose: Imprimir progreso (default: True)

    Returns:
        DataFrame con métricas (rmse_mean, rmse_std, mae_mean, mae_std)
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    results = {}

    if verbose:
        print(f"\n🔄 Ejecutando {n_splits}-Fold Cross-Validation...\n")

    for model_name, model in models.items():
        if verbose:
            print(f"Entrenando {model_name}...")

        final_regressor = _build_full_regressor(
            model, feature_pipeline, target_pipeline
        )
        cv_results = cross_validate(
            final_regressor, X, y, cv=cv, scoring=scoring, n_jobs=-1,
            return_train_score=True
        )
        results[model_name] = _extract_cv_metrics(cv_results)

        if verbose:
            print(
                f"  ✓ MAE: {results[model_name]['mae_mean']:,.0f} ± {results[model_name]['mae_std']:,.0f}"
            )
            print(
                f"  ✓ RMSE: {results[model_name]['rmse_mean']:,.0f} ± {results[model_name]['rmse_std']:,.0f}\n"
            )

    results_df = pd.DataFrame(results).T
    return results_df.sort_values("mae_mean")


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager para que joblib reporte el progreso a tqdm."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# --- Tu función modificada ---
def grid_search_cv(
    model_class: Type,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_pipeline,
    target_pipeline,
    n_splits: int = 3,
    n_iter: int = None,  # Si None → GridSearch, si número → RandomizedSearch
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta GridSearchCV o RandomizedSearchCV con barra de progreso (tqdm).

    Args:
        n_iter: Si None, hace GridSearchCV exhaustivo.
                Si es un número, hace RandomizedSearchCV con n_iter combinaciones.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Prefijo para params
    prefixed_params = {f"regressor__model__{k}": v for k, v in param_grid.items()}

    # Crear modelo base y regressor completo
    base_model = model_class(random_state=random_state)
    # Asumo que esta función _build_full_regressor la tenés definida en otro lado
    final_regressor = _build_full_regressor(
        base_model, feature_pipeline, target_pipeline
    )

    # Calcular total de fits para la barra
    if n_iter is None:
        # GridSearchCV: todas las combinaciones
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        search_type = "GridSearchCV"
    else:
        # RandomizedSearchCV: n_iter combinaciones
        n_combinations = n_iter
        search_type = "RandomizedSearchCV"

    total_fits = n_combinations * n_splits

    if verbose:
        print(
            f"\n🔍 {search_type}: {n_combinations} combinaciones × {n_splits} folds = {total_fits} fits"
        )
        print(f"   Modelo: {model_class.__name__}\n")

    # Crear el searcher apropiado
    if n_iter is None:
        search = GridSearchCV(
            estimator=final_regressor,
            param_grid=prefixed_params,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            refit="mae",
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator=final_regressor,
            param_distributions=prefixed_params,
            n_iter=n_iter,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            refit="mae",
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
            return_train_score=True,
        )

    # Ejecutar fit con el context manager de tqdm
    if verbose:
        with tqdm_joblib(tqdm(desc="Optimizando", total=total_fits)):
            search.fit(X, y)
    else:
        search.fit(X, y)

    # Extraer resultados con AMBAS métricas (train y test)
    results = []
    for i in range(len(search.cv_results_["params"])):
        params = {
            k.replace("regressor__model__", ""): v
            for k, v in search.cv_results_["params"][i].items()
        }

        # Calcular gap (overfitting indicator)
        test_mae = -search.cv_results_["mean_test_mae"][i]
        train_mae = -search.cv_results_["mean_train_mae"][i]
        gap_pct = ((test_mae / train_mae) - 1) * 100 if train_mae > 0 else 0

        results.append(
            {
                **params,
                # Test scores
                "mae_mean": test_mae,
                "mae_std": search.cv_results_["std_test_mae"][i],
                "rmse_mean": -search.cv_results_["mean_test_rmse"][i],
                "rmse_std": search.cv_results_["std_test_rmse"][i],
                # Train scores
                "train_mae_mean": train_mae,
                "train_mae_std": search.cv_results_["std_train_mae"][i],
                "train_rmse_mean": -search.cv_results_["mean_train_rmse"][i],
                "train_rmse_std": search.cv_results_["std_train_rmse"][i],
                # Overfitting indicator
                "overfit_gap_%": gap_pct,
            }
        )

    results_df = pd.DataFrame(results).sort_values("mae_mean")

    if verbose:
        print(f"\n✅ Mejor MAE: {-search.best_score_:,.0f}")
        print(f"   Mejores params: {search.best_params_}")

        # Mostrar gap de overfitting del mejor modelo
        best_idx = results_df.index[0]
        best_gap = results_df.loc[best_idx, "overfit_gap_%"]
        print(f"   Overfitting gap: {best_gap:.1f}% ", end="")
        if best_gap < 15:
            print("✅ (excelente)")
        elif best_gap < 30:
            print("⚠️ (aceptable)")
        else:
            print("❌ (overfitting!)")

    return results_df
