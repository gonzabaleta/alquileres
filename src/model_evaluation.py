from typing import Type
import pandas as pd
import numpy as np
from contextlib import contextmanager
import joblib
from tqdm.auto import tqdm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)

from sklearn.model_selection import (
    cross_validate,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor


def _build_full_regressor(model, feature_pipeline, target_pipeline):
    """Helper: construye el regressor completo con pipelines."""
    full_model = Pipeline([("features", feature_pipeline), ("model", model)])
    return TransformedTargetRegressor(
        regressor=full_model, transformer=target_pipeline, check_inverse=False
    )


def _extract_cv_metrics(cv_results: dict) -> dict:
    """Helper: extrae métricas de resultados de cross_validate."""
    # Test metrics
    test_mae = -np.mean(cv_results["test_mae"])
    test_rmse = -np.mean(cv_results["test_rmse"])
    test_mse = -np.mean(cv_results["test_mse"])
    test_r2 = np.mean(cv_results["test_r2"])
    test_mape = -np.mean(cv_results["test_mape"])
    test_medae = -np.mean(cv_results["test_medae"])

    # Train metrics (si están disponibles)
    if "train_mae" in cv_results:
        train_mae = -np.mean(cv_results["train_mae"])
        train_rmse = -np.mean(cv_results["train_rmse"])
        train_mse = -np.mean(cv_results["train_mse"])
        train_r2 = np.mean(cv_results["train_r2"])
        train_mape = -np.mean(cv_results["train_mape"])
        train_medae = -np.mean(cv_results["train_medae"])

        overfit_gap = ((test_mae / train_mae) - 1) * 100 if train_mae > 0 else 0

        return {
            # Test metrics
            "mae_mean": test_mae,
            "mae_std": np.std(cv_results["test_mae"]),
            "rmse_mean": test_rmse,
            "rmse_std": np.std(cv_results["test_rmse"]),
            "mse_mean": test_mse,
            "mse_std": np.std(cv_results["test_mse"]),
            "r2_mean": test_r2,
            "r2_std": np.std(cv_results["test_r2"]),
            "mape_mean": test_mape,
            "mape_std": np.std(cv_results["test_mape"]),
            "medae_mean": test_medae,
            "medae_std": np.std(cv_results["test_medae"]),
            # Train metrics
            "train_mae_mean": train_mae,
            "train_mae_std": np.std(cv_results["train_mae"]),
            "train_rmse_mean": train_rmse,
            "train_rmse_std": np.std(cv_results["train_rmse"]),
            "train_mse_mean": train_mse,
            "train_mse_std": np.std(cv_results["train_mse"]),
            "train_r2_mean": train_r2,
            "train_r2_std": np.std(cv_results["train_r2"]),
            "train_mape_mean": train_mape,
            "train_mape_std": np.std(cv_results["train_mape"]),
            "train_medae_mean": train_medae,
            "train_medae_std": np.std(cv_results["train_medae"]),
            "overfit_gap_%": overfit_gap,
        }
    else:
        return {
            "mae_mean": test_mae,
            "mae_std": np.std(cv_results["test_mae"]),
            "rmse_mean": test_rmse,
            "rmse_std": np.std(cv_results["test_rmse"]),
            "mse_mean": test_mse,
            "mse_std": np.std(cv_results["test_mse"]),
            "r2_mean": test_r2,
            "r2_std": np.std(cv_results["test_r2"]),
            "mape_mean": test_mape,
            "mape_std": np.std(cv_results["test_mape"]),
            "medae_mean": test_medae,
            "medae_std": np.std(cv_results["test_medae"]),
        }


def evaluate_models_cv(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_pipeline=None,
    target_pipeline=None,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evalúa múltiples modelos usando K-Fold cross-validation.

    Args:
        models: Dict {nombre: modelo_sklearn}
        X, y: Features y target sin transformar
        feature_pipeline: Pipeline de transformación de features (opcional)
        target_pipeline: Pipeline de transformación de target (opcional)
        n_splits: Folds para CV (default: 5)
        random_state: Seed (default: 42)
        verbose: Imprimir progreso (default: True)

    Returns:
        DataFrame con métricas: mae, rmse, mse, r2, mape, medae (con _mean y _std)
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
        "mape": "neg_mean_absolute_percentage_error",
        "medae": "neg_median_absolute_error",
    }
    results = {}

    if verbose:
        print(f"\nEjecutando {n_splits}-Fold Cross-Validation...\n")

    for model_name, model in models.items():
        if verbose:
            print(f"Entrenando {model_name}...")

        # Si no hay pipelines, usar el modelo directamente
        if feature_pipeline is None and target_pipeline is None:
            final_model = model
        else:
            # Si hay pipelines, envolver con _build_full_regressor
            final_model = _build_full_regressor(
                model, feature_pipeline, target_pipeline
            )

        cv_results = cross_validate(
            final_model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
        )
        results[model_name] = _extract_cv_metrics(cv_results)

        if verbose:
            print(
                f"  MAE:   {results[model_name]['mae_mean']:,.0f} ± {results[model_name]['mae_std']:,.0f}"
            )
            print(
                f"  RMSE:  {results[model_name]['rmse_mean']:,.0f} ± {results[model_name]['rmse_std']:,.0f}"
            )
            print(
                f"  R²:    {results[model_name]['r2_mean']:.4f} ± {results[model_name]['r2_std']:.4f}"
            )
            print(
                f"  MAPE:  {results[model_name]['mape_mean']:.2f}% ± {results[model_name]['mape_std']:.2f}%"
            )
            print(
                f"  MedAE: {results[model_name]['medae_mean']:,.0f} ± {results[model_name]['medae_std']:,.0f}\n"
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
    Ejecuta GridSearchCV o RandomizedSearchCV para regresores
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Prefijo para params
    prefixed_params = {f"regressor__model__{k}": v for k, v in param_grid.items()}

    # Crear modelo base y regressor completo
    base_model = model_class(random_state=random_state)
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
            f"\n{search_type}: {n_combinations} combinaciones × {n_splits} folds = {total_fits} fits"
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

    # tqdm para mostrar progreso
    if verbose:
        with tqdm_joblib(tqdm(desc="Optimizando", total=total_fits)):
            search.fit(X, y)
    else:
        search.fit(X, y)

    # Extraer resultados
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
        print(f"\nMejor MAE: {-search.best_score_:,.0f}")
        print(f"   Mejores params: {search.best_params_}")

    return results_df


def classifier_search_cv(
    model_class: Type,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_pipeline,
    n_splits: int = 5,
    n_iter: int = None,
    random_state: int = 42,
    scoring_metric: str = "f1",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta GridSearchCV o RandomizedSearchCV para clasificadores.

    Args:
        model_class: Clase del clasificador (ej: XGBClassifier)
        param_grid: Diccionario de hiperparámetros
        X, y: Features y target
        feature_pipeline: Pipeline de preprocesamiento
        n_splits: Folds para CV (default: 5)
        n_iter: Si None, hace GridSearchCV. Si número, hace RandomizedSearchCV
        random_state: Seed (default: 42)
        scoring_metric: Métrica principal para optimizar (default: 'f1')
        verbose: Imprimir progreso (default: True)

    Returns:
        DataFrame con resultados ordenados por scoring_metric
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Prefijo para params dentro del pipeline
    prefixed_params = {f"model__{k}": v for k, v in param_grid.items()}

    # Crear pipeline: features + clasificador
    base_model = model_class(random_state=random_state, eval_metric="logloss")
    pipeline = Pipeline([("features", feature_pipeline), ("model", base_model)])

    # Métricas a calcular
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }

    # Calcular total de fits
    if n_iter is None:
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        search_type = "GridSearchCV"
    else:
        n_combinations = n_iter
        search_type = "RandomizedSearchCV"

    total_fits = n_combinations * n_splits

    if verbose:
        print(
            f"\n{search_type}: {n_combinations} combinaciones × {n_splits} folds = {total_fits} fits"
        )
        print(f"   Modelo: {model_class.__name__}")
        print(f"   Optimizando: {scoring_metric}\n")

    # Crear searcher
    if n_iter is None:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=prefixed_params,
            cv=cv,
            scoring=scoring,
            refit=scoring_metric,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=prefixed_params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            refit=scoring_metric,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
            return_train_score=True,
        )

    # Ejecutar fit con progress bar
    if verbose:
        with tqdm_joblib(tqdm(desc="Optimizando", total=total_fits)):
            search.fit(X, y)
    else:
        search.fit(X, y)

    # Extraer resultados
    results = []
    for i in range(len(search.cv_results_["params"])):
        params = {
            k.replace("model__", ""): v
            for k, v in search.cv_results_["params"][i].items()
        }

        row = {
            **params,
            # Test scores
            "accuracy": search.cv_results_["mean_test_accuracy"][i],
            "precision": search.cv_results_["mean_test_precision"][i],
            "recall": search.cv_results_["mean_test_recall"][i],
            "f1": search.cv_results_["mean_test_f1"][i],
            "roc_auc": search.cv_results_["mean_test_roc_auc"][i],
            "avg_precision": search.cv_results_["mean_test_avg_precision"][i],
            # Std
            "f1_std": search.cv_results_["std_test_f1"][i],
            # Train scores (para detectar overfitting)
            "train_f1": search.cv_results_["mean_train_f1"][i],
            "train_accuracy": search.cv_results_["mean_train_accuracy"][i],
        }

        # Overfitting gap
        test_f1 = row["f1"]
        train_f1 = row["train_f1"]
        row["overfit_gap_%"] = ((train_f1 / test_f1) - 1) * 100 if test_f1 > 0 else 0

        results.append(row)

    results_df = pd.DataFrame(results).sort_values(scoring_metric, ascending=False)

    # Agregar config_id
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, "config_id", range(1, len(results_df) + 1))

    if verbose:
        best = results_df.iloc[0]
        print(f"\nMejor {scoring_metric}: {best[scoring_metric]:.4f}")
        print(f"   Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f}")
        print(
            f"   ROC-AUC: {best['roc_auc']:.4f} | Avg Precision: {best['avg_precision']:.4f}"
        )
        print(f"   Overfitting gap (F1): {best['overfit_gap_%']:.1f}%")

    return results_df


def evaluate_classifiers_cv(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_pipeline=None,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evalúa múltiples clasificadores usando Stratified K-Fold cross-validation.

    Args:
        models: Dict {nombre: clasificador_sklearn}
        X, y: Features y target sin transformar
        feature_pipeline: Pipeline de transformación de features (opcional)
        n_splits: Folds para CV (default: 5)
        random_state: Seed (default: 42)
        verbose: Imprimir progreso (default: True)

    Returns:
        DataFrame con métricas de clasificación
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }

    results = {}

    if verbose:
        print(f"\nEjecutando {n_splits}-Fold Stratified CV (Clasificación)...\n")

    for model_name, model in models.items():
        if verbose:
            print(f"Entrenando {model_name}...")

        # Si hay pipeline, envolver el modelo
        if feature_pipeline is None:
            final_model = model
        else:
            final_model = Pipeline([("features", feature_pipeline), ("model", model)])

        cv_results = cross_validate(
            final_model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
        )

        # Extraer métricas
        test_f1 = np.mean(cv_results["test_f1"])
        train_f1 = np.mean(cv_results["train_f1"])

        test_recall = np.mean(cv_results["test_recall"])
        train_recall = np.mean(cv_results["train_recall"])

        test_precision = np.mean(cv_results["test_precision"])
        train_precision = np.mean(cv_results["train_precision"])

        results[model_name] = {
            # Test scores
            "f1": test_f1,
            "f1_std": np.std(cv_results["test_f1"]),
            "recall": test_recall,
            "recall_std": np.std(cv_results["test_recall"]),
            "precision": test_precision,
            "precision_std": np.std(cv_results["test_precision"]),
            "accuracy": np.mean(cv_results["test_accuracy"]),
            "roc_auc": np.mean(cv_results["test_roc_auc"]),
            "avg_precision": np.mean(cv_results["test_avg_precision"]),
            # Train scores
            "train_f1": train_f1,
            "train_recall": train_recall,
            "train_precision": train_precision,
            "train_accuracy": np.mean(cv_results["train_accuracy"]),
            # Gaps (overfitting indicators)
            "f1_gap_%": ((train_f1 - test_f1) / test_f1) * 100 if test_f1 > 0 else 0,
            "recall_gap_%": (
                ((train_recall - test_recall) / test_recall) * 100
                if test_recall > 0
                else 0
            ),
            "precision_gap_%": (
                ((train_precision - test_precision) / test_precision) * 100
                if test_precision > 0
                else 0
            ),
        }

        if verbose:
            print(f"  F1:        {test_f1:.4f} ± {results[model_name]['f1_std']:.4f}")
            print(
                f"  Recall:    {test_recall:.4f} ± {results[model_name]['recall_std']:.4f}"
            )
            print(
                f"  Precision: {test_precision:.4f} ± {results[model_name]['precision_std']:.4f}"
            )
            print(f"  ROC-AUC:   {results[model_name]['roc_auc']:.4f}\n")

    results_df = pd.DataFrame(results).T
    return results_df.sort_values("f1", ascending=False)


def evaluate_models_test(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_pipeline=None,
    target_pipeline=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evalúa múltiples modelos usando un test set

    Similar a evaluate_models_cv, pero en lugar de cross-validation,
    entrena en train y evalúa en test.

    Args:
        models: Dict {nombre: modelo_sklearn}
        X_train, y_train: Features y target de entrenamiento sin transformar
        X_test, y_test: Features y target de test sin transformar
        feature_pipeline: Pipeline de transformación de features (opcional)
        target_pipeline: Pipeline de transformación de target (opcional)
        verbose: Imprimir progreso (default: True)

    Returns:
        DataFrame con métricas: mae, rmse, mse, r2, mape, medae
    """

    results = {}

    if verbose:
        print(f"Evaluando modelos en test set...\n")
        print(f"   Train: {len(X_train):,} observaciones")
        print(f"   Test:  {len(X_test):,} observaciones\n")

    for model_name, model in models.items():
        if verbose:
            print(f"Entrenando {model_name}...")

        # Si no hay pipelines, usar el modelo directamente
        if feature_pipeline is None and target_pipeline is None:
            final_model = model
        else:
            # Si hay pipelines, envolver con _build_full_regressor
            final_model = _build_full_regressor(
                model, feature_pipeline, target_pipeline
            )

        # Entrenar en train set
        final_model.fit(X_train, y_train)

        # Predecir en ambos sets
        y_train_pred = final_model.predict(X_train)
        y_test_pred = final_model.predict(X_test)

        # Calcular métricas en train
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        train_medae = median_absolute_error(y_train, y_train_pred)

        # Calcular métricas en test
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        test_medae = median_absolute_error(y_test, y_test_pred)

        # Calcular gap de overfitting
        overfit_gap = ((test_mae / train_mae) - 1) * 100 if train_mae > 0 else 0

        results[model_name] = {
            # Test metrics
            "mae": test_mae,
            "rmse": test_rmse,
            "mse": test_mse,
            "r2": test_r2,
            "mape": test_mape,
            "medae": test_medae,
            # Train metrics
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "train_mape": train_mape,
            "train_medae": train_medae,
            "overfit_gap_%": overfit_gap,
        }

        if verbose:
            print(f"  Test MAE:   {test_mae:,.0f}  |  Train MAE:   {train_mae:,.0f}")
            print(f"  Test RMSE:  {test_rmse:,.0f}  |  Train RMSE:  {train_rmse:,.0f}")
            print(f"  Test R²:    {test_r2:.4f}  |  Train R²:    {train_r2:.4f}")
            print(f"  Test MAPE:  {test_mape:.2f}%  |  Train MAPE:  {train_mape:.2f}%")
            print(
                f"  Test MedAE: {test_medae:,.0f}  |  Train MedAE: {train_medae:,.0f}"
            )

    results_df = pd.DataFrame(results).T
    return results_df.sort_values("mae")
