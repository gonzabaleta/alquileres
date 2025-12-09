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

        # Si no hay pipelines, usar el modelo directamente
        if feature_pipeline is None and target_pipeline is None:
            final_model = model
        else:
            # Si hay pipelines, envolver con _build_full_regressor
            final_model = _build_full_regressor(
                model, feature_pipeline, target_pipeline
            )

        cv_results = cross_validate(
            final_model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
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


def classifier_search_cv(
    model_class: Type,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_pipeline,
    n_splits: int = 5,
    n_iter: int = None,
    random_state: int = 42,
    scoring_metric: str = 'f1',
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
    base_model = model_class(random_state=random_state, eval_metric='logloss')
    pipeline = Pipeline([
        ("features", feature_pipeline),
        ("model", base_model)
    ])

    # Métricas a calcular
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'avg_precision': 'average_precision'
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
        print(f"\n🔍 {search_type}: {n_combinations} combinaciones × {n_splits} folds = {total_fits} fits")
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
            'accuracy': search.cv_results_["mean_test_accuracy"][i],
            'precision': search.cv_results_["mean_test_precision"][i],
            'recall': search.cv_results_["mean_test_recall"][i],
            'f1': search.cv_results_["mean_test_f1"][i],
            'roc_auc': search.cv_results_["mean_test_roc_auc"][i],
            'avg_precision': search.cv_results_["mean_test_avg_precision"][i],
            # Std
            'f1_std': search.cv_results_["std_test_f1"][i],
            # Train scores (para detectar overfitting)
            'train_f1': search.cv_results_["mean_train_f1"][i],
            'train_accuracy': search.cv_results_["mean_train_accuracy"][i],
        }

        # Overfitting gap
        test_f1 = row['f1']
        train_f1 = row['train_f1']
        row['overfit_gap_%'] = ((train_f1 / test_f1) - 1) * 100 if test_f1 > 0 else 0

        results.append(row)

    results_df = pd.DataFrame(results).sort_values(scoring_metric, ascending=False)

    # Agregar config_id
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, 'config_id', range(1, len(results_df) + 1))

    if verbose:
        best = results_df.iloc[0]
        print(f"\n✅ Mejor {scoring_metric}: {best[scoring_metric]:.4f}")
        print(f"   Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f}")
        print(f"   ROC-AUC: {best['roc_auc']:.4f} | Avg Precision: {best['avg_precision']:.4f}")
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
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'avg_precision': 'average_precision'
    }
    
    results = {}

    if verbose:
        print(f"\n🔄 Ejecutando {n_splits}-Fold Stratified CV (Clasificación)...\n")

    for model_name, model in models.items():
        if verbose:
            print(f"Entrenando {model_name}...")

        # Si hay pipeline, envolver el modelo
        if feature_pipeline is None:
            final_model = model
        else:
            final_model = Pipeline([
                ("features", feature_pipeline),
                ("model", model)
            ])

        cv_results = cross_validate(
            final_model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
            return_train_score=True
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
            'f1': test_f1,
            'f1_std': np.std(cv_results["test_f1"]),
            'recall': test_recall,
            'recall_std': np.std(cv_results["test_recall"]),
            'precision': test_precision,
            'precision_std': np.std(cv_results["test_precision"]),
            'accuracy': np.mean(cv_results["test_accuracy"]),
            'roc_auc': np.mean(cv_results["test_roc_auc"]),
            'avg_precision': np.mean(cv_results["test_avg_precision"]),
            
            # Train scores
            'train_f1': train_f1,
            'train_recall': train_recall,
            'train_precision': train_precision,
            'train_accuracy': np.mean(cv_results["train_accuracy"]),
            
            # Gaps (overfitting indicators)
            'f1_gap_%': ((train_f1 - test_f1) / test_f1) * 100 if test_f1 > 0 else 0,
            'recall_gap_%': ((train_recall - test_recall) / test_recall) * 100 if test_recall > 0 else 0,
            'precision_gap_%': ((train_precision - test_precision) / test_precision) * 100 if test_precision > 0 else 0,
        }

        if verbose:
            print(f"  ✓ F1:        {test_f1:.4f} ± {results[model_name]['f1_std']:.4f}")
            print(f"  ✓ Recall:    {test_recall:.4f} ± {results[model_name]['recall_std']:.4f}")
            print(f"  ✓ Precision: {test_precision:.4f} ± {results[model_name]['precision_std']:.4f}")
            print(f"  ✓ ROC-AUC:   {results[model_name]['roc_auc']:.4f}\n")

    results_df = pd.DataFrame(results).T
    return results_df.sort_values("f1", ascending=False)
