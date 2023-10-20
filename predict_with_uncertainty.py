import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from typing import List
from dataclasses import dataclass

@dataclass
class PredictionDict:
    pred: np.ndarray = np.array([])
    uncertainty: np.ndarray = np.array([])
    pred_virt: np.ndarray = np.array([])
    lcb: np.ndarray = np.array([])
    ucb: np.ndarray = np.array([])

# Функция virtual_ensemble_iterations
def virtual_ensemble_iterations(
    model: GradientBoostingRegressor=None, k: int = 20
) -> List[int]:
    n_estimators = model.n_estimators  # получаем количество деревьев
    # вычисляем индексы моделей ансамбля
    iterations = [n_estimators // 2 - 1 + i for i in range(0, n_estimators // 2, k)]
    return iterations

# Функция virtual_ensemble_predict
def virtual_ensemble_predict(model, X, k):
    # staged_predictions = list(model.staged_predict(X))
    iterations = virtual_ensemble_iterations(model, k)
    # ensemble_predictions = [staged_predictions[i] for i in iterations]
    stage_preds = np.array(list (model.staged_predict(X)))[iterations]
    stage_preds = np.reshape(stage_preds, (len(iterations), len(X))).transpose()
    return stage_preds

# Функция predict_with_uncertainty
def predict_with_uncertainty(model, X, k):
    pred = virtual_ensemble_predict(model, X, k)

    uncertainty = np.square(np.std(pred, axis=1))
    pred_virt = np.mean(pred, axis=1)

    lcb = pred_virt - 3 * np.sqrt(uncertainty)
    ucb = pred_virt + 3 * np.sqrt(uncertainty)
    return PredictionDict(pred, uncertainty, pred_virt, lcb, ucb)

