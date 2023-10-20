from typing import Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold
import numpy as np

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE"""
    denominator = np.abs(y_true)
    return np.mean(np.where(denominator > 0, np.abs(y_true - y_pred) / denominator, 0))

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate sMAPE"""
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.mean(np.where(denominator > 0, 2 * np.abs(y_pred - y_true) / denominator, 0))

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate WAPE"""
    return np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true))

def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Bias"""
    return np.sum(y_pred - y_true) / np.sum(np.abs(y_true))

class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test
    gap : int, default=0
        Number of groups between train and test sets
    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                    'b', 'b', 'b', 'b', 'b',\
                    'c', 'c', 'c', 'c',\
                    'd', 'd', 'd',
                    'e', 'e', 'e'])
    >>> splitter = GroupTimeSeriesSplit(n_splits=3, max_train_size=2, gap=1)
    >>> for i, (train_idx, test_idx) in enumerate(
    ...     splitter.split(groups, groups=groups)):
    ...     print(f"Split: {i + 1}")
    ...     print(f"Train idx: {train_idx}, test idx: {test_idx}")
    ...     print(f"Train groups: {groups[train_idx]},
                    test groups: {groups[test_idx]}\n")
    Split: 1
    Train idx: [0 1 2 3 4 5], test idx: [11 12 13 14]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a'], test groups: ['c' 'c' 'c' 'c']

    Split: 2
    Train idx: [ 0  1  2  3  4  5  6  7  8  9 10], test idx: [15 16 17]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b'],
    test groups: ['d' 'd' 'd']

    Split: 3
    Train idx: [ 6  7  8  9 10 11 12 13 14], test idx: [18 19 20]
    Train groups: ['b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c'],
    test groups: ['e' 'e' 'e']
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        ind_groups = np.arange(len(groups))
        unique_groups = np.unique(groups)
        num_groups = len(unique_groups)
        if self.test_size is None:
            test_size = num_groups // (self.n_splits + 1) # размер тестовой выборки = кол-во_уник_товаров // (кол-во_разбиений + 1)
        else:
            test_size = self.test_size
        test_ind = range(num_groups - self.n_splits * test_size, num_groups, test_size)
        for test_ind_start in test_ind:
            train_ind_end = test_ind_start - self.gap
            if self.max_train_size is None:
                train_ind_start = 0
            else:
                train_ind_start = max(train_ind_end - self.max_train_size, 0)
            test_ind_end = test_ind_start + test_size
            train = ind_groups[np.isin(groups, unique_groups[train_ind_start:train_ind_end])]
            test = ind_groups[np.isin(groups, unique_groups[test_ind_start:test_ind_end])]
            yield train, test

def best_model() -> Any:
    # YOU CODE IS HERE:
    # type your own sklearn model
    # with custom parameters
    model = GradientBoostingRegressor(random_state=42)
    # ...
    return model



# import pandas as pd

# data = {
#     'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'],
#     'product_name': ['A', 'A', 'A', 'B', 'B', 'B'],
#     'sales': [10, 12, 14, 5, 6, 7]
# }

# df = pd.DataFrame(data)
# df['date'] = pd.to_datetime(df['date'])

# Создадим случайные данные для примера
# np.random.seed(42)
# n_samples = 20
# products = ["A", "B", "C", "D", "E", "F"]
# data = {
#     "date": pd.date_range(start="2023-01-01", periods=n_samples, freq="D"),
#     "product_name": np.random.choice(products, size=n_samples),
#     "sales": np.random.randint(1, 20, size=n_samples),
# }

# df = pd.DataFrame(data)
# print('All data', df)


# print('df[product_name]',df['product_name'])

# from sklearn.model_selection import GroupTimeSeriesSplit

# cv = GroupTimeSeriesSplit(n_splits=3, max_train_size=None, test_size=None, gap=0)

# for train_idx, test_idx in cv.split(df, groups=df['product_name']):
#     train_data = df.iloc[train_idx]
#     test_data = df.iloc[test_idx]

#     print("Train Data:")
#     print(train_data)
#     print("Test Data:")
#     print(test_data)
#     print("------------------------------")



# print('ebaniy_range')
# ebaniy_range = range(3, 20, 8)
# for i in ebaniy_range:
#     print(i)
