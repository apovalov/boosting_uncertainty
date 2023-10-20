import pandas as pd
from solution import bias
from solution import GroupTimeSeriesSplit
from solution import mape
from solution import smape
from solution import wape
from solution import best_model
# from sklearn.ensemble import GradientBoostingRegressor
# from catboost import CatBoostRegressor


def main():
    # Data loading
    df_path = "data/data_train_sql.csv"
    # df = pd.read_csv(df_path, parse_dates=["monday"])
    # df = pd.read_csv(df_path, parse_dates=["monday"], date_parser=pd.to_datetime)
    df = pd.read_csv(df_path, parse_dates=["monday"], date_format='%d/%m/%y')

    y = df.pop("y")

    # monday or product_name as a groups for validation?
    # df.drop(..., axis=1, inplace=True)
    groups = df.pop('product_name')

    X = df

    # Validation loop
    cv = GroupTimeSeriesSplit(
        n_splits=5,
        max_train_size=None,
        test_size=None,
        gap=0,
    )

    fold = 1

    for train_idx, test_idx in cv.split(X, y, groups):
        # Split train/test
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model = best_model()
        model.fit(X_train, y_train)

        # Predict and print metrics
        y_pred = model.predict(X_test)

        mape_score = mape(y_test, y_pred)
        smape_score = smape(y_test, y_pred)
        wape_score = wape(y_test, y_pred)
        bias_score = bias(y_test, y_pred)
        # mae_score = mean_absolute_error(y_test, y_pred)

        print(f"Fold {fold}:")
        print(f"MAPE: {mape_score}")
        print(f"sMAPE: {smape_score}")
        print(f"WAPE: {wape_score}")
        print(f"Bias: {bias_score}")
        # print(f"MAE: {mae_score}")
        print("------------------------------")

        fold += 1



if __name__ == "__main__":
    main()
