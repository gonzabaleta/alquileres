from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def get_dummy_model():
    return DummyRegressor(strategy="mean")


def get_base_lasso(preprocessor):
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("lasso", Lasso(alpha=10, random_state=42)),
        ]
    )


def get_base_tree(preprocessor):
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(max_depth=4, random_state=42)),
        ]
    )


def get_base_xgboost(preprocessor):
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=500,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
