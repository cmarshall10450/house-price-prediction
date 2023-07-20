"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    return Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "MSSubClass_encoder",
                            OneHotEncoder(categories="auto", sparse=False),
                            ["MSSubClass_cluster"],
                        ),
                        (
                            "robust_scaler",
                            RobustScaler(quantile_range=(25.0, 75.0)),
                            ["OverallQual", "GrLivArea", "GarageCars",
                             "GarageArea", "TotalBsmtSF", "FullBath",
                             "YearBuilt", "YearRemodAdd",
                             "LotFrontage", "MSSubClass"],
                        ),
                    ]
                ),
            ),
        ]
    )
