from pandas import DataFrame
from typing import Tuple


def process_splits(
    train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Perform additional processing on the split datasets.

    :param train_df: The training dataset produced by the data splitting procedure.
    :param validation_df: The validation dataset produced by the data splitting procedure.
    :param test_df: The test dataset produced by the data splitting procedure.
    :return: A tuple containing, in order: the processed training dataset, the processed
             validation dataset, and the processed test dataset.
    """

    def process(df: DataFrame):
        cols = ["OverallQual", "GrLivArea", "GarageCars",
                "GarageArea", "TotalBsmtSF", "FullBath",
                "YearBuilt", "YearRemodAdd",
                "LotFrontage", "MSSubClass"]

        cleaned = df[["Id"]+cols+["SalePrice"]].dropna()

        # define clusters
        MSSubClass_clusters = {
            "min": [30, 45, 180], "max": [60, 120], "mean": []}
        # create new columns
        dic_flat = {v: k for k, lst in MSSubClass_clusters.items()
                    for v in lst}
        for k, v in MSSubClass_clusters.items():
            if len(v) == 0:
                residual_class = k
        cleaned["MSSubClass_cluster"] = cleaned["MSSubClass"].apply(lambda x: dic_flat[x] if x in
                                                                    dic_flat.keys() else residual_class)
        cleaned["LotFrontage"] = cleaned["LotFrontage"].fillna(
            cleaned["LotFrontage"].mean())

        return cleaned

    return process(train_df), process(validation_df), process(test_df)
