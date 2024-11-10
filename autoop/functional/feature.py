from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from pandas.api.types import is_object_dtype, is_numeric_dtype


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    This function uses pandas.api.types module to check whether or not a
    column is of a certain type (numerical or categorical).
    It creates a list with features and returns it.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []

    df = dataset.read()

    for column in df.columns:
        if is_numeric_dtype(df[column]):
            feature_type = "numerical"
        elif is_object_dtype(df[column]):
            feature_type = "categorical"
        else:
            print(
                f"Column {column} is of neither " +
                "`numerical` nor `categorical` type"
            )

        # Creating an instance of Feature
        feature = Feature(column, feature_type)
        features.append(feature)

    return features
