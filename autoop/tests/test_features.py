import unittest
from sklearn.datasets import load_iris, fetch_openml
import pandas as pd

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


class TestFeatures(unittest.TestCase):
    """
    Unit tests for the feature detection functionality
    in the feature extraction module.
    """
    def setUp(self) -> None:
        """
        Set up the test environment before each test.

        Currently does nothing but is present for future setup,
        in case additional initialization is needed for specific tests.
        """
        pass

    def test_detect_features_continuous(self) -> None:
        """
        Test detecting features in a continuous (numerical) dataset.

        This test verifies that the `detect_feature_types`
        function correctly identifies all features in the Iris
        dataset as numerical (continuous) features. The test checks
        if the function returns the correct number of features and
        if each feature is correctly identified as 'numerical'.
        """
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "numerical")
        
    def test_detect_features_with_categories(self) -> None:
        """
        Test detecting features in a dataset with both
        numerical and categorical features.

        This test verifies that the `detect_feature_types`
        function correctly identifies numerical and categorical
        features in the "adult" dataset from OpenML. It checks if
        the function returns the correct number of features and
        if each feature is classified correctly into 'numerical' or
        'categorical' based on its characteristics.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(
            lambda x: x.name in numerical_columns, features
            ):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(
            lambda x: x.name in categorical_columns, features
            ):
            self.assertEqual(detected_feature.type, "categorical")
