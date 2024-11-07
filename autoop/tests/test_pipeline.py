from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.\
    multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.metric import MSEMetric


class TestPipeline(unittest.TestCase):
    """
    Unit tests for the Pipeline class.
    """
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(type="regression"),
            input_features=list(filter(
                lambda x: x.name != "age", self.features
                )),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MSEMetric()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self) -> None:
        """
        Test the initialization of the Pipeline.

        Verifies that the Pipeline instance is correctly
        initialized and is of the `Pipeline` class type.
        """
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self) -> None:
        """
        Test the preprocessing of features in the Pipeline.

        Verifies that the `Pipeline` correctly preprocesses
        the features by checking if the number of artifacts
        (processed features) is the same as the number of
        original features. This ensures that the feature
        preprocessing step is executed correctly.
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self) -> None:
        """
        Test the splitting of data into training and testing sets.

        Verifies that the data is correctly split into training
        and testing sets with the specified ratio of 80% training
        and 20% testing. The number of samples in
        each set should be consistent with the split ratio.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(
            self.pipeline._train_X[0].shape[0],
            int(0.8 * self.ds_size))
        self.assertEqual(
            self.pipeline._test_X[0].shape[0],
            self.ds_size - int(0.8 * self.ds_size)
            )

    def test_train(self) -> None:
        """
        Test the training of the model within the Pipeline.

        Verifies that the model is successfully trained
        after the preprocessing and data splitting steps.
        The model's parameters should be populated after training.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self) -> None:
        """
        Test the evaluation of the model using the specified metrics.

        Verifies that the model's predictions are generated and
        evaluated correctly using the specified metrics.
        The test checks that the evaluation results (predictions
        and metrics) are not `None`, and the correct number
        of metrics are generated.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate(self.pipeline._test_X, self.pipeline._test_y)
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        self.assertEqual(len(self.pipeline._metrics_results), 1)
