from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    The Pipeline of the autoop.
    """
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """
        A way to initialise a pipeline.
        Pipeline: a state machine that orchestrates the different stages.
        (i.e., preprocessing, splitting, training, evaluation).

        :param metrics: the metrics used to evaluate.
        :type metrics: List[Metric]
        :param dataset: the dataset to be used in training.
        :type dataset: Dataset
        :param model: the model used.
        :type model: Model
        :param input_features: the features of the columns of the input data.
        :type input_features: List[Feature]
        :param target_feature: the target feature.
        :type target_feature: Feature
        :param split: the way the splitting is done into a test data and
        evaluation data, defaults to 0.8
        :type split: float, optional
        :raises ValueError: raises an error if you try to use
        regression model on categorical features
        :raises ValueError: raises an error if you try to use
        classification model on continuous features
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical":
            if model.type != "classification":
                raise ValueError(
                    "Model type must be classification"
                    " for categorical target feature"
                )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression"
                " for continuous target feature"
            )

    def __str__(self) -> str:
        """
        Returns a human-readable overview of the pipeline.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        A getter for the model.

        :return: the model
        :rtype: Model
        """
        return self._model

    @property
    def decoder_classes(self) -> np.ndarray:
        """
        Retrieves the classes from the decoder of the target
        features. This is used later for displaying the metrics results.

        :return: the classes from the decoder
        :rtype: np.ndarray
        """
        return self._decoder_classes

    def artifacts(
        self, pipeline_name: str, pipeline_version: str
    ) -> List[Artifact]:
        """
        Used to get the artifacts generated during
        the pipeline execution to be saved.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["LabelEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "dataset_id": self._dataset.id,
            "metrics": self._metrics,
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        model_artifact = self._model.to_artifact(
            name=f"{pipeline_name}_model:{self._model.name}",
            version=pipeline_version,
        )
        metadata = {
            "model_id": model_artifact.id,
            "model_type": model_artifact.type,
        }
        artifacts.append(
            Artifact(
                name=f"{pipeline_name}_config",
                type="pipeline",
                version=pipeline_version,
                data=pickle.dumps(pipeline_data),
                metadata=metadata,
            )
        )
        artifacts.append(model_artifact)
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        A helper method for registering an artifact in
        the self._artifact dictionary.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        The way the features are preprocessed.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]

        # Saved so that it can be displayed later to the user
        if "classes" in artifact:
            self._decoder_classes = artifact["classes"]
        else:
            self._decoder_classes = None

        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, _, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (_, data, _) in input_results]

    def _split_data(self) -> None:
        """
        Split the data into training and testing sets.
        """
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.ndarray:
        """
        Join a sequence of vectors.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        A way to train a model.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self, X_vals: np.ndarray, Y_vals: np.ndarray) -> None:
        """
        A way to evaluate a model.
        """
        X = self._compact_vectors(X_vals)
        Y = Y_vals
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """
        The way to execute a pipeline.
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        # Evaluate on the test data
        self._evaluate(self._test_X, self._test_y)

        test_evaluation = {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }

        # Evaluate on the train data
        self._evaluate(self._train_X, self._train_y)

        train_evaluation = {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
        return test_evaluation, train_evaluation

    def train(self) -> None:
        """
        Used to train the model
        """
        self._preprocess_features()
        self._split_data()
        self._train()

    def predict(
        self, input_features: list[Feature], dataset: Dataset
    ) -> np.ndarray:
        """
        Used to predict data using input data

        :param input_features: the input features
        :type input_features: list[Feature]
        :param dataset: the dataaset
        :type dataset: Dataset
        :return: the predictions
        :rtype: np.ndarray
        """
        input_results = preprocess_features(input_features, dataset)
        input_vectors = [data for (_, data, _) in input_results]
        observations = [vector for vector in input_vectors]
        X_vals = self._compact_vectors(observations)
        predictions = self._model.predict(X_vals)
        return predictions
