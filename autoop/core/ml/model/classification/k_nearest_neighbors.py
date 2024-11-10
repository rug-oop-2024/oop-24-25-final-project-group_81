import numpy as np
from collections import Counter

from autoop.core.ml.model import Model


class KNearestNeighbors(Model):
    """
    The `KNearestNeighbors` class
    implements a k-nearest neighbors algorithm
    for classification.

    It impliments methods for fitting the model and making predictions
    based on Euclidean distances.
    """

    def __init__(self, type, k_val: int = 3) -> None:
        super().__init__(type)
        self.k = k_val

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        The `fit` method takes in observations and ground truth data,
        validates them, and assigns parameters based on the observations
        and ground truth labels.

        :param observations: An array containing the
        data points or observations for training the model.
        Each row in the array represents a single data point,
        and the columns represent different features or
        dimensions of the data points
        :type observations: np.ndarray
        :param ground_truth: An array containing the true labels
        or categories corresponding to the observations
        in the `observations` array
        :type ground_truth: np.ndarray
        """
        super()._validate_input(observations, ground_truth)
        self._parameters["training_data"] = observations
        self._parameters["training_labels"] = ground_truth

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        The `predict` method takes in observations, calculates the
        k-nearest neighbors using Euclidean distance, and returns
        the most common class label among the k-nearest neighbors for each
        observation.

        :param observations: The data points for which you want to make
        predictions. Each row in the array represents
        an observation with its features.
        :type observations: np.ndarray
        :return: The `predict` method returns an array of
        predicted class labels for the input observations based on the
        k-nearest neighbors algorithm.
        """
        self._validate_fit()
        super()._validate_num_features(observations)

        predictions = []

        for observation in observations:
            distances = self._L2_norm(observation)
            sorted_indices = distances[:, 0].argsort()[: self.k]
            closest_labels = distances[sorted_indices, 1]
            most_common_class = Counter(closest_labels).most_common(1)[0][0]
            predictions.append(most_common_class)

        return np.array(predictions)

    def _validate_fit(self) -> None:
        """
        Checks if model has been properly fitted

        Raises:
            ValueError: If model has not stored
                'training_data' or 'training_labels'
        """
        if (
            "training_data" not in self._parameters
            or "training_labels" not in self._parameters
        ):
            raise ValueError("The model has not been fitted!")

    def _L2_norm(self, observation: np.ndarray) -> np.ndarray:
        """
        Calculate the L2 (Euclidean) distance between
          a single observation and all training data.

        :param observation: A single observation
          for which distances are to be calculated.
        :return: A 2D array where each row contains
          the distance and the corresponding label.
        """
        training_data = self._parameters["training_data"]
        distances = np.linalg.norm(training_data - observation, axis=1)
        labels = self._parameters["training_labels"]
        return np.column_stack((distances, labels))
