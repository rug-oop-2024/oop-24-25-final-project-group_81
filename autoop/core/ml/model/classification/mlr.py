from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression

import numpy as np


class MultinomialLogisticRegression(Model):
    """
    This is a wrapper of sklearn's LogisticRegression.
    """

    def __init__(self, type: str) -> None:
        """
        A way of instantiating MultinomialLogisticRegression.
        """
        super().__init__(type)
        self._model = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the observations and ground truth data.

        :param observations: training data
        :type observations: np.ndarray
        :param ground_truth: the classification
        :type ground_truth: np.ndarray
        """
        self._validate_input(observations, ground_truth)
        self._model = LogisticRegression(multi_class="auto")
        self._model.fit(observations, ground_truth)
        params = self._model.get_params()
        self._parameters = params

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts observations based on the fitted model.

        :param observations: the observations to be fitted
        :type observations: np.ndarray
        :return: the predictions
        :rtype: np.ndarray
        """
        self._validate_fit()
        predictions = self._model.predict(observations)
        return predictions

    def _validate_fit(self) -> None:
        """
        Used to validate if the model has been fitted.

        :raises ValueError: an exception if the model is not
        fitted.
        """
        if self._model is None:
            raise ValueError("The model has not been fitted!")
