from autoop.core.ml.model import Model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np


class PolynomialRegression(Model):
    def __init__(self, type: str) -> None:
        """
        A way of instantiating a Poly Regression.

        :param type: _description_
        :type type: _type_
        """
        super().__init__(type)
        self._model = None
        self._degree = 2

    @property
    def degree(self) -> int:
        """
        Getter for the degree of the polynomial.
        """
        return self._degree

    @degree.setter
    def degree(self, degree: int) -> None:
        """
        Setter for the degree of the polynomial.
        """
        self._degree = degree

    def fit(
            self,
            observations: np.ndarray,
            ground_truth: np.ndarray
        ) -> np.ndarray:
        """
        Fit the model.

        :param observations: Input data (features).
        :param ground_truth: Actual values (targets) to fit the model to.
        :return: Trained model parameters (weights).
        """
        self._validate_input(observations, ground_truth)

        X_val = self._preprocess_data(observations)

        self._model = LinearRegression()
        self._model.fit(X_val, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict values based on the fitted model.

        :param observations: The input data for which predictions are made.
        :return: The predicted values.
        """
        self._validate_num_features(observations)

        X_val = self._preprocess_data(observations)

        predictions = self._model.predict(X_val)

        return predictions

    def _preprocess_data(self, observations: np.ndarray) -> np.ndarray:
        """
        Preprocess tha data to perfrom a polynomial regression.

        :param observations: the observed data
        :type observations: np.ndarray
        :return: the processed data
        :rtype: np.ndarray
        """
        poly = PolynomialFeatures(degree=self._degree)
        observations_poly = poly.fit_transform(observations)
        return observations_poly
