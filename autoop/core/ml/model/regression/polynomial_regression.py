from autoop.core.ml.model import Model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np

# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.fit_transform
# Documentation: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html

class PolynomialRegression(Model):
    def __init__(self, type):
        super().__init__(type)
        self._model = None
        self._degree = 2

    @property
    def degree(self):
        """
        Getter for the degree of the polynomial.
        """
        return self._degree
    
    @degree.setter
    def degree(self, degree: int):
        """
        Setter for the degree of the polynomial.
        """
        self._degree = degree

    def fit(
            self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> np.ndarray:
        """
        Fit the model.

        :param observations: Input data (features).
        :param ground_truth: Actual values (targets) to fit the model to.
        :return: Trained model parameters (weights).
        """
        self._validate_input(observations, ground_truth)

        # Fits transformer to the training data
        poly = PolynomialFeatures(degree=self._degree)
        observations_poly = poly.fit_transform(observations)

        # Makes a linear regression model with the transformed training data
        self._model = LinearRegression()
        self._model.fit(observations_poly, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict values based on the fitted model.

        :param observations: The input data for which predictions are made.
        :return: The predicted values.
        """
        self._validate_fit()
        self._validate_num_features(observations)

        predictions = self._model.predict(observations)

        return predictions

    def _validate_fit(self) -> None:
        """
        Checks if model has been fitted

        Raises:
            ValueError: If model has not stored 'weights'
        """
        if "weights" not in self._parameters:
            raise ValueError("The model has not been fitted!")
        