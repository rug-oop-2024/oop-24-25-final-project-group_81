import numpy as np

from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    """
    A multiple linear regression model.
    """

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Fit the model using the Normal Equation.

        :param observations: Input data (features).
        :param ground_truth: Actual values (targets) to fit the model to.
        :return: Trained model parameters (weights).
        """
        self._validate_input(observations, ground_truth)

        observations_with_intercept = self._add_trailing_one_to_observations(
            observations
        )

        covariance_matrix = observations_with_intercept.T @ observations_with_intercept

        determinant = np.linalg.det(covariance_matrix)

        if determinant == 0:
            # The matrix is singular and cannot be inverted
            inverse_covariance_matrix = np.linalg.pinv(covariance_matrix)
        else:
            inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        weights = (
            inverse_covariance_matrix @ observations_with_intercept.T @ ground_truth
        )
        self._parameters["weights"] = weights

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict values based on a fitted model.

        :param observations: The input data for which predictions are made.
        :return: The predicted values.
        """
        self._validate_fit()
        self._validate_num_features(observations)

        observations_with_intercept = self._add_trailing_one_to_observations(
            observations
        )
        predictions = observations_with_intercept @ self._parameters["weights"]
        predictions = np.array(predictions).round(2)
        return predictions

    def _validate_fit(self) -> None:
        """
        Checks if model has been fitted

        Raises:
            ValueError: If model has not stored 'weights'
        """
        if "weights" not in self._parameters:
            raise ValueError("The model has not been fitted!")

    def _add_trailing_one_to_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        Add a column of ones to the beginning of a matrix
        representing observations.

        :param observations: Input data.
        :return: Array with a column of ones added.
        """
        num_samples = observations.shape[0]
        ones_column = np.ones(
            (num_samples, 1)
        )  # A column of ones for the intercept term
        observations_with_intercept = np.hstack([ones_column, observations])
        return observations_with_intercept
