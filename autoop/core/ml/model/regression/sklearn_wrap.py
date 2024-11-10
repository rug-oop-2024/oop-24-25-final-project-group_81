import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import Lasso


class LassoWrapper(Model):
    """
    Wrapper around the Lasso model from scikit-learn that follows the same
    structure as the abstract base class from BaseModel.
    """

    def __init__(self, type: str) -> None:
        """
        A way of insantiating a LassoWrapper.

        :param type: the type of model
        :type type: str
        """
        super().__init__(type)
        self._model = Lasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Lasso model to the data.
        :param observations: Input data (features).
        :param ground_truth: Target values.
        """
        super()._validate_input(observations, ground_truth)

        self._model.fit(observations, ground_truth)
        self._parameters["coefficients"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict values using the fitted Lasso model.
        :param observations: Input data to predict on.
        :return: Predicted values.
        """
        self._validate_fit()
        super()._validate_num_features(observations)
        # Casting the array into the appripriate shape
        predictions = self._model.predict(observations)
        predictions = predictions.reshape(predictions.shape[0], 1)
        return predictions.round(2)

    def _validate_fit(self) -> None:
        """
        Checks if model has been fitted

        Raises:
            ValueError: If model has not stored 'coefficients' or 'intercept'
        """
        if "coefficients" or "intercept" not in self._parameters:
            raise ValueError("The model has not been fitted!")
