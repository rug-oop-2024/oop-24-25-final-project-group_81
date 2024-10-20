import numpy as np
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
from sklearn.linear_model import Lasso


class Lasso(Model):
    """
    Wrapper around the Lasso model from scikit-learn that follows the same
    structure as the abstract base class from BaseModel.
    """

    _model: Lasso = PrivateAttr(default_factory=Lasso)

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
        predictions = predictions.reshape(predictions.shape[0], 1).round(2)
        return predictions

    def _validate_fit(self) -> None:
        """
        Checks if model has been fitted

        Raises:
            ValueError: If model has not stored 'coefficients' or 'intercept'
        """
        if (
            "coefficients" not in self._parameters
            or "intercept" not in self._parameters
        ):
            raise ValueError("The model has not been fitted!")
