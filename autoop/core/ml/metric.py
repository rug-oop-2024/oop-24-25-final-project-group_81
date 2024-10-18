from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np

from autoop.core.ml.utils import preprocess_input

METRICS = [
    "mean_squared_error",
    "accuracy",
]

def get_metric(name: str) -> "Metric":
    """
    This function creates an instance of a metric.

    :param name: the name of the metric
    :name type: str
    :return metric: an insance of a metric class.
    :type return: Metric
    """
    if name not in METRICS:
        raise ValueError(
            f"The metric you provided: {name} is not a viable metric!"
            )
    if name == "mean_squared_error":
        metric = MSEMetric()
    
    if name == "accuracy":
        metric = AccuracyMetric()

    return metric

class Metric(ABC):
    """
    Base class for all metrics.
    """
    def __call__(self, ground_truths: Any, predictions: Any):
        return self._result(ground_truths, predictions)

    @abstractmethod
    def _result(self, ground_truths: Any, predictions: Any) -> float:
        """
        The way a Metric computes the result

        :param ground_truths: the ground thruts
        :type ground_truths: Any
        :param predictions: the predictions
        :type predictions: Any
        :return: the result
        :rtype: float
        """
        pass

class AccuracyMetric(Metric):
    """
    An Accuracy metric class.
    """
    @preprocess_input
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the accuracy metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        accuracy = np.mean(ground_truths == predictions)
        return accuracy
        
class MSEMetric(Metric):
    """
    A Mean-Square Error metric class.
    """
    @preprocess_input
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the MSE metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        mse = np.mean((ground_truths - predictions) ** 2)
        return mse
    