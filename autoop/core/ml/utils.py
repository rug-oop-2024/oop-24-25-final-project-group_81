from typing import Any, Callable
import numpy as np
from functools import wraps


def preprocess_input(function: Callable[..., Any]) -> Callable[..., Any]:
    """
    This is a decorator used to preprocess input for that is used
    to compute the results of a metric.

    :param function: the funcion used to proccess the result
    for the metric
    :type function: Callable[..., Any]
    :return: returns the function ensuring the input is a numpy array.
    :rtype: Callable[..., Any]
    """
    @wraps(function)
    def results(ground_truths: Any, predictions: Any) -> float:
        """
        The function computing the result of a metric.

        :param ground_truths: the ground thruths
        :type ground_truths: Any
        :param predictions: the predictions
        :type predictions: Any
        :return: the result of the metric
        :rtype: float
        """
        if isinstance(ground_truths, list):
            ground_truths = np.array(ground_truths)
        elif not isinstance(ground_truths, np.ndarray):
            raise ValueError("ground_truths must be a list or a numpy array")

        if isinstance(predictions, list):
            predictions = np.array(predictions)
        elif not isinstance(predictions, np.ndarray):
            raise ValueError("predictions must be a list or a numpy array")

        if ground_truths.shape != predictions.shape:
            raise ValueError("The shape of ground_truths and predictions must match")

        result = function(ground_truths, predictions)
        return result
    return results
