from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np


METRICS = [
    "mean_squared_error",
    "accuracy",
    "root_mean_square_error",
    "r_squred",
    "mean_average_error",
    "f1_score",
    "sensitivity",
    "precision",
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
    if name == "root_mean_square_error":
        metric = RMSEMetric()
    if name == "r_squred":
        metric = RsquaredMetric()
    if name == "mean_average_error":
        metric = MAEMetric()
    if name == "f1_score":
        metric = F1ScoreMetric()
    if name == "sensitivity":
        metric = SensitivityMetric()
    if name == "precision":
        metric = PrecisionMetric()
        
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

    def evaluate(self, predictions: Any, ground_truths: Any) -> float:
        """
        This method is a way for evaluating a metric.

        :param predictions: the predictions
        :type predictions: Any
        :param ground_truths: the ground thruths
        :type ground_truths: Any
        :return: the evaluation
        :rtype: float
        """
        return self._result(ground_truths, predictions)

#############################################################################
#####################     Classification Metrics    #########################
#############################################################################

class ClassificationMetric:
    """
    Helper class for classification metrics.
    """
    def _compute_tp_fp_tn_fn(
            self,
            ground_truths: np.ndarray,
            predictions: np.ndarray
            ) -> tuple[int, int, int, int]:
        """
        Compute True Positives, False Positives,
        True Negatives, and False Negatives for cassifications metrics.

        :param ground_truths: the ground truths
        :ground_truths type: np.ndarray
        :param predictions: predictions
        :predictions type: np.ndarray
        :return: a tuple of tp, fp, tn, fn.
        :type return: tuple(int,int,int,int)
        """
        # Get the number of all unique classes and said classes
        classes = np.unique(ground_truths)
        num_classes = len(classes)

        # Initialize arrays for the TP, FP, and FN
        tp = np.zeros(num_classes)
        fp = np.zeros(num_classes)
        tn = np.zeros(num_classes)
        fn = np.zeros(num_classes)

        # Calculate TP, FP, TN, FN for each class
        for i, class_ in enumerate(classes):
            tp[i] = np.sum(
                (ground_truths == class_) & (predictions == class_)
                )
            fp[i] = np.sum(
                (ground_truths != class_) & (predictions == class_)
                )
            tn[i] = np.sum(
                (ground_truths != class_) & (predictions != class_)
                )
            fn[i] = np.sum(
                (ground_truths == class_) & (predictions != class_)
                )
            
        return tp, fp, tn, fn


class AccuracyMetric(Metric, ClassificationMetric):
    """
    An Accuracy metric class. Measures the proportion of
    correctly predicted instances out of the total instances.
    """
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
        tp, fp, tn, fn = self._compute_tp_fp_tn_fn(ground_truths, predictions)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy
    
class PrecisionMetric(Metric, ClassificationMetric):
    """
    A Precision metric class. Measures the proportion of true positive
    predictions out of all positive predictions.
    """
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the precision metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        tp, fp, _, _ = self._compute_tp_fp_tn_fn(ground_truths, predictions)
        precision = tp / (tp + fp)
        return precision

class SensitivityMetric(Metric, ClassificationMetric):
    """
    A Sensitivity metric class. Measures the proportion of true
    positive predictions out of all actual positive predicitons.
    """
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the sensitivity metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        tp, _, _, fn = self._compute_tp_fp_tn_fn(ground_truths, predictions)
        sensitivity = tp / (tp + fn)
        return sensitivity

class F1ScoreMetric(PrecisionMetric, SensitivityMetric):
    """
    A F1 Score metric class. The F1 Score is the harmonic mean
    of precision and sensitivity. It balances the two metrics
    which is useful when dealing with an imbalanced dataset.
    """
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the F1 Score metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        precision = PrecisionMetric._result(ground_truths, predictions)
        sensitivity = SensitivityMetric._result(ground_truths, predictions)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        return f1_score

#############################################################################
#######################     Regression Metrics    ###########################
#############################################################################

class MSEMetric(Metric):
    """
    A Mean-Square Error metric class.
    Measures the average of the squares of the errors.
    """
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

class MAEMetric(Metric):
    """
    A Mean-Average Error metric class.
    Measures the average of the absolute differences between
    the predicted and the actual values.
    """
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the MAE metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        mae = np.mean(
            np.abs(ground_truths - predictions)
            )
        return mae

class RsquaredMetric(Metric):
    """
    A R-squred metric class.
    Measures the average of the absolute differences between
    the predicted and the actual values.
    """
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the R-squared metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        # Calculating the SS total, and SS residuals
        ss_total = self._total_ss(ground_truths)
        ss_res = self._residual_ss(ground_truths, predictions)

        r_squared = 1 - ss_res / ss_total
        return r_squared

    def _total_ss(
            self,
            ground_truths: np.ndarray
            ) -> np.ndarray:
        """
        Calculates the total SS

        :param ground_truths: ground truths
        :type ground_truths: np.ndarray
        :return: the total SS
        :rtype: np.ndarray
        """
        mean = np.mean(ground_truths)
        total_ss = np.sum((ground_truths - mean) ** 2)
        return total_ss
    
    def _residual_ss(
            self,
            ground_truths: np.ndarray,
            predictions: np.ndarray
            ) -> np.ndarray:
        """
        Calculates the residuals SS

        :param ground_truths: ground truths
        :type ground_truths: np.ndarray
        :return: the residual SS
        :rtype: np.ndarray
        """
        ss_res = np.sum((ground_truths - predictions) ** 2)
        return ss_res

class RMSEMetric(MSEMetric):
    """
    A Root-Mean-Square Error metric class.
    Measures the average of the absolute differences between
    the predicted and the actual values.
    """
    def _result(
            self,
            ground_truths: List[float],
            predictions: List[float]
            ) -> float:
        """
        This method computes the result of the RMSE metric.

        :param ground_truths: the ground thruts
        :type ground_truths: List[float] or np.ndarray
        :param predictions: the predictions
        :type predictions: List[float] or np.ndarray
        :return: the result
        :rtype: float
        """
        mse = MSEMetric._result(ground_truths, predictions)
        rmse = mse ** 0.5
        return rmse
    