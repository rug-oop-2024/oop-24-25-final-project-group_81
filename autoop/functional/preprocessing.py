from typing import List, Tuple
import numpy as np

from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_features(
        features: List[Feature],
        dataset: Dataset
        ) -> List[Tuple[str, np.ndarray, dict]]:
    """Preprocess features.
    Args:
        features (List[Feature]): List of features.
        dataset (Dataset): Dataset object.
    Returns:
        List[str, Tuple[np.ndarray, dict]]:
        List of preprocessed features. Each ndarray of shape (N, ...)
    """
    results = []
    raw = dataset.read()
    for feature in features:
        if feature.type == "categorical":
            encoder = LabelEncoder()

            data = encoder.fit_transform(
                raw[feature.name].values
                )
            
            artifact = {
                "type": "LabelEncoder",
                "encoder": encoder.get_params(),
                "classes": encoder.classes_
                }
            
            results.append((feature.name, data, artifact))

        if feature.type == "numerical":
            scaler = StandardScaler()

            data = scaler.fit_transform(
                raw[feature.name].values.reshape(-1, 1)
                )
            
            artifact = {
                "type": "StandardScaler",
                "scaler": scaler.get_params()
                }
            
            results.append((feature.name, data, artifact))
            
    # Sort for consistency
    results = list(sorted(results, key=lambda x: x[0]))
    return results
