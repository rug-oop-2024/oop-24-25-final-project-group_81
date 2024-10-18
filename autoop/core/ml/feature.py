from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature():
    def __init__(self, name: str, feature_type: str):
        """
        This is the way you instatiate a Feature.

        :param name: the name of the feature.
        :type name: str
        :param feature_type: the type of the feature.
        :type feature_type: str
        """
        self._name = name
        self._type = feature_type

    @property
    def name(self) -> str:
        """
        Getter for the _name attribute.

        :return: the name of the column.
        :rtype: str
        """
        return self._name
    
    @property
    def type(self) -> str:
        """
        Getter for the _type attribute.

        :return: the feature of the column.
        :rtype: str
        """
        return self._type
    
    def __str__(self) -> str:
        """
        A way of displaying a Feature object.

        :return: `Column: name; Feature type: type`
        :rtype: str
        """
        return f"Column: {self._name}; Feature type: {self._type}"
    