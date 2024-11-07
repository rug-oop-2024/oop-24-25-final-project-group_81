import pandas as pd
import streamlit as st

from autoop.core.ml.dataset import Dataset
from app.core.abstract_server import AbstractServer

   
class ControllerWithDatasets(AbstractServer):
    """
    This class serves as a parent class for all classes trying to
    view the database for datasets. It inherits from AbstractServer
    which is the generic way of viewing items in the database.
    """
    def __init__(self):
        """
        A way of insantiating the object. Sets the constructor
        of the AbstractServer in such a way as to look for `dataset`
        items in the database.
        """
        super().__init__(item_name = "dataset")

    def _handle_view_saved_datasets(self):
        """
        Handle dataset viewing logic. Uses the
        `_handle_view_saved_items_logic` method of AbstractServer
        which gets the attributes of the item and instantiates a Dataset
        object that is displayed and later saved in the
        `self._dataset` attribute.
        """
        artifact_attributes = super()._handle_view_saved_items_logic()
        if artifact_attributes is not None:
            dataset = Dataset(**artifact_attributes)
            df = dataset.read()
            self.\
                _display_item(df)
            self._dataset = dataset

    def _display_item(self, df: pd.DataFrame):
        """
        Display the dataset as a dataframe.

        :param df: the dataframe to be dispalyed
        """
        st.write("Preview of Dataset:")
        st.dataframe(df, hide_index = True)
        