import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from app.core.streamlit_utils import GeneralUI

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


class UIWithDatasets(GeneralUI):
    """
    Class for UI, purely for consistency with
    the architecture, of how pages are constructed.
    """
    def __init__(self):
        self._actions = None

   
class ControllerWithDatasets:
    """
    This class serves as a parent class for all classes trying to
    view the database for datasets.
    """
    def __init__(self):
        """
        A way of insantiating an object.
        """
        self.ui_manager = UIWithDatasets()

    def _handle_view_saved_datasets(self):
        """
        Handle dataset viewing logic.
        """
        dataset_dict = datasets
        if not dataset_dict:
            self.ui_manager.display_error("No datasets available.")
            return
        
        dataset_list, dataset_id_list = self.\
            _get_name_and_id_lists(dataset_dict)

        selected_dataset = self.\
            _display_saved_datasets(dataset_list)

        selected_dataset_id = self.\
            _get_id(selected_dataset, dataset_list, dataset_id_list)

        if selected_dataset != "Select a set":
            self.ui_manager.progress_bar()
            dataset = self.\
                _get_dataset(selected_dataset_id)
            df = dataset.read()
            self.\
                _display_dataset(df)

    def _display_dataset(
            self,
            df: pd.DataFrame
            ) -> None:
        """
        Display the dataset as a dataframe.

        :param df: the dataframe to be dispalyed
        """
        st.write("Preview of Dataset:")
        st.dataframe(df)

    def _display_saved_datasets(
            self,
            dataset_list: list[Dataset]
            ) -> str:
        """
        Display the list of saved datasets and allow the user to select one.

        :param dataset_list: the list of saved datasets
        :type dataset_list: list[Dataset]
        :return selcted_dataset: the choice of the user
        :type return: str
        """
        st.subheader("View Existing Datasets")
        selected_dataset = st.selectbox("Choose a Dataset", dataset_list)
        return selected_dataset

    def _get_name_and_id_lists(
            self,
            dataset_dict: dict[Dataset]
            ) -> tuple[list[str], list[str]]:
        """
        Creates two list, one containing the names of the datasets
        that are avaliable along with their version. The other
        contains the ids of said lists.

        :param dataset_dict: the dictinary containing the datasets
        :type dataset_dict: dict[Dataset]
        :return: a list with names and a list with ids
        :rtype: tuple[list[str], list[str]]
        """
        dataset_id_list = ["No Id"]
        dataset_list = ["Select a set"]
        for dataset in dataset_dict:
            name = dataset.name
            version = dataset.version 
            display_name = name + " " + "(version" + " " + version + ")"
            dataset_list.append(display_name)
            dataset_id_list.append(dataset.id)
        return dataset_list, dataset_id_list
    
    def _get_id(
            self,
            selected_dataset: str,
            dataset_list: list[str],
            dataset_id_list: list[str]
            ) -> str:
        """
        A way of getting the id of the slected dataset.

        :param selected_dataset: the selected dataset's name
        :type selected_dataset: str
        :param dataset_list: the list of avaliable datasets
        :type dataset_list: list[str]
        :param dataset_id_list: the list of the ids of the avaliable
        datsets
        :type dataset_id_list: list[str]
        :return: the id of the selected dataset
        :rtype: str
        """
        selected_id_index = dataset_list.index(selected_dataset)
        selected_dataset_id = dataset_id_list[selected_id_index]
        return selected_dataset_id
    
    def _get_dataset(
            self,
            selected_dataset_id: str
            ) -> Dataset:
        """
        Creates an instance of the selected dataset using
        its id.

        :param selected_dataset_id: the id of the selected dataset
        :type selected_dataset_id: str
        :return: an instance of the selected dataset
        :rtype: Dataset
        """
        selected_artifact = automl.registry.get(selected_dataset_id)
        artifact_attributes = vars(selected_artifact)
        dataset = Dataset(**artifact_attributes)
        return dataset
    