import streamlit as st
import pandas as pd
import os

from app.core.system import AutoMLSystem
from app.core.streamlit_utils import GeneralUI
from app.core.datasets_utils import ControllerWithDatasets
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import CLASSIFICATION_METRICS, REGRESSION_METRICS
from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

class UserInterfaceModelling(GeneralUI):
    def __init__(self):
        self.action = None

    def display_model_types(self) -> str:
        """
        Display the model types that can be used in the pipeline.

        :return selected_model: the type of model the user selected.
        :type return: str
        """
        st.write("Chose the type of model you wish to use.")
        selected_model = st.selectbox(
            "Options",
            ["Select Model Type", "Regression Model", "Classification Model"]
            )
        return selected_model

    def display_models(self, models_list: list[str]) -> str:
        """
        Displays the avaliable models to be used in making
        the pipeline.

        :param models_list: the list of avaliable models
        :type models_list: list[str]
        :return: the selected model.
        :rtype: str
        """
        st.subheader("Avaliable models")
        selected_model = st.selectbox("Choose a Model", models_list)
        return selected_model
    
    def display_model_info(self, model_name: str) -> None:
        """
        Used to display some general info about the selected model by
        reading it from a file stored in "assets\\model_descriptions\\".
        The name of the file should follow the semantics of the model name.
        `Model Name` -> `model_name.txt`.

        :param model_name: the selected model name
        :type model_name: str
        """
        st.subheader(model_name)
        filename = model_name.\
            lower().\
                replace(" ", "_")
        file_path = "assets\\model_descriptions\\" + filename + ".txt"
        working_dir = os.getcwd()
        full_path = os.path.join(working_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as file:
            description = file.read()
        st.write(description)

    def display_metrics(self, metrics_list: list[str]) -> list[str]:
        """
        A way of displaying the matrics avaliable for the
        specific model type.

        :param metrics_list: a list of metrics as string
        :type metrics_list: list[str]
        :return: the selected metrics
        :rtype: list[str]
        """
        metrics_display_list = [
            metric.\
                replace("_", " ").\
                    title()
                    for metric in metrics_list
            ]
        st.subheader("Chose metric/s to evaluate the model.")
        metrics = st.multiselect("Avaliable metrics:", metrics_display_list)
        metrics_indecies = [
            index
            for index, value in enumerate(metrics_display_list)
            if value in metrics
            ]
        selected_metrics = [metrics_list[i] for i in metrics_indecies]
        return selected_metrics
    

class ControllerModelling(ControllerWithDatasets):
    def __init__(self):
        self.ui_manager = UserInterfaceModelling()
        self._selected_model_type = None
        self._selected_model = None
        self._dataset = None
        self._metrics = None

    def run(self):
        """
        Main loop to run the application.
        """
        self._handle_view_saved_datasets()
        if self._dataset is not None:
            self._detect_feature_types()
            self._select_model()
            self._chose_metrics()

    def _build_pipeline(self):
        pass

    def _get_dataset(
            self,
            selected_dataset_id: str
            ) -> Dataset:
        """
        Creates an instance of the selected dataset using
        its id. It overwrites the method in ControllerWithDatasets() class
        by adding a way of saving the slected dataset in the construcor, this
        ensures consistency with the logic of the parent class and adds the
        ability to save the dataset in question.
        
        Sets the variable self._dataset to be equal to the chosen dataset.

        :param selected_dataset_id: the id of the selected dataset
        :type selected_dataset_id: str
        :return: an instance of the selected dataset
        :rtype: Dataset
        """
        dataset = ControllerWithDatasets()._get_dataset(selected_dataset_id)
        if dataset != "Select a set":
            self._dataset = dataset
        return dataset

    def _select_model(self):
        """
        Handle select model type and specific model logic.
        """
        model_type = self.ui_manager.display_model_types()

        if model_type == "Select Model Type":
            st.write("Please select a model to continue.")
            model_type = None

        if model_type == "Regression Model":
            self.ui_manager.progress_bar()
            self.ui_manager.\
                display_success("You chose to use a regression model.")
            selected_model = self._select_specific_model(
                REGRESSION_MODELS
                )
            self._selected_model = selected_model

        elif model_type == "Classification Model":
            self.ui_manager.progress_bar()
            self.ui_manager.\
                display_success("You chose to use a classification model.")
            selected_model = self._select_specific_model(
                CLASSIFICATION_MODELS
                )
            self._selected_model = selected_model

        self._selected_model_type = model_type

    def _select_specific_model(self, model_list: list[str]):
        selected_model = self.ui_manager.display_models(model_list)
        self.ui_manager.display_model_info(selected_model)
        return selected_model
        
    def _chose_metrics(self):
        selected_metrics = None

        if self._selected_model_type == "Classification Model":
            selected_metrics = self.ui_manager.\
                display_metrics(CLASSIFICATION_METRICS)

        if self._selected_model_type == "Regression Model":
            selected_metrics = self.ui_manager.\
                display_metrics(REGRESSION_METRICS)

        if selected_metrics is not None:
            metrics_list = []
            for metric in selected_metrics:
                metric_instance = get_metric(metric)
                metrics_list.append(metric_instance)

            self._metrics = metrics_list

    def _detect_feature_types(self):
        return detect_feature_types(self._dataset)


if __name__ == "__main__":
    control = ControllerModelling()
    control.run()
