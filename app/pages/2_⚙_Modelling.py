import streamlit as st
import pandas as pd
import os
import time

from app.core.system import AutoMLSystem
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

class UserInterfaceModelling:
    def __init__(self):
        self.action = None

    def display_model_types(self) -> str:
        """
        Display the model types that can be used in the pipeline.
        """
        st.write("Chose the type of model you wish to use.")
        selected_model = st.selectbox("Options", ["Select Model Type", "Regression Model", "Classification Model"])
        return selected_model

    def display_models(self, models_list: list[str]) -> str:
        """
        Display the list of saved datasets and allow the user to select one.
        """
        st.subheader("Avaliable models")
        selected_model = st.selectbox("Choose a Model", models_list)
        return selected_model
    
    def display_model_info(self, model_name: str, filename: str):
        st.subheader(model_name)
        file_path = "assets\\model_descriptions\\" + filename
        working_dir = os.getcwd()
        full_path = os.path.join(working_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as file:
            description = file.read()
        st.write(description)

    def display_dataset_choices(self, dataset_list) -> str:
        """
        Display the list of saved datasets and allow the user to select one.
        """
        st.subheader("Chose a Dataset to Model")
        selected_dataset = st.selectbox("Datasets:", dataset_list)
        return selected_dataset
    
    def display_dataset(self, df: pd.DataFrame) -> None:
        """
        Display the dataset as a dataframe.
        """
        st.write("Preview of Dataset:")
        st.dataframe(df)

    def display_error(self, message) -> None:
        """
        Display error message.
        """
        st.error(message)

    def display_success(self, message) -> None:
        """
        Display success message.
        """
        st.success(message)

class ControllerModelling:
    def __init__(self):
        self.ui_manager = UserInterfaceModelling()

    def run(self):
        """
        Main loop to run the application.
        """
        selected_model = self._handle_select_model()
        if selected_model is not None:
            self._display_datasets()

    def _handle_select_model(self):
        """
        Handle dataset upload logic.
        """
        model_type = self.ui_manager.display_model_types()

        if model_type == "Select Model Type":
            st.write("Please select a model to continue.")
            selected_model = None

        if model_type == "Regression Model":
            self._progress_bar()
            self.ui_manager.display_success("You chose to use a regression model.")
            selected_model = self._select_specific_model(REGRESSION_MODELS)

        elif model_type == "Classification Model":
            self._progress_bar()
            self.ui_manager.display_success("You chose to use a classification model.")
            selected_model = self._select_specific_model(CLASSIFICATION_MODELS)

        return selected_model

    def _select_specific_model(self, model_list: list[str]):
        selected_model = self.ui_manager.display_models(model_list)
        selected_model_description_filename = selected_model.lower().replace(" ", "_") + ".txt"
        self.ui_manager.display_model_info(selected_model, selected_model_description_filename)
        return selected_model
    
    def _display_datasets(self):
        datasets = automl.registry.list(type="dataset")
        dataset_dict = datasets
        if not dataset_dict:
            self.ui_manager.display_error("No datasets available.")
            return
        
        dataset_id_lists = []
        dataset_lists = []
        for dataset in dataset_dict:
            name = dataset.name
            version = dataset.version 
            display_name = name + " " + "(version" + " " + version + ")"
            dataset_lists.append(display_name)
            dataset_id_lists.append(dataset.id)

        selected_dataset = self.ui_manager.display_dataset_choices(dataset_lists)

        # Retrieveing the id of the selevted dataset
        selected_id_index = dataset_lists.index(selected_dataset)
        selected_dataset_id = dataset_id_lists[selected_id_index]

        if selected_dataset:
            #try:
            selected_artifact = automl.registry.get(selected_dataset_id)
            artifact_attributes = vars(selected_artifact)
            dataset = Dataset(**artifact_attributes)
            df = dataset.read()
            self.ui_manager.display_dataset(df)

    def _progress_bar(self):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

if __name__ == "__main__":
    control = ControllerModelling()
    control.run()
