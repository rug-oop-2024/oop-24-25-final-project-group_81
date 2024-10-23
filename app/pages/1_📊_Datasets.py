import streamlit as st
import pandas as pd
import os
from typing import Any

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

class UserInterfaceDatasets:
    def __init__(self):
        self.action = None

    def render_sidebar(self) -> None:
        """
        Render the sidebar for selecting an action.
        """
        st.sidebar.header("Actions")
        self.action = st.sidebar.\
            selectbox("Choose Action", ["Upload Dataset", "View Datasets"])

    def get_dataset_upload_info(self) -> tuple[Any, str, str]:
        """
        UI for uploading a dataset and entering metadata.
        """
        st.subheader("Upload a New Dataset")
        file = st.file_uploader("Choose a CSV file", type="csv")
        dataset_name = st.text_input("Dataset Name", value="MyBigDataset")
        version = st.text_input("Version", value="1.0")

        return file, dataset_name, version

    def display_dataset(self, df: pd.DataFrame) -> None:
        """
        Display the dataset as a dataframe.
        """
        st.write("Preview of Dataset:")
        st.dataframe(df)

    def display_saved_datasets(self, dataset_list) -> str:
        """
        Display the list of saved datasets and allow the user to select one.
        """
        st.subheader("View Existing Datasets")
        selected_dataset = st.selectbox("Choose a Dataset", dataset_list)
        return selected_dataset

    def dataset_loaded_success(self, dataset_name) -> None:
        """
        Success message for loading a dataset.
        """
        st.success(f"Dataset '{dataset_name}' loaded successfully!")

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


class ControllerDatasets:
    def __init__(self):
        self.ui_manager = UserInterfaceDatasets()

    def run(self):
        """
        Main loop to run the application.
        """
        self.ui_manager.render_sidebar()

        if self.ui_manager.action == "Upload Dataset":
            self._handle_upload_dataset()
        elif self.ui_manager.action == "View Datasets":
            self._handle_view_datasets()

    def _handle_upload_dataset(self):
        """
        Handle dataset upload logic.
        """
        uploaded_file, dataset_name, version = self.\
            ui_manager.get_dataset_upload_info()

        if uploaded_file:
            # Read the uploaded file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            self.ui_manager.display_dataset(df)

            if st.button("Save Dataset"):
                # Create and save the dataset
                dataset = Dataset.\
                    from_dataframe(
                        df,
                        name=dataset_name,
                        asset_path=dataset_name,
                        version=version
                        )
                automl.registry.register(dataset)
                self.ui_manager.\
                    display_success(
                        f"Dataset '{dataset_name}' saved successfully!"
                        )

    def _handle_view_datasets(self):
        """
        Handle dataset viewing logic.
        """
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

        selected_dataset = self.ui_manager.display_saved_datasets(dataset_lists)

        # Retrieveing the id of the selevted dataset
        selected_id_index = dataset_lists.index(selected_dataset)
        selected_dataset_id = dataset_id_lists[selected_id_index]

        if selected_dataset:
            #try:
            selected_artifact = automl.registry.get(selected_dataset_id)
            artifact_attributes = vars(selected_artifact)
            dataset = Dataset(**artifact_attributes)
            self.ui_manager.dataset_loaded_success(dataset.name)
            df = dataset.read()
            self.ui_manager.display_dataset(df)

if __name__ == "__main__":
    control = ControllerDatasets()
    control.run()