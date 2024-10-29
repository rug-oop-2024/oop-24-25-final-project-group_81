import streamlit as st
import pandas as pd
import os
from typing import Any

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from app.core.streamlit_utils import GeneralUI
from app.core.datasets_utils import ControllerWithDatasets


automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

class UserInterfaceDatasets(GeneralUI):
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

class ControllerDatasets(ControllerWithDatasets):
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
            self._handle_view_saved_datasets()

    def _handle_upload_dataset(self):
        """
        Handle dataset upload logic.
        """
        uploaded_file, dataset_name, version = self.\
            ui_manager.get_dataset_upload_info()

        if uploaded_file:
            # Read the uploaded file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            self._display_dataset(df)

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

if __name__ == "__main__":
    control = ControllerDatasets()
    control.run()