import streamlit as st
import pandas as pd
import os

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

class UserInterface:
    def __init__(self):
        self.action = None

    def render_sidebar(self):
        """Render the sidebar for selecting an action."""
        st.sidebar.header("Actions")
        self.action = st.sidebar.selectbox("Choose Action", ["Upload Dataset", "View Datasets"])

    def get_dataset_upload_info(self):
        """UI for uploading a dataset and entering metadata."""
        st.subheader("Upload a New Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        dataset_name = st.text_input("Dataset Name", value="MyBigDataset")
        version = st.text_input("Version", value="1.0")

        return uploaded_file, dataset_name, version

    def display_dataset(self, df: pd.DataFrame):
        """Display the dataset in the Streamlit app."""
        st.write("Preview of Dataset:")
        st.dataframe(df)

    def display_saved_datasets(self, dataset_list):
        """Display the list of saved datasets and allow the user to load one."""
        st.subheader("View Existing Datasets")
        selected_dataset = st.selectbox("Choose a Dataset", dataset_list)
        return selected_dataset

    def dataset_loaded_success(self, dataset_name):
        """Success message for loading a dataset."""
        st.success(f"Dataset '{dataset_name}' loaded successfully!")

    def display_error(self, message):
        """Display error message."""
        st.error(message)

    def display_success(self, message):
        """Display success message."""
        st.success(message)


class Controller:
    def __init__(self):
        self.ui_manager = UserInterface()

    def run(self):
        """Main loop to run the application."""
        self.ui_manager.render_sidebar()

        if self.ui_manager.action == "Upload Dataset":
            self.handle_upload_dataset()
        elif self.ui_manager.action == "View Datasets":
            self.handle_view_datasets()

    def handle_upload_dataset(self):
        """Handle dataset upload logic."""
        uploaded_file, dataset_name, version = self.ui_manager.get_dataset_upload_info()

        if uploaded_file:
            # Read the uploaded file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            self.ui_manager.display_dataset(df)

            if st.button("Save Dataset"):
                # Create and save the dataset
                dataset = Dataset.from_dataframe(df, name=dataset_name, asset_path=dataset_name, version=version)
                automl.registry.register(dataset)
                self.ui_manager.display_success(f"Dataset '{dataset_name}' saved successfully!")

    def handle_view_datasets(self):
        """Handle dataset viewing logic."""
        dataset_dict = datasets
        st.write(dataset_dict)
        # if not dataset_dict:
        #     self.ui_manager.display_error("No datasets available.")
        #     return
        
        # dataset_names = []
        # for dataset in dataset_dict:
        #     dataset_names.append(dataset.name)

        # selected_dataset = self.ui_manager.display_saved_datasets(dataset_names)

        # if selected_dataset:
        #     name, version = selected_dataset.split(':')
        #     try:
        #         dataset = load_dataset(name, version)
        #         df = dataset.read()
        #         self.ui_manager.dataset_loaded_success(name)
        #         self.ui_manager.display_dataset(df)
        #     except Exception as e:
        #         self.ui_manager.display_error(str(e))

if __name__ == "__main__":
    control = Controller()
    control.run()