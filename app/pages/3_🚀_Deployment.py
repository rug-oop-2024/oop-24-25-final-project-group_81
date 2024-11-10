import streamlit as st
import pandas as pd
import numpy as np

from app.core.streamlit_utils import GeneralUI
from app.core.deployment_utils import ControllerWithPipelines
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.feature import Feature


class UserInterfaceDeployment(GeneralUI):
    """
    User Interface for Deployment page
    """

    def get_csv_upload_info(self) -> pd.DataFrame:
        """
        UI for uploading a csv to predict on.
        """
        st.subheader("Upload a CSV you wish to do predictions on!")
        file = st.file_uploader("Choose a CSV file", type="csv")
        return file

    def display_csv(self, file: pd.DataFrame) -> pd.DataFrame:
        """
        Reads a CSV file and displays it

        :param file: a CSV file
        :type file: pd.DataFrame
        :return df: a panda dataframe of the csv file
        :type return: pd.DataFrame
        """
        df = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(df, hide_index=True)
        return df

    def display_csv_features(
        self, features: list[Feature], input_features: list[Feature]
    ) -> list[Feature]:
        """
        Displays a CSV features selection menu.

        :param features: all the features in the csv
        :type features: list[Feature]
        :param input_features: the input features of the pipeline
        :type input_features: list[Feature]
        :return: a list of selected features, based on the
        selection criterium
        :rtype: list[Feature]
        """
        # Get the selection criterium
        req_num_features, type_ = self.\
            _get_feature_selection_criterium(input_features)

        # Get a list of feature names
        list_of_features = [feature.name for feature in features]

        input_instructions = (
            "# In order to make a prediction you need to chose "
            + str(req_num_features)
            + " "
            + type_
            + " "
            + "features!"
        )

        # User input instructions
        st.write(input_instructions)

        selections = st.multiselect(
            f"Avaliable {type_} features:",
            list_of_features,
            max_selections=req_num_features,
        )

        # Get feature from selection
        selected_features = [
            feature for feature in features if feature.name in selections
        ]
        return selected_features

    def _get_feature_selection_criterium(
        self, input_features: list[Feature]
    ) -> tuple[int, str]:
        """
        Used to generate feature slection criterium.

        :param input_features: the input feature/s
        :type input_features: list[Feature]
        :return: a tuple containing the required number of features and
        the type of feature
        :rtype: _type_
        """
        required_num_features = len(input_features)
        type_of_features = input_features[0].type
        return required_num_features, type_of_features

    def display_predictions(self, dict_with_predictions: dict) -> None:
        """
        Used to displat the predictions of the pipeline.

        :param dict_with_predictions: a dictionary containg the observations
        and predictions as keys of a dictionary
        :type dict_with_predictions: dict
        """
        df = pd.DataFrame(dict_with_predictions)
        styled_df = df.style.apply(
            self._style_the_predictions, subset=["Predicted Values"], axis=0
        )
        st.dataframe(styled_df, hide_index=True)

    def _style_the_predictions(self, column: pd.DataFrame) -> list[str]:
        """
        A way to style the column with the predictions.

        :param column: the column
        :type column: pd.DataFrame
        :return: a style map
        :rtype: list[str]
        """
        return ["background-color: rgba(0, 255, 0, 0.2)"] * len(column)


class ControllerDeployment(ControllerWithPipelines):
    """
    Controller for the Deployment page
    """

    def __init__(self) -> None:
        """
        A way of instantiating a ControllerDeployment
        """
        super().__init__()
        self.ui_manager = UserInterfaceDeployment()
        self._pipeline = None

    def run(self) -> None:
        """
        Main loop to run the application.
        """
        # Load the starting page
        self._starting_page()

        # Load the Pipeline
        self._handle_view_saved_items_logic()
        if self._pipeline is not None:
            # Display the Pipeline
            self._display_item()

            # Load a csv and get its features
            features = self._load_csv()
            if features is not None:
                # Chose features to do predictions on
                selected_features = self.ui_manager.\
                    display_csv_features(
                    features, self._input_features
                )

                # Predict when sufficient number of features selected
                if len(selected_features) == len(self._input_features):
                    if st.button("Predict"):
                        predictions = self._predict(selected_features)
                        self._display_predictions(
                            predictions,
                            selected_features
                        )

    def _starting_page(self) -> None:
        """
        This is the starting page that the user sees.
        """
        st.set_page_config(page_title="Deployment", page_icon="🚀")

        st.write("# 🚀 Deployment")

        st.write(
            "You can use this page to load and deploy existing Pipelines!"
        )

    def _predict(self, selected_features: list[Feature]) -> np.ndarray:
        """
        Used to predict on the slected features using the logic of
        the Pipeline class.

        :param selected_features: the slected features
        :type selected_features: list[Feature]
        :return: an array of predictions
        :rtype: np.ndarray
        """
        self._pipeline.train()
        predictions = self._pipeline.predict(
            selected_features, self._dataset_to_predict
        )
        return predictions

    def _display_predictions(
        self, predictions: np.ndarray, selected_features: list[Feature]
    ) -> None:
        """
        A way of displaying the predictions after they have
        been made.

        :param predictions: the array of predictions
        :type predictions: np.ndarray
        :param selected_features: the selected features that the
        predictions are made on
        :type selected_features: list[Feature]
        """
        # Initialise a dictionary to store the columns and their values
        dict_with_predictions = {}

        # Retrieving the names of the selcted features
        # to be used as keys in the dict
        selected_features_list: list[str] = [
            feature.name for feature in selected_features
        ]

        # Populating the dict
        for col, val in self._dataset_to_predict.read().items():
            if col in selected_features_list:
                dict_with_predictions[col] = val
        predictions_list = predictions.flatten().tolist()

        # Populating the dict with the predicted values
        dict_with_predictions["Predicted Values"] = predictions_list

        # Using the UI to display the predictions
        self.ui_manager.display_predictions(dict_with_predictions)

    def _load_csv(self) -> None:
        """
        Handle CSV upload logic.
        """
        uploaded_file = self.ui_manager.get_csv_upload_info()

        if uploaded_file:
            # Read the uploaded file into a pandas DataFrame
            df = self.ui_manager.display_csv(uploaded_file)

            features = self._get_csv_features(df)

            return features

    def _get_csv_features(self, df: pd.DataFrame) -> list[Feature]:
        """
        Get the features of the uoladed CSV file.

        :param df: the data of the csv file as panda dataframe
        :type df: pd.DataFrame
        :return: the features of the dataframe
        :rtype: list[Feature]
        """
        # Create a dataset object
        self._dataset_to_predict: Dataset | None = Dataset().\
            from_dataframe(
            df, name="None", asset_path="None", version="None"
        )

        # Detect features
        all_features = detect_feature_types(self._dataset)

        # Sort out permited features, based on the input features
        permited_features = []
        for feature in all_features:
            if feature.type == self._input_features[0].type:
                permited_features.append(feature)

        return permited_features


if __name__ == "__main__":
    control = ControllerDeployment()
    control.run()
