import streamlit as st
import pandas as pd
import os

from app.core.system import AutoMLSystem
from app.core.streamlit_utils import GeneralUI
from app.core.datasets_utils import ControllerWithDatasets
from autoop.core.ml.feature import Feature
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import CLASSIFICATION_METRICS, REGRESSION_METRICS
from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline


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

    def display_model_types(self, list_of_model_types: list[str]) -> str:
        """
        Display the model types that can be used in the pipeline.

        :return selected_model: the type of model the user selected.
        :type return: str
        """
        st.write("Chose the type of model you want to use.")
        selected_model = st.selectbox(
            "Options",
            list_of_model_types
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

    def display_metrics(
            self,
            metrics_list: list[str]
            ) -> list[str]:
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
    
    def display_input_features(
            self,
            features: list[Feature],
            feature_type: str
            ) -> list[str]:
        """
        Displays the avaliable input features and prompts the user
        to chose one or more from a multiselect box. It returns
        a list of the column names of the features selected by the user.

        :param features: a list of features
        :type features: list[Feature]
        :param feature_type: desired features type to filter
        all the features with
        :type feature_type: str
        :return: a list of the chosen features
        :rtype: list[str]
        """
        avaliable_columns = self._get_avaliable_features(features, feature_type)

        st.subheader("Chose input feature/s:")
        features_list = st.multiselect("Avaliable input feature/s:", avaliable_columns)
        return features_list

    def display_target_features(
            self,
            features: list[Feature],
            input_features: list[Feature],
            feature_type: str
            ) -> Feature:
        """
        Displays the remaining features, by considering the choice
        the user made for input feature/s, and prompts the user
        to chose target features.

        :param features: all the features
        :type features: list[Feature]
        :param input_features: the input features
        :type input_features: list[Feature]
        :param feature_type: the type of features
        :type feature_type: str
        :return: the selected target feature
        :rtype: str
        """
        avaliable_columns = []
        input_features_names = [name.get_column_type()[0] for name in input_features]
        for feature in features:
            name, type = feature.get_column_type()
            if type == feature_type and name not in input_features_names:
                avaliable_columns.append(feature)

        st.subheader("Chose target feature:")
        target_feature = st.selectbox("Avaliable target feature:", avaliable_columns)
        return target_feature
    
    def show_results(self, exection: dict):
        metrics_result = exection["metrics"]
        predictions_results = exection["predictions"]
        st.subheader("Metrics:")
        for metric_result in metrics_result:
            metric, result = metric_result
            st.write(f"{metric}: {round(result, 2)}")
        st.subheader("Predictions:")
        st.write(predictions_results)

    def _get_avaliable_features(self, features: list[Feature], feature_type: str) -> list[Feature]:
        avaliable_columns = []
        for feature in features:
            col, type = feature.get_column_type()
            if type == feature_type:
                avaliable_columns.append(feature)
        return avaliable_columns


class ControllerModelling(ControllerWithDatasets):
    def __init__(self):
        self.ui_manager = UserInterfaceModelling()
        self._reboot()

    def run(self):
        """
        Main loop to run the application.
        """
        self._handle_view_saved_datasets()
        if self._dataset is not None:
            self._select_model_type()

            if self._model_type != "Select Model Type":
                self._select_input_features()
                if self._input_features:
                    self._select_target_features()

        if self._target_features is not None:
            self._chose_metrics()

        if self._metrics is not None:
            self._select_model()

        if self._model is not None:
            self._chose_split()

        if self._split is not None:
            # valve = self.ui_manager.button("Create a Pipeline!")
            # if valve:
            self._build_pipeline()
        
        if self._pipeline is not None:
            st.write(self._pipeline)
            # train = self.ui_manager.button("Train!")
            # if train:
            self._exection = self._pipeline.execute()

        if self._exection is not None:
            test_evaluation, train_evaluation = self._exection
            st.header("Test Evaluation")
            self.ui_manager.show_results(test_evaluation)
            st.header("Train Evaluation")
            self.ui_manager.show_results(train_evaluation)

    def _build_pipeline(self):
        self._pipeline = Pipeline(
            self._metrics,
            self._dataset,
            self._model,
            self._input_features,
            self._target_features,
            self._split)

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
        It also gets the feature of the dataset, stores it in the
        self._feature variable.

        :param selected_dataset_id: the id of the selected dataset
        :type selected_dataset_id: str
        :return: an instance of the selected dataset
        :rtype: Dataset
        """
        dataset = ControllerWithDatasets()._get_dataset(selected_dataset_id)
        if dataset != "Select a set":
            self._dataset = dataset
            self._features = detect_feature_types(self._dataset)
        return dataset

    def _select_model(self):
        """
        Handle select model type and specific model logic.
        """
        if self._model_type == "Regression Model":
            self._chose_model(REGRESSION_MODELS, "regression")

        elif self._model_type == "Classification Model":
            self._chose_model(CLASSIFICATION_MODELS, "classification")

    def _select_model_type(self):
        # Full list of model types
        list_of_model_types = [
            "Select Model Type", "Regression Model", "Classification Model"
            ]

        # Cheecks the avaliable features in the dataset
        list_of_types = self._check_features_type()

        if "numerical" not in list_of_types:
            list_of_model_types.remove("Regression Model")

        if "categorical" not in list_of_types:
            list_of_model_types.remove("Classification Model")

        # Select the model type
        model_type = self.ui_manager.\
            display_model_types(list_of_model_types)
        
        if model_type == "Select Model Type":
            st.write("Please select a model type to continue.")
            self._feature_type = None
        
        if model_type == "Regression Model":
            self._feature_type = "numerical"

        if model_type == "Classification Model":
            self._feature_type = "categorical"

        self._model_type = model_type

    def _select_input_features(self):
        self._input_features = self.ui_manager.\
            display_input_features(
                self._features,
                self._feature_type
                )

    def _select_target_features(self):
        self._target_features = self.ui_manager.\
            display_target_features(
                self._features,
                self._input_features,
                self._feature_type
                )

    def _chose_model(self, list_of_models: list[str], type_of_models: str):
        self.ui_manager.progress_bar()
        self.ui_manager.\
            display_success(f"You chose to use a {type_of_models} model.")
        
        selected_model = self.ui_manager.display_models(list_of_models)
        self.ui_manager.display_model_info(selected_model)
        self._model = get_model(selected_model)
        
    def _chose_metrics(self):
        selected_metrics = None

        if self._model_type == "Classification Model":
            selected_metrics = self.ui_manager.\
                display_metrics(CLASSIFICATION_METRICS)

        if self._model_type == "Regression Model":
            selected_metrics = self.ui_manager.\
                display_metrics(REGRESSION_METRICS)

        if selected_metrics is not None:
            self._get_metrics(selected_metrics)

    def _get_metrics(self, selected_metrics):
        metrics_list = []
        for metric in selected_metrics:
            metric_instance = get_metric(metric)
            metrics_list.append(metric_instance)

        self._metrics = metrics_list

    def _check_features_type(self):
        list_of_types = []
        for feature in self._features:
            list_of_types\
                .append(
                    feature.type
                    )
        
        # In case the user has selected a dataset with 1 feature
        if len(list_of_types) == 1:
            self.ui_manager.display_error(
                "No avaliable model because of invalid dataset."
                "You can not do any analysis on a single column!"
                "You need to chose a different dataset."
                )
            self.ui_manager.progress_bar()
            self._reboot()
            return self.run()
        
        return list_of_types
    
    def _chose_split(self):
        self._split = st.slider("Select a dataset split", 0.01, 0.99)
    
    def _reboot(self):
        self._model_type = None
        self._model = None
        self._dataset = None
        self._metrics = None
        self._features = None
        self._feature_type = None
        self._input_features = None
        self._target_features = None
        self._split = None
        self._pipeline = None
        self._exection = None


if __name__ == "__main__":
    control = ControllerModelling()
    control.run()
