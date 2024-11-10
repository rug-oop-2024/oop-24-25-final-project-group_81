import streamlit as st
from typing import Any
from datetime import date, timedelta
import os

from app.core.streamlit_utils import GeneralUI
from pricepredictor.forcast.forecastFactory_initializer import ForcastFactoryInitializer
from pricepredictor.forcast.forcastFactory import ForcastFactory

        # stock_name: str = "AAPL",
        # number_of_predictions: int = 5,
        # raw_data_amount: int = 50,
        # sma_lookback_period: int = 3,
        # regression_window: int | None = None,
        # end_date: str = "2024-09-01",
        # interval: str = "1d",
        # architecture: list[int] = [13, 24],
        # learning_rate: int = 0.01,
        # loss_function: str = "mse",
        # metrics: list[str] = ["mae"],
        # epochs: int = 50,
        # batch_size: int = 5,
        # points_per_set: int = 10,
        # num_sets: int = 50,
        # labels_per_set: int = 1,
        # testing_percentage: float = 0.8,
        # validation_percentage: float = 0.1

class UserInterfaceStockPredictions(GeneralUI):
    def __init__(self):
        super().__init__()
        self._action_list = ["Information", "Setup the model"]
        self._sidebar_header = "Actions"

    def display_info(
            self,
            ) -> None:
        st.subheader("General Information")
        file_path = "assets\\model_descriptions\\pricepredictor.txt"
        working_dir = os.getcwd()
        full_path = os.path.join(working_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as file:
            description = file.read()
        st.write(description)
        
    def show_avaliable_stocks(self) -> str:
        st.write("# Chose a financial assset")
        selected_stocks = st.selectbox(
            "Options",
            ["AAPL", "MSFT"]
            )
        return selected_stocks
    
    def avaliable_dates_for_predictions(self) -> date:
        st.write("# Select a date to do the prediction/s from")
        # Define the date range
        min_date = date(2022, 1, 1)
        default_date = date(2024, 10, 1)

        selected_date = st.date_input(
            "Avaliable dates",
            value=default_date,  # Default to today’s date
            min_value=min_date,  # Set the minimum selectable date
            max_value=date.today()   # Set the maximum selectable date
            )
        return selected_date
        
    def display_num_of_predictions(self) -> int:
        st.write("# Chose number of predictions")
        num_of_predictions = st.slider("Options", min_value=1, max_value=5, step=1, value = 5)
        return num_of_predictions
    
    def avaliable_num_neurons(self) -> int:
        st.write("# Chose number of neurons in the first and second hidden layer of the MLP")
        first_layer_number = st.number_input(
            "Select nummber of neurons in the first layer:",
            min_value=4,
            max_value=16,
            value=8,
            step=1
            )
        
        second_layer_number = st.number_input(
            "Select nummber of neurons in the second layer:",
            min_value=0,
            max_value=16,
            value=0,
            step=1
            )

        return first_layer_number, second_layer_number

    def avaliable_learning_rates(self):
        st.write("# Chose a learning rate")
        learning_rate = st.number_input(
            "Select a learning rate:",
            min_value= 0.0005,
            max_value=0.0100,
            value=0.0035,
            step=0.0001,
            format="%.4f"
            )
        return learning_rate
    
    def avaliable_batch_size(self):
        st.write("# Chose a batch size")
        batch_size = st.slider("Select a batch size", 1, 5, step = 1, value = 3)
        return batch_size

class ControllerStockPredictions:
    def __init__(self):
        super().__init__()
        self.ui_manager = UserInterfaceStockPredictions()

    def run(self):
        """
        Main loop to run the application.
        """
        self.ui_manager.render_sidebar()

        if self.ui_manager.action == "Information":
            self._handle_information()

        if self.ui_manager.action == "Setup the model":
            self._handle_model_setup()

            if self._forcast_factory is not None:
                self._chose_button()
            
    def _handle_get_forcast(self):
        fig = self._forcast_factory.plot_predictions()
        st.pyplot(fig)

    def _handle_comparison(self):
        # Compare the predictions with observations
        mse = self._forcast_factory.compare_predictions_with_observations()
        fig = self._forcast_factory.plot_comparison()
        st.pyplot(fig)
        st.write("# Mean Squared Error")
        st.write("The MSE of the predictons compared to the actual"
                 f" closing prices is: {round(mse, 2)} $")

    def _handle_model_setup(self):
        self._chose_stock()

        if self._stock is not None:
            self._chose_date()

        if self._date is not None:
            self._chose_num_predictions()

        if self._num_predictions is not None:
            self._chose_architecture()
            
        if self._architecture is not None:
            self._chose_learning_rate()

        if self._learning_rate is not None:
            self._chose_batch_size()

        if self._batch_size is not None:
            self._build_forcast_factory()

    def _handle_information(self):
        self.ui_manager.display_info()

    def _chose_button(self):
        st.write("# Chose an Action")
        col1, col2 = st.columns(2)
        comparison_button = False

        with col1:
            forcast_button = self.ui_manager.button("Get a forcast")
        end_date_of_prediction = self._date + timedelta(days=self._num_predictions + 1)
        if end_date_of_prediction < date.today():
            with col2:
                comparison_button = self.ui_manager.button("Get a comparison")
        
        if forcast_button:
            self._handle_get_forcast()

        if comparison_button:
            self._handle_comparison()

    def _build_forcast_factory(self):
        initializer = ForcastFactoryInitializer()

        model_parameters = initializer.generate_model_parameters(
            self._architecture,
            self._learning_rate,
            batch_size = self._batch_size
            )
        
        # Generate datafacotry parameters
        datafactory_parameters = initializer.generate_datafactory_parameters()
        
        self._forcast_factory = ForcastFactory(self._stock, model_parameters, datafactory_parameters)

        self._forcast_factory.predict(
            self._num_predictions,
            end_date = self._date_str
        )

    def _chose_stock(self):
        self._stock = self.ui_manager.show_avaliable_stocks()

    def _chose_date(self):
        self._date = self.ui_manager.avaliable_dates_for_predictions()
        self._date_str = self._date.strftime("%Y-%m-%d")

    def _chose_num_predictions(self):
        self._num_predictions = self.ui_manager.display_num_of_predictions()

    def _chose_architecture(self):
        first_layer, second_layer = self.ui_manager.avaliable_num_neurons()
        if second_layer == 0:
            self._architecture = [first_layer]
        else:
            self._architecture = [first_layer, second_layer]

    def _chose_learning_rate(self):
        self._learning_rate = self.ui_manager.avaliable_learning_rates()

    def _chose_batch_size(self):
        self._batch_size = self.ui_manager.avaliable_batch_size()

if __name__ == "__main__":
    controller = ControllerStockPredictions()
    controller.run()