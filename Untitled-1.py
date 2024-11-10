from pricepredictor.main import forcast_closing_prices

if __name__ == "__main__":
    predictions, mse = forcast_closing_prices(number_of_predictions=5, end_date="2024-10-01", architecture=[8], learning_rate=0.0035, batch_size=3)