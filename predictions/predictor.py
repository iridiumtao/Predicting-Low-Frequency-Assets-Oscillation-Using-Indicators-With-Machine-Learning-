import pandas as pd
from typing import List
import numpy as np
from data.data_handler import DataHandler
from models.model import Model


class Predictor:
    """
    Handles the prediction using trained machine learning models.
    """
    def __init__(self, data_handler: DataHandler, model: Model, model_type: str):
        """
        Constructor
            Args:
                 data_handler (DataHandler): Data handler class.
                 model (Model): Model class.
                 model_type (str): type of model.
        """
        self.data_handler = data_handler
        self.model = model
        self.model_type = model_type
        self.prediction_results = pd.DataFrame(columns=['Stock Symbol', 'Prediction'])


    def predict_for_tickers(self, prediction_date: str, add_custom_indicator: bool = False, correlated_asset:str = "SPY"):
        """
        Make predictions for a given list of tickers using trained model.
            Args:
                prediction_date (str): date for prediction.
                add_custom_indicator (bool, optional): Whether to add custom indicators. Defaults to False.
                correlated_asset (str, optional): Ticker name of the correlated asset. Defaults to "SPY".
        """
        for ticker in self.data_handler.tickers:
            X_pred = self.data_handler.prepare_prediction_data(ticker, prediction_date, add_custom_indicator=add_custom_indicator, correlated_asset=correlated_asset)

            if X_pred is not None:
                prediction = self.model.predict(X_pred)
                if self.model_type == 'lstm':
                    last_prediction = prediction[-1][0]
                else:
                   last_prediction = prediction[-1]
                results_row = {'Stock Symbol': ticker, 'Prediction': last_prediction}
                self.prediction_results = self.prediction_results.append(results_row, ignore_index=True)