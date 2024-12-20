import pandas as pd
from typing import List
from data.data_handler import DataHandler
from models.model import Model


class Trainer:
    """
    Handles the training of machine learning models.
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
        self.results_df = pd.DataFrame(columns=['Stock Symbol', 'Best Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall'])
        self.top_10_results = pd.DataFrame(columns=['Stock Symbol', 'Best Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall'])


    def train_and_evaluate_all(self, add_custom_indicator: bool = False, target_type: str = 'binary_classification', correlated_asset:str = "SPY"):
        """
        Trains and evaluates models for all stock symbols.
            Args:
                add_custom_indicator (bool, optional): Whether to add custom indicators. Defaults to False.
                target_type (str, optional): Choose between 'binary_classification' or 'regression'. Defaults to 'binary_classification'.
                correlated_asset (str, optional): Ticker name of the correlated asset. Defaults to "SPY".
        """

        for stock_symbol in self.data_handler.tickers:
            X_train, X_test, y_train, y_test = self.data_handler.prepare_training_data(stock_symbol, add_custom_indicator=add_custom_indicator, target_type = target_type, correlated_asset = correlated_asset)
            if X_train is None or X_test is None:
                continue
            self.model.train_model(X_train, y_train, X_test=X_test, y_test=y_test)
            if self.model_type == 'lstm':
               loss, accuracy = self.model.evaluate_model(X_test, y_test)
               print(f'Accuracy {accuracy} for {stock_symbol} loss {loss}')
            else:
               accuracy, precision, recall, f1 = self.model.evaluate_model(X_test, y_test)
               print(f'Accuracy {accuracy}, f1 {f1} for {stock_symbol}')
               result = {
                'Stock Symbol': stock_symbol,
                'Best Model': self.model.model_type,
                'F1 Score': f1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall
                }
               result_df = pd.DataFrame([result])
               self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)
               self.top_10_results = pd.concat([self.top_10_results, result_df], ignore_index=True)
            # Sort results_df alphabetically
            self.results_df = self.results_df.sort_values(by='Stock Symbol')

        # Sort the top 10 results by F1 score
        self.top_10_results = self.top_10_results.sort_values(by='F1 Score', ascending=False).head(10)