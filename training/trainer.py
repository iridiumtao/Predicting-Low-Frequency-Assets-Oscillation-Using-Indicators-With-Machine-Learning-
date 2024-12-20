import pandas as pd
import os
from typing import List
from data.data_handler import DataHandler
from models.model import Model


class Trainer:
    """
    Handles the training of machine learning models.
    """
    def __init__(self, data_handler: DataHandler, model: Model, model_type: str, save_dir = "models/saved_models"):
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
        self.results_df = pd.DataFrame(columns=['Stock Symbol', 'Best Model', 'MAE', 'Loss', 'Precision', 'Recall'])
        self.top_10_results = pd.DataFrame(columns=['Stock Symbol', 'Best Model', 'MAE', 'Loss', 'Precision', 'Recall'])
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok = True) # Create the directory if it does not exist.


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

            # print(f"Shape of X_train: {X_train.shape}")
            # print(f"Shape of y_train: {y_train.shape}")
            # print(f"Type of X_train: {type(X_train)}")
            # print(f"Type of y_train: {type(y_train)}")
            # print(f"First 5 rows of y_train: {y_train[:5]}")
            # print(f"First 5 rows of X_train: {X_train[0, :, :5]}")  # Showing the first five features.
            #
            # print(f"Shape of X_test: {X_test.shape}")
            # print(f"Shape of y_test: {y_test.shape}")
            # print(f"Type of X_test: {type(X_test)}")
            # print(f"Type of y_test: {type(y_test)}")
            # print(f"First 5 rows of y_test: {y_test[:5]}")
            # print(f"First 5 rows of X_test: {X_test[0, :, :5]}")

            # Define the path to save the weights.
            model_path = os.path.join(self.save_dir, f"{stock_symbol}_model.weights.h5")
             # Load weights if exists, otherwise train the model
            if os.path.exists(model_path):
               print(f"Loading weights from {model_path} for {stock_symbol}")
               if self.model.model is None:
                  input_shape = X_train.shape[1:]
                  self.model.model = self.model._build_model(input_shape)
               self.model.model.load_weights(model_path)
            else:
              print(f"Training the model for {stock_symbol}")
              self.model.train_model(X_train, y_train, X_test=X_test, y_test=y_test)
              # Save weights after training.
              self.model.model.save_weights(model_path)

            if self.model_type == 'lstm':
               loss, mae = self.model.evaluate_model(X_test, y_test)
               print(f'MAE {mae} for {stock_symbol} loss {loss}')
               result = {
                'Stock Symbol': stock_symbol,
                'Best Model': self.model.model_type,
                'MAE': mae,
                'Loss': loss,
                'Precision': None,
                'Recall': None
                }
               result_df = pd.DataFrame([result])
               self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)
               self.top_10_results = pd.concat([self.top_10_results, result_df], ignore_index=True)
            else:
               accuracy, precision, recall, f1 = self.model.evaluate_model(X_test, y_test)
               print(f'Accuracy {accuracy}, f1 {f1} for {stock_symbol}')
               result = {
                'Stock Symbol': stock_symbol,
                'Best Model': self.model.model_type,
                'MAE': None,
                'Loss': None,
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
        self.top_10_results = self.top_10_results.sort_values(by='MAE', ascending=True).head(10)