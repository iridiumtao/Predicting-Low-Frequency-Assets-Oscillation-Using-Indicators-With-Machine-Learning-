import tensorflow as tf

def is_gpu_available():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs are available.")
        print(gpus)
        return True
    else:
        print("GPUs are not available.")
        return False

if __name__ == '__main__':
    print("hello world")
    is_gpu_available()


import yfinance as yf
import plotly.graph_objects as go
import stockstats
from stockstats import StockDataFrame
import pandas_ta as ta
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from google.colab import auth
from google.auth import default
import gspread
from typing import List







class TradingSystem:
    """
    Orchestrates the entire trading process, from data fetching to making predictions.
    """

    def __init__(self, data_handler: DataHandler, model: Model, model_type: str):
        self.data_handler = data_handler
        self.model = model
        self.model_type = model_type
        self.results_df = pd.DataFrame(columns=['Stock Symbol', 'Best Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall'])
        self.top_10_results = pd.DataFrame(columns=['Stock Symbol', 'Best Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall'])
        self.prediction_results = pd.DataFrame(columns=['Stock Symbol', 'Prediction'])

    def train_and_evaluate_all(self, add_custom_indicator: bool = False, target_type: str = 'binary_classification', correlated_asset:str = "SPY"):
        """Train and evaluate models for all stock symbols."""

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


    def predict_for_tickers(self, prediction_date: str, add_custom_indicator: bool = False, correlated_asset:str = "SPY"):
        """Make predictions for a given list of tickers using trained model."""
        for ticker in self.data_handler.tickers:
            X_pred = self.data_handler.prepare_prediction_data(ticker, prediction_date, add_custom_indicator=add_custom_indicator, correlated_asset=correlated_asset)

            if X_pred is not None:
              prediction = self.model.predict(X_pred)
              last_prediction = prediction[-1]
              results_row = {'Stock Symbol': ticker, 'Prediction': last_prediction}
              self.prediction_results = self.prediction_results.append(results_row, ignore_index=True)

    def display_results(self):
        """Displays the training and prediction results."""
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        display(self.top_10_results)
        display(self.results_df)
        display(self.prediction_results)

    def save_results_to_google_sheets(self, credentials):
        """Saves the results to Google Sheets."""
        gc = gspread.authorize(credentials)
        results_sheet = gc.open("trading_results")
        top_10_sheet = gc.open("top_10_results")

        results_worksheet = results_sheet.sheet1
        top_10_worksheet = top_10_sheet.sheet1

        results_worksheet.update([self.results_df.columns.values.tolist()] + self.results_df.fillna(-1).values.tolist())
        top_10_worksheet.update([self.top_10_results.columns.values.tolist()] + self.top_10_results.fillna(-1).values.tolist())

    def save_results_to_csv(self):
       self.top_10_results.to_csv('top_10_results.csv', index=False)
       self.results_df.to_csv('all_results.csv', index=False)

# --- Main Execution ---
if __name__ == '__main__':
    # Define Parameters
    start_date = "2022-05-21"
    end_date = "2023-12-13"
    prediction_date = "2023-12-22"

    tickers = [
        "AEFES", "AGHOL", "AHGAZ", "AKBNK", "AKCNS", "AKFGY", "AKFYE", "AKSA", "AKSEN", "ALARK",
        "ALBRK", "ALFAS", "ARCLK", "ASELS", "BERA", "BIENY", "BIMAS", "BRSAN", "BRYAT",
        "BUCIM", "CANTE", "CCOLA", "CIMSA", "CWENE", "DOAS", "DOHOL", "ECILC", "ECZYT", "EGEEN",
        "EKGYO", "ENJSA", "ENKAI", "EREGL", "EUPWR", "FROTO", "GARAN", "GENIL", "GESAN",
        "GLYHO", "GUBRF", "GWIND", "HALKB", "HEKTS", "IPEKE", "ISCTR", "ISDMR", "ISGYO",
        "ISMEN", "IZMDC", "KARSN", "KAYSE", "KCHOL", "KMPUR", "KONTR", "KONYA", "KORDS",
        "KOZAA", "KOZAL", "KRDMD", "KZBGY", "MAVI", "MGROS", "MIATK", "ODAS", "OTKAR", "OYAKC",
        "PENTA", "PETKM", "PGSUS", "PSGYO", "QUAGR", "SAHOL", "SASA", "SISE", "SKBNK", "SMRTG",
        "SNGYO", "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO", "TSKB", "TTKOM", "TTRAK",
        "TUKAS", "TUPRS", "ULKER", "VAKBN", "VESBE", "VESTL", "YKBNK", "YYLGD", "ZOREN"
    ]
    tickers = [symbol + ".IS" for symbol in tickers]
    tickers_for_prediction = ["AKFYE.IS", "EUPWR.IS", "YYLGD.IS", "BIMAS.IS", "GENIL.IS", "ODAS.IS", "ISMEN.IS", "ZOREN.IS", "CANTE.IS", "AHGAZ.IS"]

    # Initialize the DataHandler
    data_handler = DataHandler(start_date=start_date, end_date=end_date, tickers=tickers)

    # Choose the Model (lstm, random_forest, lightgbm or catboost)
    model = Model(model_type='lstm', output_type='regression')

    # Initialize the Trading System
    trading_system = TradingSystem(data_handler=data_handler, model=model, model_type = 'lstm')

    # Train and Evaluate the Model
    trading_system.train_and_evaluate_all(add_custom_indicator=True, target_type='regression', correlated_asset='SPY')

    # Make Prediction
    trading_system.data_handler.tickers = tickers_for_prediction
    trading_system.predict_for_tickers(prediction_date, add_custom_indicator=True, correlated_asset='SPY')

    # Display the Results
    trading_system.display_results()
    #Save the results in google sheets
    auth.authenticate_user()
    creds, _ = default()
    trading_system.save_results_to_google_sheets(credentials=creds)
    #Save the results in csv files
    trading_system.save_results_to_csv()