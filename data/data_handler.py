import yfinance as yf
import pandas as pd
import os
import numpy as np
from typing import List
from stockstats import StockDataFrame
import pandas_ta as ta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from data.utils import create_directory, check_if_file_exists, load_data_from_csv, save_data_to_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class DataHandler:
    """
    Handles fetching, preprocessing, and preparing data for modeling.
    """

    def __init__(self, start_date: str, end_date: str, tickers: List[str], data_dir: str = 'data/raw_data'):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.data_dir = data_dir
        create_directory(self.data_dir) # Create directory if it does not exist
        self.stock_data = self._fetch_stock_data()
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = MinMaxScaler()


    def _fetch_stock_data(self) -> pd.DataFrame:
        """Fetches stock data for specified tickers, using cache if available."""
        all_stock_data = {}
        for ticker in self.tickers:
            file_path = os.path.join(self.data_dir, f'{ticker}.csv')
            if check_if_file_exists(file_path):
              print(f"Loading cached data for {ticker} from {file_path}")
              # todo: warning: only loading 1000 rows for now
              all_stock_data[ticker] = load_data_from_csv(file_path)[:1000]
            else:
                try:
                    print(f"Downloading data for {ticker}")
                    stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)
                    if not stock_data.empty:
                        save_data_to_csv(stock_data, file_path)
                        all_stock_data[ticker] = stock_data
                    else:
                       print(f"No data received for {ticker}")
                except Exception as e:
                  print(f"Error getting the data for the ticker {ticker}, {e}")
        return all_stock_data

    def create_linear_regression_indicator(self, stock_df:pd.DataFrame, correlated_asset_prices):
        model = LinearRegression()
        X = np.array(correlated_asset_prices).reshape(-1, 1)
        y = stock_df['Close'].values
        model.fit(X, y)
        X_pred = np.array(correlated_asset_prices[-1]).reshape(1, -1)
        predicted_price = model.predict(X_pred)
        stock_df['custom_indicator_lr'] = predicted_price
        return stock_df

    def extract_features(self, stock_data: pd.DataFrame, add_custom_indicator: bool = False, correlated_asset:str = "SPY") -> pd.DataFrame:
        """Extracts features (technical indicators) from the stock data.

        Args:
            stock_data (pd.DataFrame): Stock data.
            add_custom_indicator (bool, optional): Whether to add the custom linear regression indicator. Defaults to False.
            correlated_asset (str, optional): Ticker name of the correlated asset. Defaults to "SPY".
        """
        stock_df = StockDataFrame.retype(stock_data.copy())
        # Add custom indicator only if specified
        if add_custom_indicator:
            correlated_asset_data = yf.download(correlated_asset, start = self.start_date, end = self.end_date)
            correlated_asset_prices = correlated_asset_data['close'].values
            stock_df = self.create_linear_regression_indicator(stock_df, correlated_asset_prices)
        return stock_df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the data by imputing missing values and scaling."""
        imputed_data = self.imputer.fit_transform(df)
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
        return imputed_df

    def create_target(self, df: pd.DataFrame, target_type: str = 'binary_classification') -> pd.DataFrame:
        """Creates a target variable for binary classification or regression.
        Args:
            df (pd.DataFrame): Input DataFrame.
            target_type (str, optional): Choose between 'binary_classification' or 'regression'. Defaults to 'binary_classification'.
        """
        if target_type == 'binary_classification':
           df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
           target_column = 'direction'
        elif target_type == 'regression':
            df['target'] = df['close'].shift(-1) - df['close']
            target_column = 'target'
        else:
            raise ValueError("Invalid target type.")
        df = df.dropna(subset = [target_column]) # Remove any data with NaN values at the target variable.
        print(df.head(10))
        return df, target_column

    def prepare_training_data(self, stock_symbol: str, test_size: float = 0.2, add_custom_indicator: bool = False, target_type: str = 'binary_classification', correlated_asset:str = "SPY"):
        """Prepares the training and test data sets.
        Args:
            stock_symbol (str): Stock symbol to be used.
            test_size (float, optional): Size of the test set. Defaults to 0.2.
            add_custom_indicator (bool, optional): Whether to add custom indicators. Defaults to False.
            target_type (str, optional): Choose between 'binary_classification' or 'regression'. Defaults to 'binary_classification'.
            correlated_asset (str, optional): Ticker name of the correlated asset. Defaults to "SPY".
        """
        try:
            stock_data = self.stock_data[stock_symbol]
        except Exception as e:
            print(f"Error {e} with the ticker {stock_symbol}")
            return None, None, None, None

        features_data = self.extract_features(stock_data, add_custom_indicator, correlated_asset)
        features_data = self.preprocess_data(features_data)
        features_data, target_column = self.create_target(features_data, target_type=target_type)

        X = features_data.drop(target_column, axis=1)
        y = features_data[target_column]

        # todo: random state is FIXED for now
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_reshaped = np.expand_dims(X_train_scaled, axis=1)
        X_test_reshaped = np.expand_dims(X_test_scaled, axis=1)

        return X_train_reshaped, X_test_reshaped, y_train, y_test

    def prepare_prediction_data(self, stock_symbol:str, prediction_date: str, add_custom_indicator: bool = False, correlated_asset:str = "SPY"):
         """Prepare data to make a prediction.
         Args:
            stock_symbol (str): Stock symbol to be used.
            prediction_date (str): Prediction end date.
            add_custom_indicator (bool, optional): Whether to add custom indicators. Defaults to False.
            correlated_asset (str, optional): Ticker name of the correlated asset. Defaults to "SPY".
         """

         try:
            stock_data = yf.download(stock_symbol, start=self.start_date, end=prediction_date)
         except Exception as e:
             print(f"Error {e} with the ticker {stock_symbol}")
             return None
         features_data = self.extract_features(stock_data, add_custom_indicator, correlated_asset)
         features_data = self.preprocess_data(features_data)
         X = features_data.drop(['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1) # Drop these column since they are not used in training
         X_scaled = self.scaler.transform(X)
         X_reshaped = np.expand_dims(X_scaled, axis=1)
         return X_reshaped