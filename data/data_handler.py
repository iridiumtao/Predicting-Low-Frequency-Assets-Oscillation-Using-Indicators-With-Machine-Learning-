class DataHandler:
    """
    Handles fetching, preprocessing, and preparing data for modeling.
    """

    def __init__(self, start_date: str, end_date: str, tickers: List[str]):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.stock_data = self._fetch_stock_data()
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = MinMaxScaler()

    def _fetch_stock_data(self) -> pd.DataFrame:
        """Fetches stock data for specified tickers."""
        all_stock_data = {}
        for ticker in self.tickers:
            try:
              all_stock_data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)
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
            correlated_asset_prices = correlated_asset_data['Close'].values
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
           df['direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
           target_column = 'direction'
        elif target_type == 'regression':
            df['target'] = df['Close'].shift(-1) - df['Close']
            target_column = 'target'
        else:
            raise ValueError("Invalid target type.")

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