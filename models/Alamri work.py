import os
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dropout, Dense, Input
from keras.api.models import Model
from keras.api.optimizers import Adam
from joblib import dump, load
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
from data.utils import create_directory, check_if_file_exists, load_data_from_csv, save_data_to_csv
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.regularizers import l2


pd.set_option('future.no_silent_downcasting', True)


class StockPredictor:
    def __init__(self, stock_ticker, period="5y", data_dir: str = 'data/raw_data'):
        self.stock_ticker = stock_ticker
        self.period = period
        self.data = None
        self.data_dir = data_dir
        create_directory(self.data_dir)  # Create directory if it does not exist
        self.classification_model = None
        self.regression_model = None
        self.lstm_model = None
        self.scaler = None

    def fetch_historical_data(self):
        file_path = os.path.join(self.data_dir, f'{self.stock_ticker}.csv')
        if check_if_file_exists(file_path):
            print(f"Loading cached data for {self.stock_ticker} from {file_path}")
            self.data = load_data_from_csv(file_path)
            return self.data
        else:
            try:
                print(f"Downloading data for {self.stock_ticker}")
                stock = yf.Ticker(self.stock_ticker)
                self.data = stock.history(period=self.period)
                if self.data.empty:
                    raise ValueError(f"No data found for {self.stock_ticker}")
                save_data_to_csv(self.data, file_path)
                return self.data
            except Exception as e:
                raise Exception(f"Error fetching data for {self.stock_ticker}: {str(e)}")

    def calculate_technical_indicators(self):
        try:
            bb = BollingerBands(close=self.data['Close'], window=20, window_dev=2)
            self.data['BB_upper'] = bb.bollinger_hband()
            self.data['BB_lower'] = bb.bollinger_lband()
            self.data['BB_width'] = self.data['BB_upper'] - self.data['BB_lower']

            rsi = RSIIndicator(close=self.data['Close'], window=14)
            self.data['RSI'] = rsi.rsi()

            for window in [20, 50, 200]:
                self.data[f'{window}_SMA'] = self.data['Close'].rolling(window=window).mean()
                self.data[f'{window}_EMA'] = self.data['Close'].ewm(span=window).mean()

            self.data['Daily_Return'] = self.data['Close'].pct_change()
            self.data['Return_Volatility'] = self.data['Daily_Return'].rolling(window=20).std()

            self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']

            self.data['ADX'] = self.calculate_adx()

            return self.data
        except Exception as e:
            raise Exception(f"Error calculating technical indicators: {str(e)}")

    def calculate_adx(self, period=14):
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()

        plus_dm = high - high.shift()
        minus_dm = low.shift() - low
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def prepare_features(self):
        self.data['Direction'] = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)
        self.data['Next_Close'] = self.data['Close'].shift(-1)

        features = [
            'Close', 'RSI', 'BB_width', 'Volume_Ratio', 'Return_Volatility',
            '20_SMA', '50_SMA', '200_SMA', '20_EMA', '50_EMA', '200_EMA',
            'ADX', 'Daily_Return'
        ]

        self.data = self.data.dropna()
        return self.data, features

    def train_models_random_forest(self, features):
        """
        Train Random Forest models for classification and regression.

        Parameters:
        features (list): List of feature names to be used for training the models.

        Returns:
        tuple: Trained classification and regression models.
        """
        X = self.data[features]
        y_class = self.data['Direction']
        y_reg = self.data['Next_Close']

        split_idx = int(len(self.data) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_class_train = y_class[:split_idx]
        y_class_test = y_class[split_idx:]
        y_reg_train = y_reg[:split_idx]
        y_reg_test = y_reg[split_idx:]

        self.classification_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.classification_model.fit(X_train, y_class_train)

        self.regression_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.regression_model.fit(X_train, y_reg_train)

        class_pred = self.classification_model.predict(X_test)
        reg_pred = self.regression_model.predict(X_test)

        print("\nModel Performance Metrics (Random Forest):")
        print(f"Classification Accuracy: {accuracy_score(y_class_test, class_pred):.3f}")
        print(f"Regression MAE: ${mean_absolute_error(y_reg_test, reg_pred):.2f}")
        print(f"Regression RMSE: ${np.sqrt(mean_squared_error(y_reg_test, reg_pred)):.2f}")

        return self.classification_model, self.regression_model

    def create_time_series_data(self, features, window_size=10):
        """Creates time series data with a sliding window."""
        X, y_class, y_reg = [], [], []
        if len(self.data) < window_size:
            return np.array([]), np.array([]), np.array([])  # Return empty arrays if not enough data
        for i in range(len(self.data) - window_size):
            X.append(self.data[features].iloc[i:i + window_size].values)
            y_class.append(
                (self.data['Next_Close'].iloc[i + window_size] > self.data['Close'].iloc[i + window_size]).astype(int))
            y_reg.append(self.data['Next_Close'].iloc[i + window_size])
        return np.array(X), np.array(y_class), np.array(y_reg)

    def train_models_LSTM(self, features, window_size=10):
        model_path = os.path.join(self.data_dir, f"{self.stock_ticker}_model.weights.h5")
        scaler_path = os.path.join(self.data_dir, f"{self.stock_ticker}_scaler.joblib")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Loading model and scaler from {self.data_dir} for {self.stock_ticker}")
            self.lstm_model = self._build_model(features, window_size)
            self.lstm_model.load_weights(model_path)
            self.scaler = load(scaler_path)
        else:
            print(f"Training the model for {self.stock_ticker}")
            self._train_models_LSTM(features, window_size)

    def _train_models_LSTM(self, features, window_size=10):
        print("Training Multi-Task LSTM Model")
        X, y_class, y_reg = self.create_time_series_data(features, window_size)

        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.3, random_state=42)

        X_train_scaled = []
        scaler = MinMaxScaler()
        for series in X_train:
            scaled_series = scaler.fit_transform(series)
            X_train_scaled.append(scaled_series)
        X_train_scaled = np.array(X_train_scaled)

        X_test_scaled = []
        for series in X_test:
            scaled_series = scaler.transform(series)
            X_test_scaled.append(scaled_series)
        X_test_scaled = np.array(X_test_scaled)

        self.scaler = scaler  # save scaler
        # build model
        model = self._build_model(features, window_size)

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'classification': 'binary_crossentropy',
                'regression': 'mean_squared_error'
            },
            metrics={
                'classification': ['accuracy'],
                'regression': ['mse']
            }
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)  # 加入 Learning rate scheduler
        # train
        model.fit(
            X_train_scaled,
            {'classification': y_class_train, 'regression': y_reg_train},
            validation_data=(X_test_scaled, {'classification': y_class_test, 'regression': y_reg_test}),
            epochs=50,
            batch_size=32,
            verbose=1,
            callbacks=[reduce_lr]
        )

        # model evaluation
        results = model.evaluate(X_test_scaled, {'classification': y_class_test, 'regression': y_reg_test})
        print("Evaluation Results:", results)

        self.lstm_model = model
        model_path = os.path.join(self.data_dir, f"{self.stock_ticker}_model.weights.h5")
        scaler_path = os.path.join(self.data_dir, f"{self.stock_ticker}_scaler.joblib")
        self.lstm_model.save_weights(model_path)
        dump(self.scaler, scaler_path)

    def _build_model(self, features, window_size):
        X, _, _ = self.create_time_series_data(features, window_size)  # get data shape
        inputs = Input(shape=(X.shape[1], X.shape[2]))

        x = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001))(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        x = LSTM(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)

        classification_output = Dense(1, activation='sigmoid', name='classification')(x)

        regression_output = Dense(1, name='regression')(x)

        model = Model(inputs=inputs, outputs=[classification_output, regression_output])
        print(model.summary())
        return model

    def predict_next_day_LSTM(self, features, window_size=10, price_uncertainty_multiplier=0.02):
        latest_data = self.data[features].iloc[-window_size:].values
        latest_scaled = self.scaler.transform(latest_data)
        latest_reshaped = np.expand_dims(latest_scaled, axis=0)  # Make sure it is 3d

        classification_pred, regression_pred = self.lstm_model.predict(latest_reshaped)

        predicted_direction = "Up" if classification_pred[0][0] > 0.5 else "Down"
        confidence = classification_pred[0][0] * 100 if predicted_direction == "Up" else (1 - classification_pred[0][0]) * 100

        # inverse normalization
        predicted_price_scaled = regression_pred[0][0] # Get scaled price
        dummy_array = np.zeros_like(latest_data[0]) # Create a dummy data with the same shape as your input
        dummy_array[-1] = predicted_price_scaled  # insert your predicted price
        dummy_array = np.expand_dims(dummy_array, axis=0)
        predicted_price = self.scaler.inverse_transform(dummy_array)[0][-1] # Invert the scaled data.


        price_uncertainty = abs(predicted_price) * price_uncertainty_multiplier
        price_range = (predicted_price - price_uncertainty, predicted_price + price_uncertainty)

        return {
            'direction': predicted_direction,
            'direction_confidence': confidence,
            'predicted_price': predicted_price,
            'price_range': price_range
        }

    def predict_next_day_rf(self, features):
        latest_data = self.data[features].iloc[-1:]

        predicted_direction = None
        predicted_price = 0
        confidence = 0
        price_uncertainty = 0

        if self.classification_model is not None:
            direction_prob = self.classification_model.predict_proba(latest_data)[0]
            predicted_direction = "Up" if direction_prob[1] > 0.5 else "Down"
            confidence = max(direction_prob) * 100

        if self.regression_model is not None:
            predicted_price = self.regression_model.predict(latest_data)[0]

            feature_importance = self.regression_model.feature_importances_
            residuals = self.regression_model.predict(self.data[features]) - self.data['Next_Close']
            std_residuals = np.std(residuals)
            price_uncertainty = std_residuals * 2 # 2 standard deviations

        return {
            'direction': predicted_direction,
            'direction_confidence': confidence,
            'predicted_price': predicted_price,
            'price_range': (predicted_price - price_uncertainty, predicted_price + price_uncertainty)
        }

class DCFValuation:
    def __init__(self, ticker, data_dir: str = 'data/raw_data'):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data_dir = data_dir
        create_directory(self.data_dir)  # Create directory if it does not exist

    def fetch_financial_data(self):
        file_path = os.path.join(self.data_dir, f'{self.ticker}_financial_data.pkl')

        # load from cache
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print("Loaded cached DCF data.")
                return cached_data
            except Exception as e:
                print(f"Error loading cached DCF data: {str(e)}")


        try:
            cf = self.stock.cash_flow
            bs = self.stock.balance_sheet
            is_ = self.stock.income_stmt

            # Save data to cache
            data = (cf, bs, is_)
            try:
                file_path = os.path.join(self.data_dir, f'{self.ticker}_financial_data.pkl')
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                print("Saved DCF data to cache.")
            except Exception as e:
                print(f"Error saving cached DCF data: {str(e)}")

            return cf, bs, is_
        except Exception as e:
            raise Exception(f"Error fetching financial data: {str(e)}")

    def calculate_fcf_historical(self, cf, bs):
        try:
            operating_cash_flow = cf.loc['Operating Cash Flow']
            capex = cf.loc['Capital Expenditure'].abs()
            fcf = operating_cash_flow - capex
            return fcf
        except Exception as e:
            raise Exception(f"Error calculating historical FCF: {str(e)}")

    def calculate_growth_rates(self, fcf_historical):
        growth_rates = fcf_historical.ffill().pct_change()  # Corrected line
        avg_growth_rate = growth_rates.mean()

        max_growth = 0.15
        min_growth = 0.02

        growth_rate = max(min(avg_growth_rate, max_growth), min_growth)
        terminal_growth = 0.02

        return growth_rate, terminal_growth
    def calculate_wacc(self, is_, bs):
        try:
            risk_free_rate = 0.0425
            market_risk_premium = 0.06
            beta = self.stock.info.get('beta', 1.0)
            cost_of_equity = risk_free_rate + beta * market_risk_premium

            interest_expense = abs(is_.loc['Interest Expense'].iloc[0])
            total_debt = bs.loc['Total Debt'].iloc[0]
            if total_debt > 0:
                cost_of_debt = interest_expense / total_debt
            else:
                cost_of_debt = 0

            tax_rate = is_.loc['Tax Rate For Calcs'].iloc[0]

            market_cap = self.stock.info.get('marketCap', 0)
            total_capital = market_cap + total_debt

            if total_capital > 0:
                equity_weight = market_cap / total_capital
                debt_weight = total_debt / total_capital
            else:
                equity_weight = 1
                debt_weight = 0

            wacc = (equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate))

            return max(wacc, 0.08)
        except (KeyError, IndexError) as e: # Catch KeyError for missing index and IndexError for empty series
            print(f"Error calculating WACC: {str(e)}. Using default WACC of 10%. Check financial data.")
            return 0.1  # Return a default WACC if there's an error

    def project_fcf(self, last_fcf, growth_rate, terminal_growth, periods=5):
        fcf_projections = []
        current_fcf = last_fcf

        for _ in range(periods):
            current_fcf *= (1 + growth_rate)
            fcf_projections.append(current_fcf)

        terminal_value = (fcf_projections[-1] * (1 + terminal_growth) / (self.wacc - terminal_growth))

        return fcf_projections, terminal_value

    def calculate_intrinsic_value(self):
        try:
            cf, bs, is_ = self.fetch_financial_data()

            historical_fcf = self.calculate_fcf_historical(cf, bs)

            growth_rate, terminal_growth = self.calculate_growth_rates(historical_fcf)

            self.wacc = self.calculate_wacc(is_, bs)

            last_fcf = historical_fcf.iloc[0]
            fcf_projections, terminal_value = self.project_fcf(last_fcf, growth_rate, terminal_growth)

            pv_fcf = sum([
                fcf / ((1 + self.wacc) ** (i + 1))
                for i, fcf in enumerate(fcf_projections)
            ])

            pv_terminal = terminal_value / ((1 + self.wacc) ** len(fcf_projections))

            total_value = pv_fcf + pv_terminal

            shares_outstanding = self.stock.info.get('sharesOutstanding', 0)

            if shares_outstanding > 0:
                intrinsic_value = total_value / shares_outstanding
            else:
                raise ValueError("Could not get shares outstanding")

            current_price = self.stock.info.get('currentPrice', 0)

            return {
                'intrinsic_value': intrinsic_value,
                'current_price': current_price,
                'upside': ((intrinsic_value - current_price) / current_price * 100),
                'wacc': self.wacc * 100,
                'growth_rate': growth_rate * 100,
                'terminal_growth': terminal_growth * 100
            }

        except Exception as e:
            raise Exception(f"Error calculating intrinsic value: {str(e)}")

def main(ticker):
    # try:
    predictor = StockPredictor(ticker)
    predictor.fetch_historical_data()
    predictor.calculate_technical_indicators()
    data, features = predictor.prepare_features()
    # predictor.train_models_random_forest(features)
    predictor.train_models_LSTM(features)
    predictions = predictor.predict_next_day_LSTM(features)
    # predictions = predictor.predict_next_day_rf(features)

    print(f"\nTechnical Analysis Predictions for {ticker}:")
    print(f"Direction: {predictions['direction']} (Confidence: {predictions['direction_confidence']:.1f}%)")
    print(f"Predicted Price: ${predictions['predicted_price']:.2f}")
    print(f"Price Range: ${predictions['price_range'][0]:.2f} - ${predictions['price_range'][1]:.2f}")

    print("\nCalculating DCF Valuation...")
    dcf = DCFValuation(ticker)
    valuation = dcf.calculate_intrinsic_value()

    print(f"\nDCF Valuation Results:")
    print(f"Intrinsic Value: ${valuation['intrinsic_value']:.2f}")
    print(f"Current Price: ${valuation['current_price']:.2f}")
    print(f"Potential Upside: {valuation['upside']:.1f}%")
    print(f"WACC: {valuation['wacc']:.1f}%")
    print(f"Growth Rate: {valuation['growth_rate']:.1f}%")
    print(f"Terminal Growth: {valuation['terminal_growth']:.1f}%")

    # except Exception as e:
    #     print(f"Error in analysis pipeline: {str(e)}")

if __name__ == "__main__":
    main("MSFT")