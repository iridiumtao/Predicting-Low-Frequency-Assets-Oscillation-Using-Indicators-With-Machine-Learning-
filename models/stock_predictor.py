# Source Code References
# 1. Stock Price Prediction Using Transformers
#    Author: Mattafrank
#    Date: May 5, 2024
#    URL: https://medium.com/@Matthew_Frank/stock-price-prediction-using-transformers-2d84341ff213
#
# 2. Stock Price Prediction with Machine Learning Models
#    Author: Neslişah Çelek
#    Date: Jan 31, 2024
#    URL: https://github.com/neslisahcelek/algorithmic-trading-with-ml/blob/main/src/model_utils.py
#
# 3. Feature selection and deep neural networks for stock price direction forecasting using technical analysis indicators. Machine Learning with Applications.
#    Author: Peng, Y. H.
#    Date: June 8, 2021
#    URL: https://www.sciencedirect.com/science/article/pii/S266682702100030X 
#
# 4. Average Directional Index (ADX): Definition and Formula
#    Author: Cory Michell
#    Date: July 23, 2024
#    URL: https://www.investopedia.com/terms/a/adx.asp


import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from joblib import dump, load
from keras.api.callbacks import ModelCheckpoint
from keras.api.layers import LSTM, Dense, Dropout, Conv1D, Flatten, Input, MultiHeadAttention, LayerNormalization, Add, \
    GlobalAveragePooling1D, RepeatVector
from keras.api.models import Model, load_model
from keras.api.optimizers import Adam
from keras.api.saving import register_keras_serializable
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from data.utils import create_directory, check_if_file_exists, load_data_from_csv, save_data_to_csv

pd.set_option('future.no_silent_downcasting', True)


# This block was made by Alamri with some help of Ai
# This brings the stock ticker to be implemented
class StockPredictor:
    def __init__(self, stock_ticker, period="5y", data_dir: str = './data/raw_data', model_dir: str = './saved_models'):
        self.stock_ticker = stock_ticker
        self.period = period
        self.data = None
        self.data_dir = data_dir
        create_directory(self.data_dir)  # Create directory if it does not exist
        self.model_dir = model_dir
        create_directory(self.model_dir)  # Create directory if it does not exist
        self.classification_model = None
        self.regression_model = None
        self.lstm_model = None
        self.transformer_model = None
        self.scaler = None
        self.transformer_scaler: MinMaxScaler = None  # For Transformer specific scaling

    # This block was made by Alamri and Chun-Ju Tao with some help of Ai
    # basically what it does is get the historical data from the fiscal earnings reports
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

    # This block was made by Alamri with some help of Ai
    # This first refernce helped with how to implement it
    # Reference: Çelek (2024) [2]
    # This second reference helped with our choice of which indicators to choose given that there are thousands of them
    # I already have a background in the subject and so this research is to confirm my theory
    # Reference: Peng, Y. H. (2021) [3]
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

    # This block was made by Alamri with some help of Ai
    # This is the average directional index, the point of adding it is to have a guestemate on the direction
    # The Next day, The refrence is an explaination of what it is from investopedia
    # Reference: Mitchell (2024) [4]
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

    # This block was made by Alamri with some help of Ai
    # This is part of the stream line for most financial models to clean out the technical indicators used
    # Here as features
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

    # Reference: Mattafrank (2024) [1]
    def prepare_features_transformer(self, sequence_length=30):
        """
        Modified from Mattafrank's implementation [1].
        """
        self.data['Direction'] = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)
        self.data['Next_Close'] = self.data['Close'].shift(-1)

        features = [
            'Close', 'RSI', 'BB_width', 'Volume_Ratio', 'Return_Volatility',
            '20_SMA', '50_SMA', '200_SMA', '20_EMA', '50_EMA', '200_EMA',
            'ADX', 'Daily_Return'
        ]

        self.data = self.data.dropna()
        file_path = os.path.join(self.data_dir, f'{self.stock_ticker}_with_indicators.csv')
        save_data_to_csv(self.data, file_path)

        # Prepare sequences for Transformer
        feature_data = self.data[features].values
        target_data = self.data['Next_Close'].values.reshape(-1, 1)

        # Scale features and target separately for Transformer
        self.transformer_scaler = MinMaxScaler()
        feature_data_scaled = self.transformer_scaler.fit_transform(feature_data)

        X, y = [], []
        for i in range(len(self.data) - sequence_length - 1):
            X.append(feature_data_scaled[i:i + sequence_length])
            y.append(target_data[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        """
        End of Modified from Mattafrank's implementation [1].
        """

        return self.data, features, X, y

    # This block was made by Alamri with some help of Ai
    # We were taught this in this class so I am simply implementing
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

    def train_models_LSTM(self, features):
        """
        Decide to load trained model or train model, developed by Chun-Ju Tao.
        """
        model_path = os.path.join(self.model_dir, f"{self.stock_ticker}_cnn_lstm_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{self.stock_ticker}_cnn_lstm_scaler.joblib")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Loading LSTM model from {self.model_dir} for {self.stock_ticker}")
            self.lstm_model = load_model(model_path)
            self.scaler = load(scaler_path)
        else:
            print(f"Training the LSTM model for {self.stock_ticker}")
            self._train_models_LSTM(features)

    # Reference: Neslişah Çelek's (2024) [2]
    def _train_models_LSTM(self, features):
        print("Training Multi-Task LSTM Model")

        """
        Modified and copied from Neslişah Çelek's implementation [2]
        """
        X = self.data[features]
        y_class = (self.data['Next_Close'] > self.data['Close']).astype(int)
        y_reg = self.data['Next_Close']

        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.3, random_state=42)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_reshaped = np.expand_dims(X_train_scaled, axis=1)
        X_test_reshaped = np.expand_dims(X_test_scaled, axis=1)

        # build model
        model = self._build_model(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

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
        # train
        model.fit(
            X_train_reshaped,
            {'classification': y_class_train, 'regression': y_reg_train},
            validation_data=(X_test_reshaped, {'classification': y_class_test, 'regression': y_reg_test}),
            epochs=50,
            batch_size=32,
            verbose=1
        )

        # model evaluation
        results = model.evaluate(X_test_reshaped, {'classification': y_class_test, 'regression': y_reg_test})
        print("Evaluation Results:", results)

        """
        End of Modified and copied from Neslişah Çelek's implementation [2]
        """

        """
        Store model, developed by Chun-Ju Tao.
        """

        self.lstm_model = model
        model_path = os.path.join(self.model_dir, f"{self.stock_ticker}_cnn_lstm_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{self.stock_ticker}_cnn_lstm_scaler.joblib")
        self.lstm_model.save(model_path)
        self.scaler = scaler
        dump(self.scaler, scaler_path)

        """
        End of Store model, developed by Chun-Ju Tao.
        """

    # Reference: Neslişah Çelek (2024) [2]
    def _build_model(self, shape):
        """
        Build a CNN-LSTM model for stock price prediction.
        Modified from Neslişah Çelek's implementation [2]

        Parameters:
        shape (tuple): Shape of the input data.

        Returns:
        keras.Model: Compiled CNN-LSTM model.
        """
        """
        CNN layers, developed by Chun-Ju Tao.
        """
        inputs = Input(shape=shape)

        # CNN
        x = Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(inputs)
        x = Conv1D(filters=128, kernel_size=1, activation='relu', padding='same')(x)

        # Flatten CNN
        x = Flatten()(x)
        x = RepeatVector(shape[0])(x)

        """
        End of CNN layers, developed by Chun-Ju Tao.
        """

        """
        LSTM Layers and output layer, modified from Neslişah Çelek's implementation [2].
        """

        # LSTM
        x = LSTM(64, activation='relu', return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64, activation='relu')(x)
        x = Dropout(0.2)(x)

        # output layer
        classification_output = Dense(1, activation='sigmoid', name='classification')(x)
        regression_output = Dense(1, name='regression')(x)

        model = Model(inputs=inputs, outputs=[classification_output, regression_output])
        print(model.summary())

        """
        End of LSTM Layers and output layer, modified from Neslişah Çelek's implementation [2].
        """
        return model

    def train_models_transformer(self, X, y):
        """
        Decide to load trained model or train model, developed by Chun-Ju Tao.
        """
        model_path = os.path.join(self.model_dir, f"{self.stock_ticker}_transformer_model.keras")

        if os.path.exists(model_path):
            print(f"Loading Transformer model from {self.model_dir} for {self.stock_ticker}")
            self.transformer_model = load_model(model_path, custom_objects={'custom_mae_loss': self._custom_mae_loss,
                                                                            'dir_acc': self._dir_acc})
        else:
            print(f"Training the Transformer model for {self.stock_ticker}")
            self._train_transformer(X, y)

    # Reference: Mattafrank (2024) [1]
    def _train_transformer(self, all_sequences, all_labels):
        """
        Copied from Mattafrank's implementation [1].
        """
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(all_sequences))
        all_sequences = all_sequences[shuffled_indices]
        all_labels = all_labels[shuffled_indices]

        train_size = int(len(all_sequences) * 0.9)

        train_sequences = all_sequences[:train_size]
        train_labels = all_labels[:train_size]

        other_sequences = all_sequences[train_size:]
        other_labels = all_labels[train_size:]

        shuffled_indices = np.random.permutation(len(other_sequences))
        other_sequences = other_sequences[shuffled_indices]
        other_labels = other_labels[shuffled_indices]

        val_size = int(len(other_sequences) * 0.5)

        validation_sequences = other_sequences[:val_size]
        validation_labels = other_labels[:val_size]

        test_sequences = other_sequences[val_size:]
        test_labels = other_labels[val_size:]

        # Model parameters
        input_shape = train_sequences.shape[1:]
        head_size = 256
        num_heads = 16
        ff_dim = 1024
        num_layers = 12
        dropout = 0.20

        # Build the model
        self.transformer_model = self._build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers,
                                                               dropout)
        self.transformer_model.summary()

        # Compile the model
        optimizer = Adam()
        self.transformer_model.compile(optimizer=optimizer, loss=self._custom_mae_loss, metrics=[self._dir_acc])

        # Define callbacks
        checkpoint_callback_train = ModelCheckpoint(
            os.path.join(self.model_dir, f"{self.stock_ticker}_transformer_train_model.keras"),
            monitor="dir_acc",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        checkpoint_callback_val = ModelCheckpoint(
            os.path.join(self.model_dir, f"{self.stock_ticker}_transformer_val_model.keras"),
            monitor="val_dir_acc",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        BATCH_SIZE = 64
        EPOCHS = 50
        self.transformer_model.fit(train_sequences, train_labels,
                                   validation_data=(validation_sequences, validation_labels),
                                   epochs=EPOCHS,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   callbacks=[checkpoint_callback_train, checkpoint_callback_val,
                                              self._get_lr_callback(batch_size=BATCH_SIZE, epochs=EPOCHS)])

        # Save the entire model
        self.transformer_model.save(os.path.join(self.model_dir, f"{self.stock_ticker}_transformer_model.keras"))

        # Evaluate
        accuracy = self.transformer_model.evaluate(test_sequences, test_labels)[1]
        print(f"\nTransformer Model Accuracy on Test Data: {accuracy}")

        predictions = self.transformer_model.predict(test_sequences)
        r2 = r2_score(test_labels, predictions[:, 0])
        print(f"Transformer Model R-squared on Test Data: {r2}")

        """
        End of copied from Mattafrank's implementation [1].
        """

    # Reference: Mattafrank (2024) [1]
    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Copied from Mattafrank's implementation [1].
        """
        # Attention and Normalization
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Add()([x, inputs])

        # Feed Forward Part
        y = LayerNormalization(epsilon=1e-6)(x)
        y = Dense(ff_dim, activation="relu")(y)
        y = Dropout(dropout)(y)
        y = Dense(inputs.shape[-1])(y)
        return Add()([y, x])

        """
        End of copied from Mattafrank's implementation [1].
        """

    # Reference: Mattafrank (2024) [1]
    def _build_transformer_model(self, input_shape, head_size, num_heads, ff_dim, num_layers, dropout=0):
        """
        Copied from Mattafrank's implementation [1].
        """
        inputs = Input(shape=input_shape)
        x = inputs

        # Create multiple layers of the Transformer block
        for _ in range(num_layers):
            x = self._transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        # Final part of the model
        x = GlobalAveragePooling1D()(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        outputs = Dense(1, activation="linear")(x)  # Regression task

        # Build model
        model = Model(inputs=inputs, outputs=outputs)
        return model

        """
        End of copied from Mattafrank's implementation [1].
        """

    # Reference: Mattafrank (2024) [1]
    @register_keras_serializable()
    def _custom_mae_loss(self, y_true, y_pred):
        """
        Copied from Mattafrank's implementation [1].
        """
        y_true_next = tf.cast(y_true[:, 0], tf.float64)
        y_pred_next = tf.cast(y_pred[:, 0], tf.float64)
        abs_error = tf.abs(y_true_next - y_pred_next)
        return tf.reduce_mean(abs_error)
        """
        End of copied from Mattafrank's implementation [1].
        """

    # Reference: Mattafrank (2024) [1]
    @register_keras_serializable()
    def _dir_acc(self, y_true, y_pred):
        """
        Copied and modified from Mattafrank's implementation [1].
        """
        print(y_true.shape, y_true, type(y_true))
        # Get current closing prices
        current_close_prices = self.data['Close'].iloc[-(y_true.shape[1] + 1):-1].values
        current_close_tensor = tf.cast(current_close_prices, tf.float64)

        y_true_next = tf.cast(y_true[:, 0], tf.float64)
        y_pred_next = tf.cast(y_pred[:, 0], tf.float64)

        true_change = y_true_next - current_close_tensor
        pred_change = y_pred_next - current_close_tensor

        correct_direction = tf.equal(tf.sign(true_change), tf.sign(pred_change))
        return tf.reduce_mean(tf.cast(correct_direction, tf.float64))
        """
        End of copied and modified from Mattafrank's implementation [1].
        """

    # Reference: Mattafrank (2024) [1]
    def _get_lr_callback(self, batch_size=16, mode='cos', epochs=500, plot=False):
        """
        Copied from Mattafrank's implementation [1].
        """
        lr_start, lr_max, lr_min = 0.0001, 0.005, 0.00001
        lr_ramp_ep = int(0.30 * epochs)
        lr_sus_ep = max(0, int(0.10 * epochs) - lr_ramp_ep)

        def lrfn(epoch):
            if epoch < lr_ramp_ep:
                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            elif epoch < lr_ramp_ep + lr_sus_ep:
                lr = lr_max
            elif mode == 'cos':
                decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep, epoch - lr_ramp_ep - lr_sus_ep
                phase = math.pi * decay_epoch_index / decay_total_epochs
                lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
            else:
                lr = lr_min
            return lr

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Scheduler')
            plt.show()

        return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
        """
        End of copied from Mattafrank's implementation [1].
        """

    def predict_next_day_LSTM(self, features):
        """
        Modified from Alamri's predict_next_day_rf() by Chun-Ju Tao.
        """
        latest_data = self.data[features].iloc[-1:]
        latest_scaled = self.scaler.transform(latest_data)
        latest_reshaped = np.expand_dims(latest_scaled, axis=1)

        classification_pred, regression_pred = self.lstm_model.predict(latest_reshaped)

        predicted_direction = "Up" if classification_pred[0][0] > 0.5 else "Down"
        confidence = classification_pred[0][0] * 100 if predicted_direction == "Up" else (1 - classification_pred[0][
            0]) * 100

        predicted_price = regression_pred[0][0]
        residuals = regression_pred.flatten() - self.data['Next_Close']
        std_residuals = np.std(residuals)
        price_uncertainty = std_residuals * 2

        return {
            'direction': predicted_direction,
            'direction_confidence': confidence,
            'predicted_price': predicted_price,
            'price_range': (predicted_price - price_uncertainty, predicted_price + price_uncertainty)
        }
    # This block was made by Alamri with some help of Ai
    # This is the part where we get the results No refrences just trial and error and Ai
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
            price_uncertainty = std_residuals * 2  # 2 standard deviations

        return {
            'direction': predicted_direction,
            'direction_confidence': confidence,
            'predicted_price': predicted_price,
            'price_range': (predicted_price - price_uncertainty, predicted_price + price_uncertainty)
        }

    # Reference: Neslişah Çelek's [2], Mattafrank (2024) [1]
    def predict_next_day_transformer(self, sequence_length=30):
        """
        Copied and modified from Mattafrank's implementation [1].
        Copied and modified from Neslişah Çelek's implementation [2].
        """
        # Get the latest sequence of data
        features = [
            'Close', 'RSI', 'BB_width', 'Volume_Ratio', 'Return_Volatility',
            '20_SMA', '50_SMA', '200_SMA', '20_EMA', '50_EMA', '200_EMA',
            'ADX', 'Daily_Return'
        ]
        latest_data = self.data[features].iloc[-sequence_length:].values
        latest_scaled = self.transformer_scaler.transform(latest_data)
        latest_reshaped = np.expand_dims(latest_scaled, axis=0)  # Add batch dimension

        # Make prediction
        predicted_price_scaled = self.transformer_model.predict(latest_reshaped)[0][0]

        # The transformer predicts the 'Next_Close' which was directly scaled.
        # We need to inverse transform it. To do this, we need a dummy array
        # with the same number of features as what the scaler was trained on.
        dummy_array = np.zeros((1, len(features)))
        close_index = features.index('Close')
        dummy_array[0, close_index] = predicted_price_scaled
        predicted_price = self.transformer_scaler.inverse_transform(dummy_array)[0][close_index]

        # Calculate a rough price range (you might want a more sophisticated method)
        price_uncertainty = np.std(self.data['Close'][-sequence_length:])

        # Determine direction (this is a simplification, as the transformer directly predicts price)
        predicted_direction = "Up" if predicted_price > self.data['Close'].iloc[-1] else "Down"
        confidence = None  # Confidence is harder to derive directly from a regression transformer

        """
        End of copied and modified from Mattafrank's implementation [1].
        End of copied and modified from Neslişah Çelek's implementation [2].
        """

        return {
            'direction': predicted_direction,
            'direction_confidence': confidence,
            'predicted_price': predicted_price,
            'price_range': (predicted_price - price_uncertainty, predicted_price + price_uncertainty)
        }
