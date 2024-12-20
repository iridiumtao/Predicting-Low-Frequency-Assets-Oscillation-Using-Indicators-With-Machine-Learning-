from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


class Model:
    """
    Defines and trains machine learning models.
    """

    def __init__(self, model_type: str = 'lstm', output_type: str = 'binary_classification'):
        self.model_type = model_type
        self.output_type = output_type
        self.model = self._build_model()
        self.best_params = {}

    def _build_model(self) -> Sequential:
        """Builds the LSTM model based on the output type."""
        if self.model_type == 'lstm':
            regressor = Sequential()
            regressor.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(1, None)))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
            regressor.add(Dropout(0.3))
            regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
            regressor.add(Dropout(0.4))
            regressor.add(LSTM(units=120, activation='relu'))
            regressor.add(Dropout(0.5))

            if self.output_type == 'binary_classification':
                regressor.add(Dense(units=1, activation='sigmoid'))
                regressor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            elif self.output_type == 'regression':
                regressor.add(Dense(units=1))
                regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) # or 'huber_loss'
            else:
                raise ValueError("Invalid output type.")

            return regressor
        elif self.model_type == 'random_forest':
             return None
        elif self.model_type == 'lightgbm':
             return None
        elif self.model_type == 'catboost':
             return None
        else:
            raise ValueError("Invalid model type.")

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32, X_test: np.ndarray = None, y_test: np.ndarray = None):
        """Trains the specified model."""
        if self.model_type == 'lstm':
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=[X_test, y_test] if (X_test is not None and y_test is not None) else None, verbose=0)
        elif self.model_type == 'random_forest':
             self._optimize_hyperparameters(X_train, y_train, X_test, y_test, self.objective_rf)
             final_model = RandomForestClassifier(**self.best_params, random_state=42)
             final_model.fit(X_train, y_train)
             self.model = final_model
        elif self.model_type == 'lightgbm':
            self._optimize_hyperparameters(X_train, y_train, X_test, y_test, self.objective_lgb)
            final_model = lgb.LGBMClassifier(**self.best_params, random_state=42)
            final_model.fit(X_train, y_train)
            self.model = final_model
        elif self.model_type == 'catboost':
             self._optimize_hyperparameters(X_train, y_train, X_test, y_test, self.objective_catboost)
             final_model = CatBoostClassifier(**self.best_params, random_state=42, silent=True)
             final_model.fit(X_train, y_train)
             self.model = final_model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluates the model and returns accuracy and loss metrics."""
        if self.model_type == 'lstm':
             loss, metric = self.model.evaluate(X_test, y_test)
             return loss, metric
        elif self.model_type == 'random_forest':
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            return accuracy, precision, recall, f1
        elif self.model_type == 'lightgbm':
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            return accuracy, precision, recall, f1
        elif self.model_type == 'catboost':
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            return accuracy, precision, recall, f1
        else:
            return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained model."""
        if self.model_type == 'lstm':
            preds = self.model.predict(X)
            return preds
        elif self.model_type in ['random_forest', 'lightgbm', 'catboost']:
           return self.model.predict(X)
        else:
            return None

    def objective_rf(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        return model

    def objective_lgb(self, trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-9, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-9, 10.0),
        }
        model = lgb.LGBMClassifier(**params, random_state=42)
        return model

    def objective_catboost(self, trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            'depth': trial.suggest_int('depth', 2, 12),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-9, 10.0),
            'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.1, 10.0),
            'border_count': trial.suggest_int('border_count', 1, 255),
            'eval_metric': 'Accuracy',
        }
        model = CatBoostClassifier(**params, random_state=42, silent=True)
        return model

    def _optimize_hyperparameters(self, X_train, y_train, X_test, y_test, objective_func, n_trials:int = 100):
        """Optimize hyperparameters for given model type."""
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._optuna_objective(trial, objective_func, X_train, y_train, X_test, y_test), n_trials=n_trials)
        self.best_params = study.best_params
        print(f"Best Hyperparameters:", self.best_params)

    def _optuna_objective(self, trial, objective_func, X_train, y_train, X_test, y_test):
            model = objective_func(trial)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy