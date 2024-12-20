from data.data_handler import DataHandler
from models.model import Model
import pandas as pd


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