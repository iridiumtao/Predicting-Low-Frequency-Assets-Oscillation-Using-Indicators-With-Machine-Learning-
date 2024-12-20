import pandas as pd

class ReportGenerator:
    """
    Handles the generation and storage of reports.
    """
    def __init__(self, top_10_results: pd.DataFrame, results_df: pd.DataFrame, prediction_results: pd.DataFrame):
        """
        Constructor
            Args:
                top_10_results (pd.DataFrame): top 10 results data frame.
                results_df (pd.DataFrame): data frame with results.
                prediction_results (pd.DataFrame): dataframe with prediction results.
        """
        self.top_10_results = top_10_results
        self.results_df = results_df
        self.prediction_results = prediction_results

    # def save_results_to_google_sheets(self, credentials):
    #      # todo: not implemented
    #      """
    #      Saves the results to Google Sheets.
    #          Args:
    #              credentials: credentials for accessing the google sheets.
    #      """
    #      gc = gspread.authorize(credentials)
    #      results_sheet = gc.open("trading_results")
    #      top_10_sheet = gc.open("top_10_results")
    #
    #      results_worksheet = results_sheet.sheet1
    #      top_10_worksheet = top_10_sheet.sheet1
    #
    #      results_worksheet.update([self.results_df.columns.values.tolist()] + self.results_df.fillna(-1).values.tolist())
    #      top_10_worksheet.update([self.top_10_results.columns.values.tolist()] + self.top_10_results.fillna(-1).values.tolist())

    def save_results_to_csv(self):
        """
         Saves the results to csv files.
         """
        self.top_10_results.to_csv('top_10_results.csv', index=False)
        self.results_df.to_csv('all_results.csv', index=False)