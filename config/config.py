from typing import List

class Config:
    """
    Configuration class to hold project settings.
    """
    def __init__(self, start_date: str, end_date: str, prediction_date: str, tickers: List[str], tickers_for_prediction: List[str]):
        """
        Constructor method
            Args:
                start_date (str): start date for getting the data.
                end_date (str): end date for getting the data.
                prediction_date (str): prediction date.
                tickers (List[str]): tickers to be trained on.
                tickers_for_prediction (List[str]): tickers to be predicted.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_date = prediction_date
        self.tickers = tickers
        self.tickers_for_prediction = tickers_for_prediction