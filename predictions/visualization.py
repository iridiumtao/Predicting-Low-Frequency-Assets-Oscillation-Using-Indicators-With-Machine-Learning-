import pandas as pd
from IPython.display import display

def display_results(results_df, top_10_results, prediction_results):
    """
    Displays the training and prediction results.
        Args:
          results_df (pd.DataFrame): Dataframe with the results.
          top_10_results (pd.DataFrame): Top 10 results dataframe.
          prediction_results (pd.DataFrame): Dataframe with the predictions.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    display(top_10_results)
    display(results_df)
    display(prediction_results)