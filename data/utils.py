import os
import pandas as pd

def create_directory(directory_path: str):
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def check_if_file_exists(file_path: str) -> bool:
    """Checks if a file exists."""
    return os.path.exists(file_path)

def save_data_to_csv(df: pd.DataFrame, file_path: str):
    """Saves DataFrame to a CSV file."""
    df.to_csv(file_path)

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """Loads DataFrame from a CSV file, specifying date format."""
    df = pd.read_csv(file_path)
    # df.rename(columns={'Unnamed: 0': 'Date',
    #                    'Unnamed: 1': 'Open',
    #                    'Unnamed: 2': 'High',
    #                    'Unnamed: 3': 'Low',
    #                    'Unnamed: 4': 'Close',
    #                    'Unnamed: 5': 'Volume'
    #                    }, inplace=True)
    # df.set_index('Date', inplace=True)
    # df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return df

if __name__ == '__main__':
    df = load_data_from_csv('../models/data/raw_data/MSFT.csv')
    print(df)