import pandas as pd

def csv_as_df(file_path):
    df = pd.read_csv(file_path)
    return df