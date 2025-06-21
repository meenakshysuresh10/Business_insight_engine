import pandas as pd

def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df
