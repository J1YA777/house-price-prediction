import pandas as pd
from pathlib import Path

def load_data(data_dir="data"):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test

def get_X_y(train_df, target="SalePrice"):
    y = train_df[target].copy()
    X = train_df.drop(columns=[target])
    return X, y
