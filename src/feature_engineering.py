import pandas as pd

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced interaction features."""
    df = df.copy()
    df["OverallQual_x_TotalSF"] = df["OverallQual"] * (df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0))
    df["TotalRooms"] = df[["BedroomAbvGr", "FullBath", "HalfBath", "KitchenAbvGr"]].sum(axis=1)
    return df

