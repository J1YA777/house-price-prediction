import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add some engineered features for house price prediction."""
    df = df.copy()
    
    # Total square footage
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    
    # Age features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    
    # Boolean features
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    df["HasBasement"] = (~df["BsmtFinSF1"].isna()).astype(int)
    
    # Interaction features
    df["OverallQual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    
    # Total rooms
    df["TotalRooms"] = df[["BedroomAbvGr", "FullBath", "HalfBath", "KitchenAbvGr"]].sum(axis=1)
    
    return df
