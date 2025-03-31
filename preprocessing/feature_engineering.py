import pandas as pd
import numpy as np
import ta
from typing import List, Dict, Union, Optional

def extract_features(df:pd.DataFrame, ohlcv_cols:Dict[str, str] = None) -> pd.DataFrame:
    """
    Extract technical indicators and features from price data.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        ohlcv_cols (Dict[str, str]): Mapping of OHLCV column names
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    result = df.copy()
    if ohlcv_cols is None:
        ohlcv_cols = {"open":"Open",
                      "high":"High",
                      "low":"Low",
                      "close":"Close",
                      "volume":"Volume"}
    for col_types, col_name in ohlcv_cols.items():
        if col_name not in result.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
        
    # Add momentum indicators
    result = ta.add_momentum_ta(
        result,
        high = ohlcv_cols["high"],
        low = ohlcv_cols["low"],
        close = ohlcv_cols["close"],
        volume = ohlcv_cols["volume"],
        filling = True
    )
    # Add trend indicators
    result = ta.add_trend_ta(
        result,
        high = ohlcv_cols["high"],
        low = ohlcv_cols["low"],
        close = ohlcv_cols["close"],
        filling = True
    )
    # Volume indicators
    result = ta.add_volume_ta(
        result, 
        high = ohlcv_cols["high"],
        low = ohlcv_cols["low"],
        close = ohlcv_cols["close"],
        volume = ohlcv_cols["volume"],
        filling = True
    )
    # Add custom price-based features
    result["returns"] = result[ohlcv_cols["close"]].pct_change()
    result["log_returns"] = np.log(result[ohlcv_cols["close"]]) - np.log(result[ohlcv_cols["close"]].shift(1))
    # Add rolling statistics
    for window in [5,10,20,50]:
        result[f"return_rolling_mean_{window}"] = result["returns"].rolling(window = window).mean()
        result[f"return_rolling_std_{window}"] = result["returns"].rolling(window = window).std()
        result[f"volume_rolling_mean_{window}"] = result[ohlcv_cols["volume"]].rolling(window = window).mean()
    return result

def normalize_multi_modal(data_dict:Dict[str, pd.DataFrame],
                          time_col:str = "Date",
                          aligment_method:str = "inner") -> pd.DataFrame:
    """
    Align and normalize data from multiple sources (price, sentiment, etc.)
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary with DataFrames from different sources
        time_col (str): Column name for time/date used for alignment
        alignment_method (str): Join method ('inner', 'outer', 'left', 'right')
        
    Returns:
        pd.DataFrame: Combined and normalized DataFrame
    """
    if not data_dict:
        raise ValueError("data_dict cannot be empty")
    
    for source, df in data_dict.items():
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in {source} data")
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            data_dict[source][time_col] = pd.to_datetime(df[time_col])
    sources = list(data_dict.keys())
    result = data_dict[sources[0]].copy()
    result = result.set_index(time_col)
    # Merge with other DataFrames
    for source in sources[1:]:
        df = data_dict[source].copy()
        # Add prefix to prevent column name conflicts
        df.columns = [f"{source}_{col}" if col != time_col else col for col in df.columns]
        df = df.set_index(time_col)
        result = result.join(df, how=aligment_method)
    result = result.reset_index()

    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].fillna(method = "ffill").fillna(method="bfill")
            
    return result