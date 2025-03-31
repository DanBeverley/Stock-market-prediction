import json
import requests
import pandas as pd
import numpy as np

def ingest_data(source:str) -> pd.DataFrame:
    """
    Ingest data from a specified source.

    Args:
        source (str): Path to a file (CSV or JSON) or a URL to an API endpoint.

    Returns:
        pd.DataFrame: The ingested data as a DataFrame.

    Raises:
        ValueError: If the source is unsupported or if there's an error fetching from the API.
    """
    if source.endswith(".csv"):
        return pd.read_csv(source)
    elif source.endswith(".json"):
        return pd.read_json(source)
    elif source.startswith("http"):
        response = requests.get(source)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Failed to fetch data from {source}: Status code {response.status_code}")
    else:
        raise ValueError(f"Unsupported source: {source}")

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by handling missing values and outliers.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace = True)
        else:
            df[col].fillna(df[col].mode()[0], inplace = True)
    # Handle outliers using IQR method
    for col in df.select_dtypes(include = ["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def generate_synthetic_data(gan_model, num_samples:int) -> np.ndarray:
    """
    Generate synthetic data using a GAN model.

    Args:
        gan_model: A GAN model with a `generate` method that takes the number of samples and returns a numpy array or DataFrame.
        num_samples (int): The number of synthetic samples to generate.

    Returns:
        np.ndarray: Generated synthetic data

    Raises:
        ValueError: If the GAN model's generate method does not return a numpy array or DataFrame.
    """
    if not hasattr(gan_model, "generate"):
        raise ValueError("Provided GAN model must have 'generate' method")
    
    synthetic_data = gan_model.generate(num_samples)

    if isinstance(synthetic_data, np.ndarray):
        # Create DataFrame with default column names
        return pd.DataFrame(synthetic_data, columns=[f"feature_{i}" for i in range(synthetic_data.shape[1])])
    elif isinstance(synthetic_data, pd.DataFrame):
        return synthetic_data
    else:
        raise ValueError("GAN model generate method must return a numpy array or a pandas DataFrame")

def fetch_market_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance API.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d', '1wk', '1mo')
        
    Returns:
        pd.DataFrame: DataFrame with market data
    """
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        data.reset_index(inplace=True)
        return data
    except ImportError:
        raise ImportError("yfinance library is required. Install with 'pip install yfinance'")
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

def fetch_sentiment_data(source: str, ticker: str = None) -> pd.DataFrame:
    """
    Fetch sentiment data from various sources like Twitter, Reddit, etc.
    
    Args:
        source (str): Source name ('twitter', 'reddit', 'news')
        ticker (str): Stock ticker symbol to filter results
        
    Returns:
        pd.DataFrame: DataFrame with sentiment data
    """
    if source == "twitter":
        pass
    elif source == "reddit":
        pass
    elif source == "news":
        pass
    
    import random
    dates = pd.date_range(start="2023-01-01", end="2023-01-10")
    sentiment_data = {
        'date': dates,
        'sentiment_score': [random.uniform(-1, 1) for _ in range(len(dates))],
        'volume': [random.randint(100, 10000) for _ in range(len(dates))]
    }
    return pd.DataFrame(sentiment_data)