import json
import requests
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, List

def ingest_data(source:str, headers:Optional[Dict[str, str]] = None, timeout:int=30,
                max_retries:int=3, retry_delay:int=5) -> pd.DataFrame:
    """
    Ingest data from a specified source (File or API).

    Args:
        source (str): Path to a file (CSV or JSON) or a URL to an API endpoint.
        headers (Optional[Dict[str, str]]): Optional headers for API requests (e.g., Authorization).
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for failed API requests.
        retry_delay (int): Delay between retries in seconds.

    Returns:
        pd.DataFrame: The ingested data as a DataFrame.

    Raises:
        ValueError: If the source is unsupported or if there's an unrecoverable error fetching from the API.
        FileNotFoundError: If a local file source is not found.
        requests.exceptions.RequestException: For network-related API errors after retries.
    """
    if source.lower().endswith(".csv"):
        try:
            return pd.read_csv(source)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {source}")
        except Exception as e:
            raise ValueError(f"Error reading CSV {source}: {e}")
    elif source.lower().endswith(".json"):
        try:
            return pd.read_json(source, orient="records")
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at: {source}")
        except Exception as e:
            raise ValueError(f"Error reading JSON {source}: {e}")
    elif source.startswith("http"):
        attempts = 0
        while attempts < max_retries:
            try:
                response = requests.get(source, headers = headers, timeout = timeout)
                response.raise_for_status() # Rauses HTTPError for bad response
                # Handle potential JSON decoding errors
                try:
                    data = response.json()
                    # Check if the response is a list of records (common for APIs)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    # Cases where data might be nested e.g.{'results':[...]}
                    elif isinstance(data, dict) and len(data) == 1:
                        key = list(data.keys())[0]
                        if isinstance(data[key], list):
                            print(f"Warning: Assuming data is under key '{key}'")
                            return pd.DataFrame(data[key])
                        else:
                            # Attempt to load dict directly
                            return pd.DataFrame([data])
                    else:
                        # Attempt to load dict directly
                        print("Warning: API response is not a list of records. Attempting direct DataFrame conversion")
                        return pd.DataFrame([data]) # Wrap dict in a list
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to decode JSON from {source}. Response test: {response.text[:200]}...")
                except Exception as e:
                    raise ValueError(f"Error structuring data from {source} into DataFrame: {e}")
            except requests.exceptions.Timeout:
                attempts += 1
                print(f"Request timed out for {source}. Retrying ({attempts}/{max_retries})...")
                if attempts >= max_retries:
                    raise requests.exceptions.RequestException(f"Request timed out after {max_retries} attempts for {source}")
                time.sleep(retry_delay)
            except requests.exceptions.HTTPError as e:
                # Rate limit handling
                if e.response.status_code == 429:
                    print(f"Rate limit hit for {source}. Consider increasing delay or reducing request frequency")
                attempts += 1
                print(f"HTTP Error: {e.response.status_code} for {source}. Retrying ({attempts}/{max_retries})...")
                if attempts >= max_retries:
                    raise requests.exceptions.RequestException(f"HTTP Error {e.response.status_code} after {max_retries} attempts for {source}:{e}")
                time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                attempts += 1
                print(f"Request failed for {source}:{e}. Retrying ({attempts}/{max_retries})...")
                if attempts >= max_retries:
                    raise requests.exceptions.RequestException(f"Failed to fetch data from {source} after {max_retries} attempts: {e}")
                time.sleep(retry_delay)
    else:
        raise ValueError(f"Unsupported source format:{source}. Please use .csv, .json or an http(s) URL")
    
def fetch_market_data(ticker:str, start_date:str, end_date:str, interval:str="id") -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance API.

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d', '1wk', '1mo')

    Returns:
        pd.DataFrame: DataFrame with market data, index reset.

    Raises:
        ImportError: If yfinance is not installed.
        ValueError: If dates are invalid or data fetching fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance library is required. Install with 'pip install yfinance'")
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except ValueError:
        raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.")
    try:
        print(f"Fetching {ticker} data from {start_date} to {end_date} ({interval})...")
        data = yf.download(ticker, start = start_date, end=end_date, interval = interval,
                        progress = False)
        if data.empty:
            print(f"Warning: No data returned for {ticker} in the specified range/interval")
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        data.reset_index(inplace= True)
        # Ensure date column is in standard Datetime
        data["Date"] = pd.to_datetime(data["Date"])
        print(f"Successfully fetched (len{data}) rows for {ticker}")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker} using yfinance: {str(e)}")
        
def fetch_sentiment_data(source_type:str, query:str, api_key:Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Fetch sentiment data from various social media and news sources.

    Args:
        source_type (str): Source name ('twitter', 'reddit', 'newsapi').
        query (str): Search query (e.g., stock ticker '$AAPL', company name).
        api_key (Optional[str]): API key if required by the source.
        **kwargs: Additional parameters specific to the source API:
            - twitter: consumer_key, consumer_secret, access_token, access_token_secret, count (default 100)
            - reddit: client_id, client_secret, user_agent, limit (default 100), subreddit (default "wallstreetbets,stocks,investing")
            - newsapi: from_date, to_date, language (default 'en'), sort_by (default 'publishedAt')

    Returns:
        pd.DataFrame: DataFrame with sentiment data (timestamp, text, score, source).

    Raises:
        ValueError: For invalid source_type or missing credentials.
        ImportError: If required packages are not installed.
        Exception: For API-specific errors.
    """
    source_type = source_type.lower()
    
    if source_type == 'twitter':
        return _fetch_twitter_data(query, api_key, **kwargs)
    elif source_type == 'reddit':
        return _fetch_reddit_data(query, **kwargs)
    elif source_type == 'newsapi':
        return _fetch_news_data(query, api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported source type: {source_type}. Supported types: 'twitter', 'reddit', 'newsapi'")

def _fetch_twitter_data(query: str, api_key: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Fetch data from Twitter using Tweepy."""
    try:
        import tweepy
    except ImportError:
        raise ImportError("Tweepy package is required. Install with 'pip install tweepy'")
    
    # Extract credentials
    consumer_key = kwargs.get('consumer_key')
    consumer_secret = kwargs.get('consumer_secret')
    access_token = kwargs.get('access_token')
    access_token_secret = kwargs.get('access_token_secret')
    count = kwargs.get('count', 100)
    
    # Validate credentials
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        raise ValueError("Twitter API requires consumer_key, consumer_secret, access_token, and access_token_secret")
    
    try:
        # Authenticate to Twitter
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Search tweets
        tweets = []
        for tweet in tweepy.Cursor(api.search_tweets, q=query, tweet_mode='extended', lang='en').items(count):
            tweets.append({
                'id': tweet.id_str,
                'created_at': tweet.created_at,
                'text': tweet.full_text if hasattr(tweet, 'full_text') else tweet.text,
                'user': tweet.user.screen_name,
                'retweet_count': tweet.retweet_count,
                'favorite_count': tweet.favorite_count,
                'source': 'twitter'
            })
        
        if not tweets:
            print(f"No tweets found for query: '{query}'")
            return pd.DataFrame(columns=['created_at', 'text', 'user', 'retweet_count', 'favorite_count', 'source', 'id'])
        
        df = pd.DataFrame(tweets)
        print(f"Successfully fetched {len(df)} tweets for '{query}'")
        return df
        
    except tweepy.TweepyException as e:
        raise Exception(f"Twitter API error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error fetching Twitter data: {str(e)}")

def _fetch_reddit_data(query: str, **kwargs) -> pd.DataFrame:
    """Fetch data from Reddit using PRAW."""
    try:
        import praw
    except ImportError:
        raise ImportError("PRAW package is required. Install with 'pip install praw'")
    
    # Extract parameters
    client_id = kwargs.get('client_id')
    client_secret = kwargs.get('client_secret')
    user_agent = kwargs.get('user_agent')
    limit = kwargs.get('limit', 100)
    subreddit_names = kwargs.get('subreddit', 'wallstreetbets,stocks,investing')
    
    # Validate credentials
    if not all([client_id, client_secret, user_agent]):
        raise ValueError("Reddit API requires client_id, client_secret, and user_agent")
    
    try:
        # Initialize the Reddit API client
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        posts = []
        subreddits = subreddit_names.split(',')
        
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name.strip())
            
            # Search for submissions
            for submission in subreddit.search(query, limit=limit//len(subreddits)):
                posts.append({
                    'id': submission.id,
                    'created_at': pd.to_datetime(submission.created_utc, unit='s'),
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'comments': submission.num_comments,
                    'url': submission.url,
                    'subreddit': subreddit_name,
                    'source': 'reddit'
                })
        
        if not posts:
            print(f"No Reddit posts found for query: '{query}' in subreddits: {subreddit_names}")
            return pd.DataFrame(columns=['created_at', 'title', 'text', 'score', 'comments', 'url', 'subreddit', 'source', 'id'])
        
        df = pd.DataFrame(posts)
        print(f"Successfully fetched {len(df)} Reddit posts for '{query}'")
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching Reddit data: {str(e)}")

def _fetch_news_data(query: str, api_key: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Fetch data from News API."""
    try:
        from newsapi import NewsApiClient
    except ImportError:
        raise ImportError("NewsAPI package is required. Install with 'pip install newsapi-python'")
    
    if not api_key:
        raise ValueError("NewsAPI requires an API key")
    
    # Extract parameters
    from_date = kwargs.get('from_date')
    to_date = kwargs.get('to_date')
    language = kwargs.get('language', 'en')
    sort_by = kwargs.get('sort_by', 'publishedAt')
    
    try:
        # Initialize the News API client
        newsapi = NewsApiClient(api_key=api_key)
        
        # Prepare parameters
        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
        }
        
        # Add optional date parameters if provided
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
            
        # Fetch articles
        response = newsapi.get_everything(**params)
        
        if response['status'] != 'ok':
            raise Exception(f"NewsAPI error: {response.get('message', 'Unknown error')}")
        
        articles = response['articles']
        
        if not articles:
            print(f"No news articles found for query: '{query}'")
            return pd.DataFrame(columns=['published_at', 'title', 'description', 'content', 'url', 'source_name', 'source'])
        
        # Transform to DataFrame
        articles_data = []
        for article in articles:
            articles_data.append({
                'published_at': pd.to_datetime(article['publishedAt']),
                'title': article['title'],
                'description': article['description'],
                'content': article['content'],
                'url': article['url'],
                'source_name': article['source']['name'],
                'source': 'newsapi'
            })
        
        df = pd.DataFrame(articles_data)
        print(f"Successfully fetched {len(df)} news articles for '{query}'")
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching NewsAPI data: {str(e)}")

def generate_synthetic_data(gan_model:Any, num_samples:int, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate synthetic data using a pre-trained GAN model.

    Args:
        gan_model: A trained GAN model object with a `generate` method.
                   The method should take num_samples and return a numpy array or DataFrame.
        num_samples (int): The number of synthetic samples to generate.
        columns (Optional[List[str]]): Optional list of column names for the generated data.

    Returns:
        pd.DataFrame: Generated synthetic data as a DataFrame.

    Raises:
        ValueError: If the GAN model is invalid or its output is not convertible to a DataFrame.
        AttributeError: If the gan_model lacks a 'generate' method.
    """
    if not hasattr(gan_model, "generate"):
        raise AttributeError("Provided GAN model must have a 'generate' method")
    try:
        synthetic_data = gan_model.generate(num_samples)
    except Exception as e:
        raise ValueError(f"Error during GAN model generation: {e}")
    if isinstance(synthetic_data, pd.DataFrame):
        return synthetic_data
    elif isinstance(synthetic_data, np.ndarray):
        if columns and len(columns) == synthetic_data.shape[1]:
            col_names = columns
        else:
            if columns:
                print(f"Warning: Provided column count ({len(columns)}) doesn't match generated data columns ({synthetic_data.shape[1]}). Using default names.")
            col_names = [f"feature_{i}" for i in range(synthetic_data.shape[1])]
        return pd.DataFrame(synthetic_data, columns = col_names)
    else:
        raise ValueError("GAN model 'generate' method must return a numpy array or a pandas DataFrame")
