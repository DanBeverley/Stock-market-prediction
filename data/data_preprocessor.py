import pandas as pd
from typing import List, Callable, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, preprocessing_steps:List[Callable[[pd.DataFrame], pd.DataFrame]]=None,
                 feature_configs:Dict[str, Any]=None) -> None:
        """
        Initialize the DataPreprocessor.

        Args:
            preprocessing_steps (List[Callable[[pd.DataFrame], pd.DataFrame]], optional): 
                List of functions to apply to the data. Each function takes a DataFrame and returns a 
                processed DataFrame. Defaults to None.
            feature_configs (Dict[str, Any], optional): Configuration for specific feature transformations 
                (e.g., {'numerical': ['col1', 'col2'], 'categorical': ['col3']}). Defaults to None.
        """
        self.preprocessing_steps = preprocessing_steps or []
        self.feature_configs = feature_configs or {}
        self.transformers:Dict[str, Any] = {}
    
    def fit(self, data:pd.DataFrame) -> None:
        """
        Fit the preprocessor to the data, initializing transformers based on feature_configs.

        Args:
            data (pd.DataFrame): The input dataset to fit the preprocessor on.
        Raises:
            ValueError: If feature_configs references columns not in the data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        # Fit transformers based on feature configurations
        if "numerical" in self.feature_configs:
            scaler = StandardScaler()
            num_cols = self.feature_configs["numerical"]
            if not set(num_cols).issubset(data.columns):
                raise ValueError("Numerical column in feature_configs not found")
            self.transformers["scaler"] = scaler.fit(data[num_cols])
        
        if "categorical" in self.feature_configs:
            cat_cols = self.feature_configs["categorical"]
            if not set(cat_cols).issubset(data.columns):
                raise ValueError("Categorical columns in feature_configs not found")
            self.transformers["encoders"] = {
                col:LabelEncoder().fit(data[col]) for col in cat_cols
            }
    
    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps and fitted transformers to the data.

        Args:
            data (pd.DataFrame): The input dataset to transform.

        Returns:
            pd.DataFrame: The preprocessed dataset.

        Raises:
            ValueError: If transformers are not fitted or columns are missing.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        processed_data = data.copy()

        if "scaler" in self.transformers:
            num_cols = self.feature_configs["numerical"]
            if not set(num_cols).issubset(processed_data.columns):
                raise ValueError("Numerical columns missing in data for transformation")
            processed_data[num_cols] = self.transformers["scaler"].transform(processed_data[num_cols])
        
        if "encoders" in self.transformers:
            for col, encoder in self.transformers["encoders"].items():
                if col not in processed_data.columns:
                    raise ValueError(f"Categorical column '{col}' missing in data for transformation")
                processed_data[col] = encoder.transform(processed_data[col])
        
        for step in self.preprocessing_steps:
            processed_data = step(processed_data)

        return processed_data

    def fit_transform(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor to the data and then transform it.

        Args:
            data (pd.DataFrame): The input dataset to fit and transform.

        Returns:
            pd.DataFrame: The preprocessed dataset.

        Raises:
            ValueError: If input is invalid or feature_configs are misconfigured.
        """
        self.fit(data)
        return self.transform(data)
    
    def validate_data(self, data:pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the dataset and return a report of issues (e.g., missing values, data types).

        Args:
            data (pd.DataFrame): The input dataset to validate.

        Returns:
            Dict[str, Any]: A dictionary containing validation results (e.g., missing values per column).

        Raises:
            ValueError: If input is not a DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        validation_report = {
            "missing_values":data.isnull().sum().to_dict(),
            "data_types":data.dtypes.to_dict(),
            "shape":data.shape
        }
        return validation_report
    
    def add_preprocessing_step(self, step:Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """
        Add a preprocessing step to the pipeline.

        Args:
            step (Callable[[pd.DataFrame], pd.DataFrame]): A function that takes a DataFrame and 
                returns a processed DataFrame.
        """
        self.preprocessing_steps.append(step)
    
    def add_time_series_features(self, data:pd.DataFrame, date_column:str) -> pd.DataFrame:
        """
        Add time series specific features like lag, rolling statistics, and seasonality components.
        
        Args:
            data (pd.DataFrame): Input dataframe with time series data
            date_column (str): Column containing date/time information
            
        Returns:
            pd.DataFrame: DataFrame with additional time series features
        """
        df = data.copy()
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            # Extract datetime components
            df['year'] = df[date_column].dt.year
            df['month'] = df[date_column].dt.month
            df['day'] = df[date_column].dt.day
            df['day_of_week'] = df[date_column].dt.dayofweek
            df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        if "numerical" in self.feature_configs:
            for col in self.feature_configs["numerical"]:
                for lag in [1,2,3,5,7,14,21]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                # Add rolling statistics
                for window in [7,14,30]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    
                # Add rate of change
                df[f'{col}_pct_change'] = df[col].pct_change()
        return df
    