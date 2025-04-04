import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Callable, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
try:
    import pandera as pa
    from pandera.typing import DataFrame, Series
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    print("Pandera not installed. Advanced data validation will be limited.")

try:
    from holidays import country_holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    print("Holidays package not installed. Holiday features will not be available.")

class DataPreprocessor:
    def __init__(self, feature_configs:Optional[Dict[str, Any]]=None) -> None:
        """
        Initialize the DataPreprocessor.

        Args:
            feature_configs (Dict[str, Any], optional): Configuration for feature transformations.
                Example:
                {
                    'numerical': ['col1', 'col2'],
                    'categorical': ['col3'],
                    'date_column': 'Date', # If time features needed
                    'scaling_strategy': 'standard', # 'standard' or 'minmax'
                    'encoding_strategy': 'label', # 'label' or 'onehot'
                    'time_features': { # Optional section for time series
                       'lags': [1, 3, 7],
                       'windows': [7, 14],
                       'add_date_components': True
                    }
                }
                Defaults to None (no scaling/encoding unless methods are called manually).
        """
        self.feature_configs = feature_configs or {}
        self.transformers:Dict[str, Any] = {}
        self._is_fitted = False

    def _validate_columns(self, data:pd.DataFrame, column_type:str) -> None:
        """Helper to validate column existence"""
        if column_type in self.feature_configs:
            cols = self.feature_configs[column_type]
            if isinstance(cols, list):
                missing_cols = [col for col in cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"'{column_type}' columns not found in data: {missing_cols}")
            elif isinstance(cols, str):
                if cols not in data.columns:
                    raise ValueError(f"'{column_type}' column '{cols}' not found in data")
                
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
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        self._validate_columns(data, "numerical")
        self._validate_columns(data, "categorical")
        # Fit transformers based on feature configurations
        self.transformers = {}

        if "numerical" in self.feature_configs and self.feature_configs["numerical"]:
            num_cols = self.feature_configs["numerical"]
            # Defaut to standard if not specified
            strategy = self.feature_configs.get("scaling_strategy", "standard").lower()
            if strategy == "minmax":
                feature_range = self.feature_configs.get("scaling_range", (0,1))
                print("Fitting MinMaxScaler with range {feature_range} on columns: {num_cols}")
                scaler = MinMaxScaler(feature_range=feature_range)
            elif strategy == "standard":
                print(f"Fitting StandardScaler on columns: {num_cols}")
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unsupported scaling_strategy: {strategy}. Choose 'standard' or 'minmax'.")
            
            # Ensure columns exist before fitting
            missing_cols = [col for col in num_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Numerical columns for scaling not found in data: {missing_cols}")
            self.transformers["scaler"] = scaler.fit(data[num_cols])
            self.transformers["numerical_cols"] = num_cols

        if "categorical" in self.feature_configs and self.feature_configs["categorical"]:
            cat_cols = self.feature_configs["categorical"]
            strategy = self.feature_configs.get("encoding_strategy", "label").lower()
            if strategy == "label":
                self.transformers["encoders"] = {col:LabelEncoder().fit(data[col].astype(str)) for col in cat_cols}
            elif strategy == "onehot":
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.transformers["onehot_encoder"] = encoder.fit(data[cat_cols])
                # Store feature name generated by OneHotEncoder
                self.transformers["onehot_features"] = encoder.get_feature_names_out(cat_cols).tolist()
            else:
                raise ValueError(f"Unsupported encoding_strategy: {strategy}")
            self.transformers["categorical_cols"] = cat_cols
        self._is_fitted = True
        print("Preprocessor fitted")
    
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
        if not self._is_fitted:
            raise ValueError("Preprocessor has not been fitted. Call fit() first")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame")
        processed_data = data.copy()
        # Apply scaler
        if "scaler" in self.transformers:
            num_cols = self.transformers["numerical_cols"]
            self._validate_columns(processed_data, "numerical") # Check cols exist in data to transform
            processed_data[num_cols] = self.transformers["scaler"].transform(processed_data[num_cols])

        # Apply encoders
        if "encoders" in self.transformers: # Label encoding
            cat_cols = self.transformers["categorical_cols"]
            self._validate_columns(processed_data, "categorical")
            for col, encoder in self.transformers["encoders"].items():
                # Handle potential unseen labels during transfrom if LabelEncoder used
                processed_data[col] = processed_data[col].astype(str).apply(lambda x:encoder.transform([x])[0] if x in encoder.classes_ else - 1)
                if (processed_data[col] == -1).any():
                    print(f"Warning: Unseen labels encountered in column '{col}' during transform, mapped to -1")
        elif "onehot_encoder" in self.transformers:
            cat_cols = self.transformers["categorical_cols"]
            self._validate_columns(processed_data, "categorical")
            encoded_features = self.transformers["onehot_encoder"].transform(processed_data[cat_cols])
            encoded_df = pd.DataFrame(encoded_features, columns = self.transformers["onehot_features"],
                                      index = processed_data.index)
            processed_data = pd.concat([processed_data.drop(columns = cat_cols), encoded_df], axis = 1)

            # Apply time series features if specified in config
            if "time_features" in self.feature_configs and "date_column" in self.feature_configs:
                if self.feature_configs["date_column"] in processed_data.columns:
                    processed_data = self.add_time_series_features(processed_data)
                else:
                    print(f"Warning: Date column '{self.feature_config['date_column']}' not found. Skipping time series features")
            
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
        report = {
            "shape":data.shape,
            "missing_values_count":data.isnull().sum().to_dict(),
            "missing_values_percent":(data.isnull().sum()/len(data)*100).round(2).to_dict(),
            "data_types":data.dtypes.apply(lambda x:str(x)).to_dict(),
            "unique_counts":{col:data[col].nunique() for col in data.columns if data[col].dtype == "object" or data[col].nunique() < 20},
            "validation_errors": []
        }
        
        # Basic statistics for numerical columns
        num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if num_cols:
            report["numerical_stats"] = {
                "min": data[num_cols].min().to_dict(),
                "max": data[num_cols].max().to_dict(),
                "mean": data[num_cols].mean().to_dict(),
                "median": data[num_cols].median().to_dict(),
                "std": data[num_cols].std().to_dict(),
                "skew": data[num_cols].skew().to_dict(),
                "outliers": {}
            }
            
            # Check for outliers using IQR method
            for col in num_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                report["numerical_stats"]["outliers"][col] = len(outliers)
        
        # Advanced validation with Pandera if available
        if PANDERA_AVAILABLE and self.feature_configs:
            try:
                # Create a schema based on feature_configs
                schema_dict = {"columns": {}}
                
                # Add checks for numerical columns
                if "numerical" in self.feature_configs:
                    for col in self.feature_configs["numerical"]:
                        if col in data.columns:
                            schema_dict["columns"][col] = pa.Column(
                                pa.Float, 
                                nullable=False,
                                checks=[
                                    pa.Check.not_nan(),
                                    pa.Check(lambda x: ~np.isinf(x), error="infinity values detected")
                                ]
                            )
                
                # Add checks for categorical columns
                if "categorical" in self.feature_configs:
                    for col in self.feature_configs["categorical"]:
                        if col in data.columns:
                            unique_vals = data[col].unique().tolist()
                            schema_dict["columns"][col] = pa.Column(
                                pa.String if data[col].dtype == "object" else data[col].dtype,
                                checks=[pa.Check.isin(unique_vals)]
                            )
                
                # Add date column check
                if "date_column" in self.feature_configs:
                    date_col = self.feature_configs["date_column"]
                    if date_col in data.columns:
                        schema_dict["columns"][date_col] = pa.Column(
                            pa.DateTime,
                            checks=[
                                pa.Check(lambda x: pd.notna(x), error="date cannot be null")
                            ]
                        )
                
                # Create and validate with schema
                schema = pa.DataFrameSchema(**schema_dict)
                try:
                    schema.validate(data, lazy=True)
                except pa.errors.SchemaErrors as e:
                    report["validation_errors"] = e.failure_cases.to_dict(orient="records")
            
            except Exception as e:
                print(f"Pandera schema validation error: {e}")
                report["validation_errors"].append(str(e))
        
        print("--- Data Validation Report ---")
        print(f"Shape: {report['shape']}")
        print("\nMissing Values (%):")
        for col, pct in report['missing_values_percent'].items():
             if pct > 0:
                  print(f"  {col}: {pct}% ({report['missing_values_count'][col]})")
        print("\nData Types:")
        for col, dtype in report['data_types'].items():
             print(f"  {col}: {dtype}")
             
        # Print numerical stats summary
        if "numerical_stats" in report:
            print("\nNumerical Column Statistics:")
            for col in num_cols:
                print(f"  {col}:")
                print(f"    Range: {report['numerical_stats']['min'][col]} to {report['numerical_stats']['max'][col]}")
                print(f"    Mean: {report['numerical_stats']['mean'][col]:.2f}, Median: {report['numerical_stats']['median'][col]:.2f}")
                print(f"    Standard Deviation: {report['numerical_stats']['std'][col]:.2f}")
                print(f"    Skewness: {report['numerical_stats']['skew'][col]:.2f}")
                print(f"    Potential Outliers: {report['numerical_stats']['outliers'][col]}")
        
        # Print validation errors
        if report["validation_errors"]:
            print("\nValidation Errors:")
            for error in report["validation_errors"][:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(report["validation_errors"]) > 10:
                print(f"  ... and {len(report['validation_errors']) - 10} more errors")
        
        return report
    
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
        config = self.feature_configs.get("time_features", {})
        date_col = self.feature_configs.get("date_column")
        if not date_col or date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not configured or not found in DataFrame")
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            # It's often beneficial to set the date column as index for time-series ops
            # df = df.set_index(date_col).sort_index() # Optional, depends on workflow
        except Exception as e:
            raise ValueError(f"Could not parse date column '{date_col}':{e}")
        # Add date components (Year, Month, Day, etc.)
        if config.get("add_date_components", False):
            df[f'{date_col}_year'] = df[date_col].dt.year
            df[f'{date_col}_month'] = df[date_col].dt.month
            df[f'{date_col}_day'] = df[date_col].dt.day
            df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
            df[f'{date_col}_dayofyear'] = df[date_col].dt.dayofyear
            df[f'{date_col}_weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
            df[f'{date_col}_quarter'] = df[date_col].dt.quarter
            df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
            df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
            
            # Add weekend indicator
            df[f'{date_col}_is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Add holiday indicators if configured and holidays package is available
        if config.get("add_holidays", False):
            country_code = config.get("holiday_country", "US")
            if HOLIDAYS_AVAILABLE:
                # Get min and max year from the dataset
                min_year = df[date_col].dt.year.min()
                max_year = df[date_col].dt.year.max()
                
                # Get holidays for the relevant years
                try:
                    holidays_dict = country_holidays(country_code, years=range(min_year, max_year + 1))
                    
                    # Add holiday indicator
                    df[f'{date_col}_is_holiday'] = df[date_col].dt.date.isin(holidays_dict.keys()).astype(int)
                    
                    # Add holiday name (if it's a holiday)
                    holiday_names = df[date_col].dt.date.apply(lambda x: holidays_dict.get(x, ""))
                    df[f'{date_col}_holiday_name'] = holiday_names
                    
                    # Add days to/from nearest holiday
                    date_series = pd.Series(df[date_col].dt.date)
                    holiday_dates = sorted(holidays_dict.keys())
                    
                    # Function to find distance to nearest holiday
                    def days_to_nearest_holiday(date):
                        if date in holidays_dict:
                            return 0
                        
                        # Find closest holiday date
                        days_to_holiday = [abs((date - hdate).days) for hdate in holiday_dates]
                        return min(days_to_holiday) if days_to_holiday else 999
                    
                    df[f'{date_col}_days_to_holiday'] = date_series.apply(days_to_nearest_holiday)
                    
                    # Add special market periods if specified
                    if config.get("add_market_periods", False):
                        # Tax season indicators (US-centric, adjust as needed)
                        df[f'{date_col}_is_tax_season'] = ((df[date_col].dt.month >= 1) & 
                                                          (df[date_col].dt.month <= 4)).astype(int)
                        
                        # Quarter end indicator (often important for financial markets)
                        df[f'{date_col}_is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
                        
                        # End of fiscal year indicator 
                        df[f'{date_col}_is_year_end'] = ((df[date_col].dt.month == 12) & 
                                                        (df[date_col].dt.day >= 28)).astype(int)
                    
                    print(f"Added holiday indicators for {country_code}")
                except Exception as e:
                    print(f"Error adding holidays for {country_code}: {e}")
            else:
                print("Warning: 'holidays' package not available. Install with 'pip install holidays'")
        
        # Add Lags and Rolling Features for numerical columns
        num_cols = self.feature_configs.get("numerical", [])
        lags = config.get("lags", [])
        windows = config.get("windows", [])

        if not num_cols:
             print("Warning: No numerical columns defined in config for lag/rolling features.")
             return df # Return early if no columns to process

        print(f"Adding time features: lags={lags}, windows={windows} for columns: {num_cols}")

        # Important: Ensure data is sorted by date for lags/rolling features to be meaningful
        df = df.sort_values(by=date_col)
        for col in num_cols:
            if col in df.columns: # Ensure column exists after potential encoding charges
                # Lags
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                # Rolling statistics
                for window in windows:
                    if len(df) >= window:
                        df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                        df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()
                    else:
                        print(f"Warning: window size {window} > data length {len(df)} for column {col}. Skipping rolling features")
        # Handle NaNs introduced by shift/rolling
        # Strategies: forward fill, backward fill, drop rows, impute
        # Simple approach: backfill then forward fill
        initial_len = len(df)
        df.dropna(inplace = True)
        print(f"Dropped {initial_len - len(df)} rows with NaNs after adding time features")
        return df
    
    def save(self, filepath: str) -> None:
        """Save the fitted preprocessor state (transformers and config)."""
        if not self._is_fitted:
             print("Warning: Preprocessor is not fitted. Saving configuration only.")
        state = {
            "feature_configs": self.feature_configs,
            "transformers": self.transformers,
            "_is_fitted": self._is_fitted
        }
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(state, filepath)
            print(f"Preprocessor state saved to {filepath}")
        except Exception as e:
            print(f"Error saving preprocessor state: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load a saved preprocessor state."""
        try:
            state = joblib.load(filepath)
            preprocessor = cls(feature_configs=state["feature_configs"])
            preprocessor.transformers = state["transformers"]
            preprocessor._is_fitted = state["_is_fitted"]
            print(f"Preprocessor state loaded from {filepath}")
            return preprocessor
        except FileNotFoundError:
            raise FileNotFoundError(f"Preprocessor state file not found at: {filepath}")
        except Exception as e:
            raise IOError(f"Error loading preprocessor state from {filepath}: {e}")