import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union

from data_pipeline import ingest_data
from data_preprocessor import DataPreprocessor

class DataLoader:
    def __init__(self, source:str, preprocessor:Optional[DataPreprocessor] = None,
                 target_column:Optional[str] = None, ingest_options:Optional[dict] = None) -> None:
        """
        Initialize the DataLoader.

        Args:
            source (str): Path to the dataset file (CSV, JSON) or API URL.
                          Passed directly to `ingest_data`.
            preprocessor (DataPreprocessor, optional): An instance of DataPreprocessor
                                                       to apply transformations. Defaults to None.
            target_column (str, optional): Name of the target variable for supervised tasks.
                                           Defaults to None.
            ingest_options (dict, optional): Dictionary of options to pass to `ingest_data`
                                             (e.g., {'headers': {...}, 'timeout': 60}).
        """
        self.source = source
        self.preprocessor = preprocessor
        self.target_column = target_column
        self.ingest_options = ingest_options or {}
        self.raw_data:Optional[pd.DataFrame] = None
        self.data:Optional[pd.DataFrame] = None # Hold processed Data
    
    def load_data(self) -> None:
        """
        Load the data from the specified source using `ingest_data`.

        Raises:
            Exceptions from `ingest_data` (e.g., FileNotFoundError, ValueError, RequestException).
        """
        print(f"Loading data from source: {self.source}")
        try:
            self.raw_data = ingest_data(self.source, **self.ingest_options)
            self.data = self.raw_data.copy() 
            print(f"Loaded raw data with shape: {self.raw_data.shape}")
        except Exception as e:
            print(f"Error loading data from {self.source}: {e}")
            raise
    
    def preprocess_data(self) -> None:
        """
        Apply preprocessing using the provided DataPreprocessor instance.
        Assumes the preprocessor should be fit on this data or is already fitted.

        Raises:
            ValueError: If data hasn't been loaded or no preprocessor is provided.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")
        if self.preprocessor is None:
            print("Warning: No preprocessor provided. Skipping preprocessing step.")
            raise ValueError("Preprocessor instance is required for preprocess_data()")
        print("Applying preprocessing steps...")
        try:
            if not self.preprocessor._is_fitted:
                print("Preprocessor not fitted. Fitting and transforming")
                self.data = self.preprocessor.fit_transform(self.data)
            else:
                print("Preprocessor already fitted. Applying transform...")
                self.data = self.preprocessor.transform(self.data)
            print(f"Preprocessing complete. Processed data shape: {self.data.shape}")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise
    def prepare_data(self) -> None:
        """Load and preprocess the data sequentially"""
        self.load_data()
        if self.preprocessor:
            self.preprocessor.validate_data(self.data)
        self.preprocess_data()

    def get_data(self) -> pd.DataFrame:
        """Retrieve the processed data"""
        if self.data is None:
            raise ValueError("Data not loaded or prepared. Call prepare_data() first")
        return self.data
    
    def get_raw_data(self) -> pd.DataFrame:
        """Retrieve the raw, unprocessed data"""
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_data() first")
        return self.raw_data
    
    def get_train_val_test_split(self,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15
                                ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                                           Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                 pd.Series, pd.Series, pd.Series]]:
        """
        Split the data chronologically into training, validation, and test sets.
        Assumes the data is sorted by time if applicable.

        Args:
            train_ratio (float): Proportion for the training set (default: 0.7).
            val_ratio (float): Proportion for the validation set (default: 0.15).
            test_ratio (float): Proportion for the test set (default: 0.15).

        Returns:
            Tuple:
                If target_column is specified:
                (X_train, X_val, X_test, y_train, y_val, y_test)
                If target_column is None:
                (train_data, val_data, test_data)

        Raises:
            ValueError: If data is not loaded, ratios don't sum close to 1.0,
                        or target_column is invalid.
        """
        if self.data is None:
            raise ValueError("Data not loaded/prepared. Call prepare_data() first")
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
        if train_ratio <= 0or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Ratios must be non-negative, train_ratio must be positive")
        n = len(self.data)
        train_end_idx = int(n * train_ratio)
        val_end_idx = train_end_idx + int(n * val_ratio)
        print(f"Splitting data chronologically: Train ({train_ratio*100:.1f}%), Validation ({val_ratio*100:.1f}%), Test ({test_ratio*100:.1f}%)")
        print(f"Indices: Train [0:{train_end_idx}], Validation [{train_end_idx}:{val_end_idx}], Test [{val_end_idx}:{n}]")

        if self.target_column:
            if self.target_column not in self.data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in processed data columns: {self.data.columns.tolist()}")
            X = self.data.drop(columns = [self.target_column])
            y = self.data[self.target_column]

            X_train = X.iloc[:train_end_idx]
            y_train = y.iloc[:train_end_idx]

            X_val = X.iloc[train_end_idx:val_end_idx]
            y_val = y.iloc[train_end_idx:val_end_idx]

            X_test = X.iloc[val_end_idx:]
            y_test = y.iloc[val_end_idx:]

            # Validate shapes
            assert len(X_train) + len(X_val) + len(X_test) == len(X)
            assert len(y_train) + len(y_val) + len(y_test) == len(y)

            print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
            print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
            print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Unsupervised case
            train_data = self.data.iloc[:train_end_idx]
            val_data = self.data.iloc[train_end_idx:val_end_idx]
            test_data = self.data.iloc[val_end_idx:]

             # Validate shapes
            assert len(train_data) + len(val_data) + len(test_data) == len(self.data)

            print(f"Train shape: {train_data.shape}")
            print(f"Val shape: {val_data.shape}")
            print(f"Test shape: {test_data.shape}")

            return train_data, val_data, test_data # Return only 3 items
        
    def check_missing_values(self, check_raw=False) -> Optional[pd.Series]:
        """Check for missing values in the processed (or raw) dataset."""
        data_to_check = self.raw_data if check_raw else self.data
        if data_to_check is None:
            status = "Raw data" if check_raw else "Processed data"
            print(f"Warning: {status} not available for missing value check.")
            return None
        print(f"--- Missing Values ({'Raw' if check_raw else 'Processed'} Data) ---")
        missing = data_to_check.isnull().sum()
        print(missing[missing > 0])
        if missing.sum() == 0:
             print("No missing values found.")
        return missing

    def get_summary_statistics(self, check_raw=False) -> Optional[pd.DataFrame]:
        """Get summary statistics of the processed (or raw) dataset."""
        data_to_check = self.raw_data if check_raw else self.data
        if data_to_check is None:
            status = "Raw data" if check_raw else "Processed data"
            print(f"Warning: {status} not available for summary statistics.")
            return None
        print(f"--- Summary Statistics ({'Raw' if check_raw else 'Processed'} Data) ---")
        summary = data_to_check.describe(include='all')
        print(summary)
        return summary
    
    def prepare_sequences(data: Union[pd.DataFrame, np.ndarray],
                      target: Optional[Union[pd.Series, np.ndarray]],
                      sequence_length: int,
                      forecast_horizon: int = 1
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data into sequences for time series models (e.g., LSTM).
        Applies to a single dataset (e.g., train, val, or test Features + Target).

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Feature data, assumed to be ordered chronologically.
            target (Optional[Union[pd.Series, np.ndarray]]): Target data corresponding to the features.
                                                            Required for supervised learning.
            sequence_length (int): Length of the input sequences (lookback window).
            forecast_horizon (int): How many steps ahead the target is (default=1).
                                    The target y[i] will correspond to forecast_horizon steps
                                    after the end of sequence X[i].

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - X_seq: Array of input sequences (shape: [n_samples, sequence_length, n_features])
                - y_seq: Array of targets (shape: [n_samples, forecast_horizon] or [n_samples,] if horizon=1).
                        Returns None if target input is None.

        Raises:
            ValueError: If data length is insufficient or target mismatch.
        """
        X_data = data.values if isinstance(data, pd.DataFrame) else data
        y_data = target.values if isinstance(target, pd.Series) else target

        if target is not None and len(X_data) != len(y_data):
            raise ValueError("Feature data and target data must have the same length.")

        n_samples = len(X_data)
        n_features = X_data.shape[1]

        if n_samples < sequence_length + forecast_horizon:
            raise ValueError(
                f"Data length ({n_samples}) is too short for "
                f"sequence_length ({sequence_length}) + forecast_horizon ({forecast_horizon})."
            )

        X_seq, y_seq_list = [], []

        for i in range(n_samples - sequence_length - forecast_horizon + 1):
            seq_end = i + sequence_length
            target_idx = seq_end + forecast_horizon - 1 # Index of the target value

            X_seq.append(X_data[i:seq_end])

            if y_data is not None:
                y_seq_list.append(y_data[target_idx])
                target_start = seq_end + forecast_horizon - 1
                target_end = target_start + forecast_horizon
                y_seq_list.append(y_data[target_start : target_end])
        X_seq_arr = np.array(X_seq)
        if y_data is not None: 
            y_seq_arr = np.array(y_seq_list)
            return X_seq_arr, y_seq_arr
        else: 
            return X_seq_arr, None