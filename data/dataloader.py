import pandas as pd
from typing import List, Callable, Tuple
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dataset_path:str, preprocessing_steps:List[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 target_column:str = None) -> None:
        """
        Initialize the DataLoader.

        Args:
            dataset_path (str): Path to the dataset file (CSV).
            preprocessing_steps (List[Callable[[pd.DataFrame], pd.DataFrame]], optional): 
                List of preprocessing functions to apply to the data. Each function takes a DataFrame 
                and returns a processed DataFrame. Defaults to None.
            target_column (str, optional): Name of the target column for supervised learning tasks. 
                If None, assumes an unsupervised task. Defaults to None.
        """
        self.dataset_path = dataset_path
        self.preprocessing_steps = preprocessing_steps or []
        self.target_column = target_column
        self.data:pd.DataFrame | None = None
    
    def load_data(self) -> None:
        """
        Load the data from the specified path into a pandas DataFrame.

        Raises:
            FileNotFoundError: If the dataset file does not exist at the specified path.
            Exception: If an error occurs during data loading (e.g., invalid CSV format).
        """
        try:
            self.data = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def preprocess_data(self) -> None:
        """
        Apply the specified preprocessing steps to the data.

        Raises:
            ValueError: If data has not been loaded prior to preprocessing.
        """
        if self.data is None:
            raise ValueError("Data not loaded, Call load_data() first")
        for step in self.preprocessing_steps:
            self.data = step(self.data)
    
    def prepare_data(self) -> None:
        """
        Load and preprocess the data in a single step.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            Exception: If an error occurs during loading or preprocessing.
        """
        self.load_data()
        self.preprocess_data()
    
    def get_data(self) -> pd.DataFrame:
        """
        Retrieve the loaded and preprocessed data.

        Returns:
            pd.DataFrame: The processed dataset.

        Raises:
            ValueError: If data has not been loaded or prepared.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call prepare_data() first")
        return self.data
    
    def get_train_val_test_split(self, train_ratio:float=0.7, val_ratio:float=0.15, test_ratio:float=0.15,
                                 random_state:int|None=None
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None, pd.Series | None]:
        """
        Split the data into training, validation, and test sets.

        Args:
            train_ratio (float): Proportion of data for the training set (default: 0.7).
            val_ratio (float): Proportion of data for the validation set (default: 0.15).
            test_ratio (float): Proportion of data for the test set (default: 0.15).
            random_state (int | None, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            Tuple:
                - X_train (pd.DataFrame): Training features.
                - X_val (pd.DataFrame): Validation features.
                - X_test (pd.DataFrame): Test features.
                - y_train (pd.Series | None): Training targets (None if target_column is not specified).
                - y_val (pd.Series | None): Validation targets (None if target_column is not specified).
                - y_test (pd.Series | None): Test targets (None if target_column is not specified).

        Raises:
            ValueError: If data is not loaded, ratios do not sum to 1.0, or target_column is invalid.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call prepare_data() first")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratio must sum to 1.0")
        if self.target_column:
            if self.target_column not in self.data_columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_ratio,
                                                                random_state=random_state)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                             train_size=val_ratio/(val_ratio + test_ratio),
                                                             random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            train, temp = train_test_split(self.data, train_size=train_ratio, random_state=random_state)
            val, test = train_test_split(temp, train_size=val_ratio/(val_ratio+test_ratio),
                                        random_state=random_state)
            return train, val, test, None, None, None

    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the dataset.

        Returns:
            pd.Series: Number of missing values per column.

        Raises:
            ValueError: If data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")
        return self.data.isnull().sum()
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics of the dataset (e.g., mean, std, min, max).

        Returns:
            pd.DataFrame: Summary statistics of the data.

        Raises:
            ValueError: If data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")
        return self.data.describe()