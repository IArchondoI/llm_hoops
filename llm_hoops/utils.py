"""Group all utils for the model."""

import pandas as pd
from pathlib import Path
from typing import Union

def load_data(data_path:Path)->pd.DataFrame:
    """Load data."""
    return pd.read_csv(data_path)

def load_query(file_path: Union[str, Path]) -> str:
    """Load a query from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {file_path} was not found.") from e
    except IOError as e:
        raise IOError(f"An error occurred while reading the file at {file_path}.") from e
