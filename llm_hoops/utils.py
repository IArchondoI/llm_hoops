"""Group all utils for the model."""

import pandas as pd
from pathlib import Path
import email
from email import policy


def load_data(data_path: Path) -> pd.DataFrame:
    """Load data."""
    return pd.read_csv(data_path)


def load_text(file_path):
    """Load a query from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {file_path} was not found.") from e
    except IOError as e:
        raise IOError(
            f"An error occurred while reading the file at {file_path}."
        ) from e

def load_email(file_path: Path) -> str:
    """Load and decode a single .eml file, returning the plain text body."""
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    # Extract the plain text body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()

    return body.strip()