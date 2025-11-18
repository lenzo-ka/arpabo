"""Data utilities for ArpaLM"""

import os
from typing import Optional


def get_data_path(filename: str) -> str:
    """Get the absolute path to a data file.

    Args:
        filename: Name of the data file

    Returns:
        Absolute path to the data file

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filename}")

    return filepath


def get_example_corpus(name: Optional[str] = None) -> str:
    """Get path to an example corpus file.

    Args:
        name: Name of the example corpus (default: 'alice.txt')

    Returns:
        Absolute path to the example corpus
    """
    if name is None:
        name = "alice.txt"
    return get_data_path(name)


def list_example_corpora() -> list[str]:
    """List available example corpus files.

    Returns:
        List of example corpus filenames
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        return []

    return [f for f in os.listdir(data_dir) if f.endswith(".txt")]
