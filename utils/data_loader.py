"""Utility functions for loading project datasets."""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_crop_data() -> pd.DataFrame:
    """Load the crop recommendation CSV into a DataFrame."""
    path = os.path.join(DATA_DIR, "crop_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"crop_data.csv not found at {path}. "
            "Run data/generate_crop_data.py first."
        )
    return pd.read_csv(path)


def load_market_prices() -> pd.DataFrame:
    """Load market price data for all crops."""
    path = os.path.join(DATA_DIR, "market_prices.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"market_prices.csv not found at {path}.")
    return pd.read_csv(path)
