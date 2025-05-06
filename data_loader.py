"""
Functions for loading and performing initial preparation of the dataset.
"""
import pandas as pd
import os
import warnings

def load_data(config):
    """
    Loads the dataset from the specified CSV file.

    Args:
        config: The configuration module/object containing file paths and settings.

    Returns:
        pandas.DataFrame: The loaded dataframe, or None if loading fails.
    """
    print(f"Loading data from {config.INPUT_CSV}...")
    if not os.path.exists(config.INPUT_CSV):
        print(f"Error: Input file not found at {config.INPUT_CSV}")
        print("Please download the Sentiment140 dataset and place it in the project root.")
        return None

    try:
        df = pd.read_csv(
            config.INPUT_CSV,
            encoding=config.CSV_ENCODING,
            header=None,
            names=config.CSV_COLUMN_NAMES,
            nrows=config.SAMPLE_SIZE # Use SAMPLE_SIZE from config
        )
        print(f"Loaded {len(df)} rows.")

        # --- Validate essential columns ---
        if config.TEXT_COLUMN not in df.columns:
            print(f"Error: Text column '{config.TEXT_COLUMN}' not found in the CSV.")
            return None
        if config.DATE_COLUMN not in df.columns:
            print(f"Warning: Date column '{config.DATE_COLUMN}' not found. Time analysis will be skipped.")
        else:
            # --- Date Parsing ---
            print(f"Parsing date column: {config.DATE_COLUMN}")
            # Suppress warnings during inference if format is consistent but non-standard
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=UserWarning)
                 df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN], errors='coerce')
            invalid_dates = df[config.DATE_COLUMN].isnull().sum()
            if invalid_dates > 0:
                print(f"Warning: {invalid_dates} rows have invalid dates after parsing.")

        # Ensure text column is string and handle NaNs
        df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).fillna('')

        return df

    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        return None