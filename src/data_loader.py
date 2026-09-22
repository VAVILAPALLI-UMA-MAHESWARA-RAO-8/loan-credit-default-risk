import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "default_of_credit_card_clients.xls"


def load_data():
    """Load the credit card default dataset."""

    df = pd.read_excel(DATA_PATH, engine="xlrd")

    return df


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("CREDIT CARD DEFAULT DATASET")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())