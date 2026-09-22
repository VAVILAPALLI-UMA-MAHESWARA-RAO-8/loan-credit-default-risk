import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "default_of_credit_card_clients.xls"

CLEAN_DATA_PATH = BASE_DIR / "data" / "credit_card_default_clean.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data():
    """Load the original Excel dataset."""

    df = pd.read_excel(
        DATA_PATH,
        engine="xlrd"
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    """Clean and prepare the credit card default dataset."""

    # --------------------------------------------------------
    # Step 1: First row contains the actual column names
    # --------------------------------------------------------

    df.columns = df.iloc[0]

    # Remove the first row because it is now used as headers
    df = df.iloc[1:].copy()

    # --------------------------------------------------------
    # Step 2: Remove unnecessary first column
    # --------------------------------------------------------

    df = df.drop(columns=["ID"])

    # --------------------------------------------------------
    # Step 3: Convert all columns to numeric
    # --------------------------------------------------------

    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Step 4: Rename target column
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "default payment next month": "default"
        }
    )

    # --------------------------------------------------------
    # Step 5: Reset index
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    # Load raw data
    df_raw = load_raw_data()

    print("\nRaw dataset shape:")
    print(df_raw.shape)

    # Clean data
    df_clean = clean_data(df_raw)

    print("\nCleaned dataset shape:")
    print(df_clean.shape)

    print("\nCleaned column names:")
    print(df_clean.columns.tolist())

    print("\nFirst 5 rows:")
    print(df_clean.head())

    print("\nData types:")
    print(df_clean.dtypes)

    print("\nMissing values:")
    print(df_clean.isnull().sum())

    print("\nDuplicate rows:")
    print(df_clean.duplicated().sum())

    # Save cleaned dataset
    df_clean.to_csv(
        CLEAN_DATA_PATH,
        index=False
    )

    print("\nCleaned dataset saved to:")
    print(CLEAN_DATA_PATH)