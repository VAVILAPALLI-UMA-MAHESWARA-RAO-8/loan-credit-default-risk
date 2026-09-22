import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_DATA_PATH = BASE_DIR / "data" / "credit_card_default_clean.csv"

FEATURE_DATA_PATH = BASE_DIR / "data" / "credit_card_default_features.csv"


# ============================================================
# LOAD CLEAN DATA
# ============================================================

def load_clean_data():
    """Load the cleaned dataset."""

    df = pd.read_csv(CLEAN_DATA_PATH)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """Create meaningful financial and behavioral features."""

    df = df.copy()

    # --------------------------------------------------------
    # 1. Remove duplicate records
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    after = len(df)

    print(f"Duplicate rows removed: {before - after}")

    # --------------------------------------------------------
    # 2. Average bill amount
    # --------------------------------------------------------

    bill_columns = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6"
    ]

    df["AVG_BILL_AMT"] = df[bill_columns].mean(axis=1)

    # --------------------------------------------------------
    # 3. Average payment amount
    # --------------------------------------------------------

    payment_columns = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6"
    ]

    df["AVG_PAY_AMT"] = df[payment_columns].mean(axis=1)

    # --------------------------------------------------------
    # 4. Maximum bill amount
    # --------------------------------------------------------

    df["MAX_BILL_AMT"] = df[bill_columns].max(axis=1)

    # --------------------------------------------------------
    # 5. Maximum payment amount
    # --------------------------------------------------------

    df["MAX_PAY_AMT"] = df[payment_columns].max(axis=1)

    # --------------------------------------------------------
    # 6. Credit utilization
    # --------------------------------------------------------

    df["CREDIT_UTILIZATION"] = (
        df["AVG_BILL_AMT"] / df["LIMIT_BAL"]
    )

    # Avoid infinite values
    df["CREDIT_UTILIZATION"] = (
        df["CREDIT_UTILIZATION"]
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 7. Total payment
    # --------------------------------------------------------

    df["TOTAL_PAYMENT"] = df[payment_columns].sum(axis=1)

    # --------------------------------------------------------
    # 8. Total bills
    # --------------------------------------------------------

    df["TOTAL_BILL"] = df[bill_columns].sum(axis=1)

    # --------------------------------------------------------
    # 9. Payment-to-bill ratio
    # --------------------------------------------------------

    df["PAYMENT_TO_BILL_RATIO"] = (
        df["TOTAL_PAYMENT"] /
        df["TOTAL_BILL"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 10. Average repayment status
    # --------------------------------------------------------

    pay_status_columns = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6"
    ]

    df["AVG_PAYMENT_STATUS"] = (
        df[pay_status_columns].mean(axis=1)
    )

    # --------------------------------------------------------
    # 11. Number of delayed payments
    # --------------------------------------------------------

    df["NUM_DELAYED_PAYMENTS"] = (
        df[pay_status_columns] > 0
    ).sum(axis=1)

    # --------------------------------------------------------
    # 12. Age group
    # --------------------------------------------------------

    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[0, 25, 35, 50, 100],
        labels=[
            "18-25",
            "26-35",
            "36-50",
            "51+"
        ]
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Load cleaned data
    df = load_clean_data()

    print("\nOriginal shape:")
    print(df.shape)

    # Create features
    df_features = create_features(df)

    print("\nNew shape:")
    print(df_features.shape)

    print("\nNew features:")
    new_features = [
        "AVG_BILL_AMT",
        "AVG_PAY_AMT",
        "MAX_BILL_AMT",
        "MAX_PAY_AMT",
        "CREDIT_UTILIZATION",
        "TOTAL_PAYMENT",
        "TOTAL_BILL",
        "PAYMENT_TO_BILL_RATIO",
        "AVG_PAYMENT_STATUS",
        "NUM_DELAYED_PAYMENTS",
        "AGE_GROUP"
    ]

    print(new_features)

    print("\nFeature preview:")
    print(df_features[new_features].head())

    print("\nTarget distribution:")
    print(df_features["default"].value_counts())

    print("\nTarget percentage:")
    print(
        df_features["default"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # Save
    df_features.to_csv(
        FEATURE_DATA_PATH,
        index=False
    )

    print("\nFeature-engineered dataset saved to:")
    print(FEATURE_DATA_PATH)