import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "credit_card_default_features.csv"

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # Remove AGE_GROUP because it is a categorical feature
    # We will handle categorical features later.
    df = df.drop(columns=["AGE_GROUP"])

    # Separate features and target
    X = df.drop(columns=["default"])

    y = df["default"]

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LOGISTIC REGRESSION MODEL TRAINING")
    print("=" * 60)

    # Load dataset
    df = load_data()

    print("\nDataset shape:")
    print(df.shape)

    # Prepare data
    X, y = prepare_data(df)

    print("\nFeatures shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:")
    print(X_train.shape[0])

    print("\nTesting samples:")
    print(X_test.shape[0])

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    # Save model
    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel trained successfully.")

    print("\nModel saved to:")
    print(MODEL_PATH)