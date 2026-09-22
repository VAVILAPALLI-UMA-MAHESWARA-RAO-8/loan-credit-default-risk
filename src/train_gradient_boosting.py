import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "credit_card_default_features.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "gradient_boosting_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the feature-engineered dataset."""

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """Separate features and target."""

    # AGE_GROUP is categorical.
    # We will handle categorical features properly
    # in a later final preprocessing pipeline.
    df = df.drop(columns=["AGE_GROUP"])

    # Features
    X = df.drop(columns=["default"])

    # Target
    y = df["default"]

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):
    """Train the Gradient Boosting classifier."""

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GRADIENT BOOSTING MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    print("\nDataset shape:")
    print(df.shape)

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X, y = prepare_data(df)

    print("\nFeatures shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nTraining model...")

    model = train_model(
        X_train,
        y_train
    )

    print("Training completed successfully.")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING TRAINING COMPLETED")
    print("=" * 60)