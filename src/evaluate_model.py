import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "credit_card_default_features.csv"
)

# Current model being evaluated
MODEL_NAME = "gradient_boosting_model.pkl"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / MODEL_NAME
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

    # AGE_GROUP is categorical and is not used
    # in our current baseline models.
    df = df.drop(columns=["AGE_GROUP"])

    X = df.drop(columns=["default"])

    y = df["default"]

    return X, y


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    # Generate class predictions
    y_pred = model.predict(X_test)

    # Generate probability of default
    y_probability = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    return (
        y_pred,
        y_probability,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nModel: {MODEL_NAME}")

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

    # --------------------------------------------------------
    # Same train/test split used during training
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
    # Load trained model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        y_pred,
        y_probability,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    print("\nPerformance Metrics")
    print("-" * 40)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix")
    print("-" * 40)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report")
    print("-" * 40)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print("\nModel Interpretation")
    print("-" * 40)

    print(
        "Recall shows how many actual defaulters "
        "were correctly identified."
    )

    print(
        "ROC-AUC measures how well the model separates "
        "default and non-default cases across thresholds."
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 60)