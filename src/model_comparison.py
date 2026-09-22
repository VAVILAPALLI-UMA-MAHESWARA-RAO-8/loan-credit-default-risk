import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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

MODELS_DIR = BASE_DIR / "models"


# ============================================================
# MODELS TO COMPARE
# ============================================================

MODELS = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl"
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

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

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_probability
        )
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Same split for every model
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = []

    # --------------------------------------------------------
    # Evaluate every model
    # --------------------------------------------------------

    for model_name, model_file in MODELS.items():

        print(f"\nEvaluating: {model_name}")

        model_path = MODELS_DIR / model_file

        model = joblib.load(model_path)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = model_name

        results.append(metrics)

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ]

    # --------------------------------------------------------
    # Sort by ROC-AUC
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)