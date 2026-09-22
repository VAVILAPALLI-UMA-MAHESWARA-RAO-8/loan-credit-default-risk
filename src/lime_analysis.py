import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# PATHS
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
    / "final_gradient_boosting_model.pkl"
)

REPORTS_DIR = (
    BASE_DIR / "reports"
)

LIME_HTML_PATH = (
    REPORTS_DIR
    / "final_lime_individual_customer.html"
)

LIME_CSV_PATH = (
    REPORTS_DIR
    / "final_lime_individual_customer.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.35


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    print("=" * 70)
    print("FINAL LIME EXPLAINABILITY")
    print("=" * 70)

    print("\nDataset shape:")
    print(df.shape)

    return df


# ============================================================
# CREATE SAME TEST SET
# ============================================================

def create_test_set(df):

    X = df.drop(
        columns=[
            "default",
            "AGE_GROUP"
        ]
    )

    y = df["default"]

    # Same split used everywhere

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return X_test, y_test


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading final model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Final Gradient Boosting model loaded."
    )

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Create exact test set
    # --------------------------------------------------------

    X_test, y_test = create_test_set(
        df
    )

    print(
        "\nTest dataset shape:"
    )

    print(
        X_test.shape
    )

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Create LIME explainer
    # --------------------------------------------------------

    print(
        "\nCreating LIME explainer..."
    )

    explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=[
            "Non-Default",
            "Default"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Select sample customer
    # --------------------------------------------------------

    customer_index = 0

    customer = X_test.iloc[
        customer_index
    ]

    customer_array = (
        customer.values
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        customer_array.reshape(1, -1)
    )[0][1]

    prediction = int(
        probability >= FINAL_THRESHOLD
    )

    print("\n")
    print("=" * 70)
    print("INDIVIDUAL CUSTOMER")
    print("=" * 70)

    print(
        f"\nDefault probability: "
        f"{probability:.4f}"
    )

    print(
        f"Final threshold: "
        f"{FINAL_THRESHOLD}"
    )

    print(
        "Prediction:",
        "DEFAULT"
        if prediction == 1
        else "NON-DEFAULT"
    )

    # --------------------------------------------------------
    # LIME explanation
    # --------------------------------------------------------

    print(
        "\nGenerating LIME explanation..."
    )

    explanation = explainer.explain_instance(
        customer_array,
        model.predict_proba,
        num_features=10
    )

    # --------------------------------------------------------
    # Print explanation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LIME EXPLANATION")
    print("=" * 70)

    lime_results = []

    for feature, weight in (
        explanation.as_list()
    ):

        print(
            f"{feature:50s} "
            f"{weight:+.6f}"
        )

        lime_results.append(
            {
                "Feature": feature,
                "LIME_Weight": weight
            }
        )

    # --------------------------------------------------------
    # Save HTML
    # --------------------------------------------------------

    explanation.save_to_file(
        str(LIME_HTML_PATH)
    )

    print(
        "\nLIME HTML saved to:"
    )

    print(
        LIME_HTML_PATH
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    lime_df = pd.DataFrame(
        lime_results
    )

    lime_df.to_csv(
        LIME_CSV_PATH,
        index=False
    )

    print(
        "\nLIME CSV saved to:"
    )

    print(
        LIME_CSV_PATH
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL LIME ANALYSIS COMPLETED")
    print("=" * 70)