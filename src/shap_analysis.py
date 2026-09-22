import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


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

FIGURES_DIR = (
    BASE_DIR
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
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

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("FINAL SHAP EXPLAINABILITY")
    print("=" * 70)

    print("\nDataset shape:")
    print(df.shape)

    return df


# ============================================================
# CREATE SAME TEST SET
# ============================================================

def create_test_set(df):

    X = df.drop(
        columns=["default", "AGE_GROUP"]
    )

    y = df["default"]

    # Same split used by finalize_model.py

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
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Create exact test set
    # --------------------------------------------------------

    X_test, y_test = create_test_set(df)

    print("\nTest dataset shape:")
    print(X_test.shape)

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(
        model
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print("\nCalculating SHAP values...")

    shap_values = explainer.shap_values(
        X_test
    )

    # Handle possible SHAP output formats

    if isinstance(shap_values, list):

        shap_values_for_plot = shap_values[0]

    else:

        shap_values_for_plot = shap_values

    print(
        "\nSHAP values shape:"
    )

    print(
        shap_values_for_plot.shape
    )

    # ========================================================
    # 1. GLOBAL SHAP SUMMARY
    # ========================================================

    print("\nCreating SHAP summary plot...")

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values_for_plot,
        X_test,
        show=False
    )

    plt.tight_layout()

    summary_path = (
        FIGURES_DIR
        / "final_shap_feature_importance.png"
    )

    plt.savefig(
        summary_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        summary_path
    )

    # ========================================================
    # 2. SHAP BAR PLOT
    # ========================================================

    print("\nCreating SHAP bar plot...")

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values_for_plot,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    bar_path = (
        FIGURES_DIR
        / "final_shap_feature_importance_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        bar_path
    )

    # ========================================================
    # 3. GLOBAL FEATURE IMPORTANCE TABLE
    # ========================================================

    mean_abs_shap = np.abs(
        shap_values_for_plot
    ).mean(axis=0)

    feature_importance = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Mean_Absolute_SHAP": mean_abs_shap
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n")
    print("=" * 70)
    print("TOP SHAP FEATURES")
    print("=" * 70)

    print(
        feature_importance.head(15).to_string(
            index=False
        )
    )

    # ========================================================
    # 4. INDIVIDUAL CUSTOMER
    # ========================================================

    customer_index = 0

    customer = X_test.iloc[
        customer_index
    ]

    customer_shap_values = (
        shap_values_for_plot[
            customer_index
        ]
    )

    customer_probability = (
        model.predict_proba(
            X_test.iloc[
                [customer_index]
            ]
        )[0][1]
    )

    customer_prediction = (
        int(
            customer_probability
            >= FINAL_THRESHOLD
        )
    )

    print("\n")
    print("=" * 70)
    print("INDIVIDUAL CUSTOMER EXPLANATION")
    print("=" * 70)

    print(
        f"\nCustomer probability: "
        f"{customer_probability:.4f}"
    )

    print(
        f"Final threshold: "
        f"{FINAL_THRESHOLD}"
    )

    print(
        "Prediction:",
        "DEFAULT" if customer_prediction == 1
        else "NON-DEFAULT"
    )

    # ========================================================
    # TOP INDIVIDUAL SHAP FACTORS
    # ========================================================

    customer_explanation = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "SHAP_Value": customer_shap_values,
            "Feature_Value": customer.values
        }
    )

    customer_explanation[
        "Absolute_SHAP"
    ] = customer_explanation[
        "SHAP_Value"
    ].abs()

    customer_explanation = (
        customer_explanation
        .sort_values(
            "Absolute_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nTop individual factors:")

    print(
        customer_explanation
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # 5. WATERFALL PLOT
    # ========================================================

    print("\nCreating individual SHAP waterfall...")

    base_value = explainer.expected_value

    if hasattr(
        base_value,
        "__len__"
    ):
        base_value = base_value[0]

    explanation = shap.Explanation(
        values=customer_shap_values,
        base_values=base_value,
        data=customer.values,
        feature_names=X_test.columns.tolist()
    )

    plt.figure(
        figsize=(10, 7)
    )

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )

    plt.tight_layout()

    waterfall_path = (
        FIGURES_DIR
        / "final_shap_individual_customer.png"
    )

    plt.savefig(
        waterfall_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        waterfall_path
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SHAP ANALYSIS COMPLETED")
    print("=" * 70)