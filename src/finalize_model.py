import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


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
    print("FINAL MODEL CREATION")
    print("=" * 70)

    print("\nDataset shape:")
    print(df.shape)

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    X = df.drop(columns=["default", "AGE_GROUP"])

    y = df["default"]

    return X, y


# ============================================================
# CREATE 60 / 20 / 20 SPLIT
# ============================================================

def create_split(X, y):

    # First split:
    # 60% train
    # 40% temporary

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split temporary 50/50:
    # 20% validation
    # 20% test

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print("\nDataset split:")

    print("Training:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ============================================================
# SELECT THRESHOLD
# ============================================================

def select_threshold(model, X_validation, y_validation):

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70
    ]

    results = []

    print("\nValidation threshold analysis:")
    print("-" * 70)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        )

        print(
            f"{threshold:.2f} "
            f"Precision={precision:.4f} "
            f"Recall={recall:.4f} "
            f"F1={f1:.4f}"
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    selected_threshold = float(
        best_row["threshold"]
    )

    print("\nSelected threshold:")
    print(selected_threshold)

    return selected_threshold


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    X_train,
    X_validation,
    y_train,
    y_validation
):

    # Combine training + validation data.
    #
    # The test set remains completely untouched.

    X_final_train = pd.concat(
        [X_train, X_validation],
        axis=0
    )

    y_final_train = pd.concat(
        [y_train, y_validation],
        axis=0
    )

    print("\nFinal training dataset:")
    print(X_final_train.shape)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_final_train,
        y_final_train
    )

    return model


# ============================================================
# EVALUATE FINAL MODEL
# ============================================================

def evaluate_final_model(
    model,
    X_test,
    y_test,
    threshold
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n")
    print("=" * 70)
    print("FINAL MODEL TEST RESULTS")
    print("=" * 70)

    print(f"\nThreshold: {threshold}")

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nConfusion Matrix Interpretation:")

    tn, fp, fn, tp = matrix.ravel()

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    return probabilities, predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    X, y = prepare_features(df)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = create_split(X, y)

    # --------------------------------------------------------
    # Temporary model for threshold selection
    # --------------------------------------------------------

    threshold_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    threshold_model.fit(
        X_train,
        y_train
    )

    selected_threshold = select_threshold(
        threshold_model,
        X_validation,
        y_validation
    )

    # --------------------------------------------------------
    # Train final model using train + validation
    # --------------------------------------------------------

    final_model = train_final_model(
        X_train,
        X_validation,
        y_train,
        y_validation
    )

    # --------------------------------------------------------
    # Evaluate ONLY on untouched test set
    # --------------------------------------------------------

    evaluate_final_model(
        final_model,
        X_test,
        y_test,
        selected_threshold
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    print("\nFinal model saved to:")
    print(MODEL_PATH)

    print("\nFinal threshold:")
    print(selected_threshold)

    print("\nFinal model creation completed successfully.")