import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "credit_card_default_features.csv"
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# AGE_GROUP is used later for fairness analysis,
# but it is not used as a model feature.
df_model = df.drop(columns=["AGE_GROUP"])

X = df_model.drop(columns=["default"])
y = df_model["default"]


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 80% temporary data
# 20% final test data

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split:
# 60% training
# 20% validation

X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# DISPLAY DATA SPLIT
# ============================================================

print("=" * 75)
print("PROPER THRESHOLD TUNING")
print("=" * 75)

print("\nDataset split:")
print("-" * 40)

print(f"Total samples      : {len(X)}")
print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_validation)}")
print(f"Testing samples    : {len(X_test)}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading Gradient Boosting model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# THRESHOLD ANALYSIS ON VALIDATION SET
# ============================================================

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

print("\n" + "=" * 75)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)

print("-" * 75)


for threshold in thresholds:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1:<15.4f}"
    )


# ============================================================
# SELECT BEST THRESHOLD
# ============================================================

best_result = max(
    results,
    key=lambda x: x["f1"]
)

best_threshold = best_result["threshold"]
best_validation_f1 = best_result["f1"]


print("\n" + "=" * 75)
print("SELECTED THRESHOLD")
print("=" * 75)

print(f"Best threshold : {best_threshold:.2f}")
print(f"Validation F1  : {best_validation_f1:.4f}")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST SET EVALUATION")
print("=" * 75)

# IMPORTANT:
# The test set was NOT used to select the threshold.

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# TEST METRICS
# ============================================================

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)


print("\nFinal Test Metrics:")
print("-" * 40)

print(f"Threshold : {best_threshold:.2f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1 Score  : {test_f1:.4f}")
print(f"ROC-AUC   : {test_roc_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nConfusion Matrix:")
print("-" * 40)

print(cm)

print("\nInterpretation:")
print(f"True Negatives  : {cm[0][0]}")
print(f"False Positives : {cm[0][1]}")
print(f"False Negatives : {cm[1][0]}")
print(f"True Positives  : {cm[1][1]}")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 75)