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
    roc_auc_score
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

REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

FINAL_THRESHOLD = 0.30

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("FAIRNESS MITIGATION EXPERIMENT")
print("=" * 75)

df = pd.read_csv(
    DATA_PATH
)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# CREATE TRAIN / TEST SPLIT
# ============================================================

train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["default"]
)

train_df = df.loc[
    train_indices
].copy()

test_df = df.loc[
    test_indices
].copy()

train_df = train_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# PREPARE TARGET
# ============================================================

y_train = train_df[
    "default"
]

y_test = test_df[
    "default"
]


# ============================================================
# MODEL A
# ORIGINAL FEATURE SET
# ============================================================

print("\n" + "=" * 75)
print("MODEL A - ORIGINAL FEATURES")
print("=" * 75)

model_a_df = train_df.drop(
    columns=["AGE_GROUP"]
)

X_train_a = model_a_df.drop(
    columns=["default"]
)

X_test_a = test_df.drop(
    columns=["AGE_GROUP", "default"]
)

print("\nNumber of features:")
print(X_train_a.shape[1])


# ============================================================
# TRAIN MODEL A
# ============================================================

model_a = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE
)

print("\nTraining Model A...")

model_a.fit(
    X_train_a,
    y_train
)

print(
    "Model A training completed."
)


# ============================================================
# MODEL B
# REMOVE DEMOGRAPHIC FEATURES
# ============================================================

print("\n" + "=" * 75)
print("MODEL B - DEMOGRAPHIC-REDUCED FEATURES")
print("=" * 75)

features_to_remove = [
    "SEX",
    "AGE"
]

model_b_df = train_df.drop(
    columns=[
        "AGE_GROUP",
        "SEX",
        "AGE"
    ]
)

X_train_b = model_b_df.drop(
    columns=["default"]
)

X_test_b = test_df.drop(
    columns=[
        "AGE_GROUP",
        "SEX",
        "AGE",
        "default"
    ]
)

print("\nRemoved features:")
print(features_to_remove)

print("\nNumber of features:")
print(X_train_b.shape[1])


# ============================================================
# TRAIN MODEL B
# ============================================================

model_b = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE
)

print("\nTraining Model B...")

model_b.fit(
    X_train_b,
    y_train
)

print(
    "Model B training completed."
)


# ============================================================
# SAVE MODELS
# ============================================================

model_a_path = (
    MODELS_DIR
    / "gradient_boosting_original.pkl"
)

model_b_path = (
    MODELS_DIR
    / "gradient_boosting_demographic_reduced.pkl"
)

joblib.dump(
    model_a,
    model_a_path
)

joblib.dump(
    model_b,
    model_b_path
)

print("\nModels saved:")
print(model_a_path)
print(model_b_path)


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_performance(
    model,
    X_test,
    y_test
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "Probabilities": probabilities,

        "Predictions": predictions
    }


# ============================================================
# EVALUATE MODELS
# ============================================================

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)

performance_a = calculate_performance(
    model_a,
    X_test_a,
    y_test
)

performance_b = calculate_performance(
    model_b,
    X_test_b,
    y_test
)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

performance_table = pd.DataFrame([
    {
        "Model": "Original",
        "Accuracy": performance_a["Accuracy"],
        "Precision": performance_a["Precision"],
        "Recall": performance_a["Recall"],
        "F1": performance_a["F1"],
        "ROC-AUC": performance_a["ROC-AUC"]
    },
    {
        "Model": "Demographic Reduced",
        "Accuracy": performance_b["Accuracy"],
        "Precision": performance_b["Precision"],
        "Recall": performance_b["Recall"],
        "F1": performance_b["F1"],
        "ROC-AUC": performance_b["ROC-AUC"]
    }
])

print(
    performance_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# FAIRNESS DATAFRAME
# ============================================================

fairness_test = test_df[
    [
        "SEX",
        "AGE_GROUP"
    ]
].copy()

fairness_test["actual"] = (
    y_test.values
)

fairness_test["original_prediction"] = (
    performance_a["Predictions"]
)

fairness_test["reduced_prediction"] = (
    performance_b["Predictions"]
)


# ============================================================
# GROUP METRICS
# ============================================================

def group_metrics(
    actual,
    predictions
):

    actual = pd.Series(
        actual
    ).reset_index(
        drop=True
    )

    predictions = pd.Series(
        predictions
    ).reset_index(
        drop=True
    )

    tp = (
        ((actual == 1) & (predictions == 1))
        .sum()
    )

    fp = (
        ((actual == 0) & (predictions == 1))
        .sum()
    )

    tn = (
        ((actual == 0) & (predictions == 0))
        .sum()
    )

    fn = (
        ((actual == 1) & (predictions == 0))
        .sum()
    )

    predicted_default_rate = (
        predictions.mean()
    )

    tpr_denominator = (
        tp + fn
    )

    fpr_denominator = (
        fp + tn
    )

    tpr = (
        tp / tpr_denominator
        if tpr_denominator > 0
        else 0
    )

    fpr = (
        fp / fpr_denominator
        if fpr_denominator > 0
        else 0
    )

    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )

    return {
        "Predicted_Default_Rate":
            predicted_default_rate,

        "TPR":
            tpr,

        "FPR":
            fpr,

        "Precision":
            precision
    }


# ============================================================
# FAIRNESS COMPARISON
# ============================================================

def compare_fairness(
    dataframe,
    attribute,
    prediction_column,
    model_name
):

    rows = []

    for group in (
        dataframe[attribute]
        .dropna()
        .unique()
    ):

        group_data = dataframe[
            dataframe[attribute] == group
        ]

        metrics = group_metrics(
            group_data["actual"],
            group_data[prediction_column]
        )

        metrics["Model"] = model_name
        metrics["Attribute"] = attribute
        metrics["Group"] = str(group)

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ORIGINAL MODEL FAIRNESS
# ============================================================

original_sex = compare_fairness(
    fairness_test,
    "SEX",
    "original_prediction",
    "Original"
)

original_age = compare_fairness(
    fairness_test,
    "AGE_GROUP",
    "original_prediction",
    "Original"
)


# ============================================================
# REDUCED MODEL FAIRNESS
# ============================================================

reduced_sex = compare_fairness(
    fairness_test,
    "SEX",
    "reduced_prediction",
    "Demographic Reduced"
)

reduced_age = compare_fairness(
    fairness_test,
    "AGE_GROUP",
    "reduced_prediction",
    "Demographic Reduced"
)


# ============================================================
# COMBINE FAIRNESS RESULTS
# ============================================================

fairness_results = pd.concat(
    [
        original_sex,
        original_age,
        reduced_sex,
        reduced_age
    ],
    ignore_index=True
)


# ============================================================
# DISPLAY FAIRNESS COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FAIRNESS COMPARISON")
print("=" * 75)

print(
    fairness_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CALCULATE DISPARITIES
# ============================================================

def calculate_disparity(
    results,
    attribute,
    model_name
):

    filtered = results[
        (results["Attribute"] == attribute)
        &
        (results["Model"] == model_name)
    ]

    metrics = [
        "Predicted_Default_Rate",
        "TPR",
        "FPR",
        "Precision"
    ]

    rows = []

    for metric in metrics:

        maximum = filtered[
            metric
        ].max()

        minimum = filtered[
            metric
        ].min()

        rows.append({
            "Model": model_name,
            "Attribute": attribute,
            "Metric": metric,
            "Absolute_Difference":
                maximum - minimum
        })

    return pd.DataFrame(
        rows
    )


disparities = pd.concat([
    calculate_disparity(
        fairness_results,
        "SEX",
        "Original"
    ),

    calculate_disparity(
        fairness_results,
        "SEX",
        "Demographic Reduced"
    ),

    calculate_disparity(
        fairness_results,
        "AGE_GROUP",
        "Original"
    ),

    calculate_disparity(
        fairness_results,
        "AGE_GROUP",
        "Demographic Reduced"
    )
], ignore_index=True)


# ============================================================
# DISPLAY DISPARITIES
# ============================================================

print("\n" + "=" * 75)
print("DISPARITY COMPARISON")
print("=" * 75)

print(
    disparities.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

performance_path = (
    REPORTS_DIR
    / "fairness_model_performance_comparison.csv"
)

fairness_path = (
    REPORTS_DIR
    / "fairness_model_group_comparison.csv"
)

disparity_path = (
    REPORTS_DIR
    / "fairness_model_disparity_comparison.csv"
)

performance_table.to_csv(
    performance_path,
    index=False
)

fairness_results.to_csv(
    fairness_path,
    index=False
)

disparities.to_csv(
    disparity_path,
    index=False
)


print("\nReports saved:")
print(performance_path)
print(fairness_path)
print(disparity_path)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("FAIRNESS MITIGATION EXPERIMENT COMPLETED")
print("=" * 75)