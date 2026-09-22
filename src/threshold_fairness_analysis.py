import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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
    / "gradient_boosting_model.pkl"
)

REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

THRESHOLDS = [
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("THRESHOLD VS FAIRNESS ANALYSIS")
print("=" * 75)

df = pd.read_csv(
    DATA_PATH
)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# MODEL DATA
# ============================================================

# AGE_GROUP is kept separately for fairness auditing.

df_model = df.drop(
    columns=["AGE_GROUP"]
)

X = df_model.drop(
    columns=["default"]
)

y = df_model["default"]


# ============================================================
# CONSISTENT TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# DEMOGRAPHIC TEST DATA
# ============================================================

_, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["default"]
)

demographic_test = df.loc[
    test_indices
].copy()

demographic_test = demographic_test.reset_index(
    drop=True
)

y_test = y_test.reset_index(
    drop=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Gradient Boosting model loaded successfully."
)


# ============================================================
# MODEL PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# FAIRNESS DATA
# ============================================================

fairness_base = demographic_test[
    [
        "SEX",
        "AGE_GROUP"
    ]
].copy()

fairness_base["actual"] = (
    y_test.values
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_group_metrics(
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
# RUN THRESHOLD ANALYSIS
# ============================================================

all_rows = []

print("\n" + "=" * 75)
print("THRESHOLD PERFORMANCE")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1':<15}"
)

print("-" * 75)


for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    print(
        f"{threshold:<12.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1:<15.4f}"
    )

    # --------------------------------------------------------
    # Store SEX group metrics
    # --------------------------------------------------------

    sex_data = fairness_base.copy()

    sex_data["prediction"] = predictions

    for group in (
        sex_data["SEX"]
        .dropna()
        .unique()
    ):

        group_data = sex_data[
            sex_data["SEX"] == group
        ]

        metrics = calculate_group_metrics(
            group_data["actual"],
            group_data["prediction"]
        )

        metrics["Threshold"] = threshold
        metrics["Attribute"] = "SEX"
        metrics["Group"] = str(group)

        all_rows.append(
            metrics
        )

    # --------------------------------------------------------
    # Store AGE_GROUP metrics
    # --------------------------------------------------------

    age_data = fairness_base.copy()

    age_data["prediction"] = predictions

    for group in (
        age_data["AGE_GROUP"]
        .dropna()
        .unique()
    ):

        group_data = age_data[
            age_data["AGE_GROUP"] == group
        ]

        metrics = calculate_group_metrics(
            group_data["actual"],
            group_data["prediction"]
        )

        metrics["Threshold"] = threshold
        metrics["Attribute"] = "AGE_GROUP"
        metrics["Group"] = str(group)

        all_rows.append(
            metrics
        )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_rows
)

results_df = results_df[
    [
        "Threshold",
        "Attribute",
        "Group",
        "Predicted_Default_Rate",
        "TPR",
        "FPR",
        "Precision"
    ]
]


# ============================================================
# CALCULATE DISPARITIES
# ============================================================

disparity_rows = []

for threshold in THRESHOLDS:

    threshold_data = results_df[
        results_df["Threshold"] == threshold
    ]

    for attribute in [
        "SEX",
        "AGE_GROUP"
    ]:

        attribute_data = threshold_data[
            threshold_data["Attribute"]
            == attribute
        ]

        metrics = [
            "Predicted_Default_Rate",
            "TPR",
            "FPR",
            "Precision"
        ]

        for metric in metrics:

            maximum = (
                attribute_data[metric]
                .max()
            )

            minimum = (
                attribute_data[metric]
                .min()
            )

            disparity = (
                maximum - minimum
            )

            disparity_rows.append({
                "Threshold": threshold,
                "Attribute": attribute,
                "Metric": metric,
                "Absolute_Difference":
                    disparity
            })


disparity_df = pd.DataFrame(
    disparity_rows
)


# ============================================================
# DISPLAY THRESHOLD 0.30
# ============================================================

print("\n" + "=" * 75)
print("FAIRNESS METRICS AT FINAL THRESHOLD = 0.30")
print("=" * 75)

final_threshold_results = results_df[
    results_df["Threshold"] == 0.30
]

print(
    final_threshold_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# DISPLAY DISPARITY AT 0.30
# ============================================================

print("\n" + "=" * 75)
print("DISPARITIES AT FINAL THRESHOLD = 0.30")
print("=" * 75)

final_disparities = disparity_df[
    disparity_df["Threshold"] == 0.30
]

print(
    final_disparities.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_path = (
    REPORTS_DIR
    / "threshold_fairness_metrics.csv"
)

disparity_path = (
    REPORTS_DIR
    / "threshold_fairness_disparities.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)

disparity_df.to_csv(
    disparity_path,
    index=False
)

print("\nResults saved:")
print(metrics_path)
print(disparity_path)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("THRESHOLD VS FAIRNESS ANALYSIS COMPLETED")
print("=" * 75)