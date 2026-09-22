import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score
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
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FINAL MODEL THRESHOLD
# ============================================================

FINAL_THRESHOLD = 0.30


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("ENHANCED FAIRNESS AUDIT")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# PREPARE MODEL DATA
# ============================================================

# AGE_GROUP is retained separately for fairness auditing.
# It is not used as a model feature.

df_model = df.drop(
    columns=["AGE_GROUP"]
)

X = df_model.drop(
    columns=["default"]
)

y = df_model["default"]


# ============================================================
# CREATE CONSISTENT TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# GET TEST ROWS FOR DEMOGRAPHIC ATTRIBUTES
# ============================================================

# The same random_state and stratification reproduce
# the corresponding test rows.

_, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
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
# TEST PREDICTIONS
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# BUILD FAIRNESS DATAFRAME
# ============================================================

fairness_df = demographic_test[
    [
        "SEX",
        "AGE_GROUP"
    ]
].copy()

fairness_df["actual"] = y_test.values

fairness_df["predicted"] = predictions

fairness_df["probability"] = probabilities


# ============================================================
# FAIRNESS METRIC FUNCTION
# ============================================================

def calculate_metrics(group_data):

    actual = group_data["actual"]

    predicted = group_data["predicted"]

    true_positive = (
        ((actual == 1) & (predicted == 1))
        .sum()
    )

    false_positive = (
        ((actual == 0) & (predicted == 1))
        .sum()
    )

    true_negative = (
        ((actual == 0) & (predicted == 0))
        .sum()
    )

    false_negative = (
        ((actual == 1) & (predicted == 0))
        .sum()
    )

    total = len(group_data)

    # --------------------------------------------------------
    # Predicted default rate
    # --------------------------------------------------------

    predicted_default_rate = (
        predicted.mean()
    )

    # --------------------------------------------------------
    # True Positive Rate
    # --------------------------------------------------------

    tpr_denominator = (
        true_positive + false_negative
    )

    tpr = (
        true_positive / tpr_denominator
        if tpr_denominator > 0
        else 0
    )

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    fpr_denominator = (
        false_positive + true_negative
    )

    fpr = (
        false_positive / fpr_denominator
        if fpr_denominator > 0
        else 0
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    return {
        "Samples": total,
        "Predicted_Default_Rate": predicted_default_rate,
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "Recall": recall
    }


# ============================================================
# AUDIT ATTRIBUTE
# ============================================================

def audit_attribute(
    dataframe,
    attribute
):

    results = []

    groups = (
        dataframe[attribute]
        .dropna()
        .unique()
    )

    for group in groups:

        group_data = dataframe[
            dataframe[attribute] == group
        ]

        metrics = calculate_metrics(
            group_data
        )

        metrics["Attribute"] = attribute
        metrics["Group"] = str(group)

        results.append(
            metrics
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# RUN AUDITS
# ============================================================

sex_results = audit_attribute(
    fairness_df,
    "SEX"
)

age_results = audit_attribute(
    fairness_df,
    "AGE_GROUP"
)


# ============================================================
# DISPLAY SEX RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FAIRNESS RESULTS: SEX")
print("=" * 75)

print(
    sex_results[
        [
            "Group",
            "Samples",
            "Predicted_Default_Rate",
            "TPR",
            "FPR",
            "Precision",
            "Recall"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# DISPLAY AGE RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FAIRNESS RESULTS: AGE GROUP")
print("=" * 75)

print(
    age_results[
        [
            "Group",
            "Samples",
            "Predicted_Default_Rate",
            "TPR",
            "FPR",
            "Precision",
            "Recall"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = pd.concat(
    [
        sex_results,
        age_results
    ],
    ignore_index=True
)


# ============================================================
# SAVE RAW FAIRNESS RESULTS
# ============================================================

results_path = (
    REPORTS_DIR
    / "fairness_audit_results.csv"
)

all_results.to_csv(
    results_path,
    index=False
)

print("\nFairness results saved to:")
print(results_path)


# ============================================================
# DISPARITY CALCULATION
# ============================================================

def calculate_disparities(
    results,
    attribute
):

    attribute_results = results[
        results["Attribute"] == attribute
    ]

    metrics = [
        "Predicted_Default_Rate",
        "TPR",
        "FPR",
        "Precision"
    ]

    disparity_rows = []

    for metric in metrics:

        maximum = (
            attribute_results[metric]
            .max()
        )

        minimum = (
            attribute_results[metric]
            .min()
        )

        difference = (
            maximum - minimum
        )

        disparity_rows.append({
            "Attribute": attribute,
            "Metric": metric,
            "Maximum": maximum,
            "Minimum": minimum,
            "Absolute_Difference": difference
        })

    return pd.DataFrame(
        disparity_rows
    )


sex_disparities = calculate_disparities(
    all_results,
    "SEX"
)

age_disparities = calculate_disparities(
    all_results,
    "AGE_GROUP"
)

disparity_results = pd.concat(
    [
        sex_disparities,
        age_disparities
    ],
    ignore_index=True
)


# ============================================================
# DISPLAY DISPARITIES
# ============================================================

print("\n" + "=" * 75)
print("FAIRNESS DISPARITY SUMMARY")
print("=" * 75)

print(
    disparity_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE DISPARITIES
# ============================================================

disparity_path = (
    REPORTS_DIR
    / "fairness_disparity_results.csv"
)

disparity_results.to_csv(
    disparity_path,
    index=False
)

print("\nDisparity results saved to:")
print(disparity_path)


# ============================================================
# CREATE FAIRNESS CHARTS
# ============================================================

def create_metric_chart(
    results,
    attribute,
    metric,
    filename,
    title
):

    attribute_results = results[
        results["Attribute"] == attribute
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        attribute_results["Group"],
        attribute_results[metric]
    )

    plt.xlabel(attribute)

    plt.ylabel(metric)

    plt.title(title)

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Chart saved: {output_path}"
    )


# ============================================================
# SEX CHARTS
# ============================================================

create_metric_chart(
    sex_results,
    "SEX",
    "Predicted_Default_Rate",
    "fairness_sex_default_rate.png",
    "Predicted Default Rate by Sex"
)

create_metric_chart(
    sex_results,
    "SEX",
    "TPR",
    "fairness_sex_tpr.png",
    "True Positive Rate by Sex"
)

create_metric_chart(
    sex_results,
    "SEX",
    "FPR",
    "fairness_sex_fpr.png",
    "False Positive Rate by Sex"
)


# ============================================================
# AGE GROUP CHARTS
# ============================================================

create_metric_chart(
    age_results,
    "AGE_GROUP",
    "Predicted_Default_Rate",
    "fairness_age_default_rate.png",
    "Predicted Default Rate by Age Group"
)

create_metric_chart(
    age_results,
    "AGE_GROUP",
    "TPR",
    "fairness_age_tpr.png",
    "True Positive Rate by Age Group"
)

create_metric_chart(
    age_results,
    "AGE_GROUP",
    "FPR",
    "fairness_age_fpr.png",
    "False Positive Rate by Age Group"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("ENHANCED FAIRNESS AUDIT COMPLETED")
print("=" * 75)

import pandas as pd
import numpy as np
import joblib
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

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

AUDIT_RESULTS_PATH = (
    REPORTS_DIR / "fairness_audit_results.csv"
)

DISPARITY_RESULTS_PATH = (
    REPORTS_DIR / "fairness_disparity_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Final threshold selected using validation data
FINAL_THRESHOLD = 0.35


# ============================================================
# CREATE DIRECTORIES
# ============================================================

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    print("=" * 75)
    print("FINAL FAIRNESS AUDIT")
    print("=" * 75)

    print("\nDataset shape:")
    print(df.shape)

    return df


# ============================================================
# CREATE SAME TEST SPLIT
# ============================================================

def create_test_split(df):

    X = df.drop(
        columns=["default", "AGE_GROUP"]
    )

    y = df["default"]

    # Same 60 / 20 / 20 split used during
    # final model creation.

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

    # Keep the demographic columns from the
    # original dataset for fairness auditing.

    demographic_test = df.loc[
        X_test.index
    ].copy()

    return (
        X_test,
        y_test,
        demographic_test
    )


# ============================================================
# LOAD FINAL MODEL
# ============================================================

def load_model():

    print("\nLoading final model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Final Gradient Boosting model loaded successfully."
    )

    return model


# ============================================================
# CALCULATE GROUP METRICS
# ============================================================

def calculate_group_metrics(
    y_true,
    y_pred,
    group_values
):

    results = []

    unique_groups = (
        group_values
        .dropna()
        .unique()
    )

    for group in unique_groups:

        mask = (
            group_values == group
        )

        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]

        tp = (
            (group_y_true == 1) &
            (group_y_pred == 1)
        ).sum()

        tn = (
            (group_y_true == 0) &
            (group_y_pred == 0)
        ).sum()

        fp = (
            (group_y_true == 0) &
            (group_y_pred == 1)
        ).sum()

        fn = (
            (group_y_true == 1) &
            (group_y_pred == 0)
        ).sum()

        samples = len(group_y_true)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        predicted_default_rate = (
            group_y_pred.mean()
        )

        actual_default_rate = (
            group_y_true.mean()
        )

        if (tp + fn) > 0:
            tpr = tp / (tp + fn)
        else:
            tpr = np.nan

        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = np.nan

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = np.nan

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = np.nan

        results.append(
            {
                "Group": group,
                "Samples": samples,
                "Actual_Default_Rate":
                    actual_default_rate,
                "Predicted_Default_Rate":
                    predicted_default_rate,
                "TPR": tpr,
                "FPR": fpr,
                "Precision": precision,
                "Specificity": specificity
            }
        )

    return pd.DataFrame(results)


# ============================================================
# CALCULATE DISPARITIES
# ============================================================

def calculate_disparities(metrics_df):

    numeric_metrics = [
        "Predicted_Default_Rate",
        "TPR",
        "FPR",
        "Precision",
        "Specificity"
    ]

    disparities = []

    for metric in numeric_metrics:

        maximum = metrics_df[
            metric
        ].max()

        minimum = metrics_df[
            metric
        ].min()

        disparity = maximum - minimum

        disparities.append(
            {
                "Metric": metric,
                "Max_Group_Value": maximum,
                "Min_Group_Value": minimum,
                "Absolute_Disparity": disparity
            }
        )

    return pd.DataFrame(disparities)


# ============================================================
# PLOT FAIRNESS METRICS
# ============================================================

def create_fairness_charts(
    metrics_df,
    attribute_name
):

    metrics = [
        (
            "Predicted_Default_Rate",
            "Predicted Default Rate",
            f"fairness_{attribute_name.lower()}_default_rate.png"
        ),
        (
            "TPR",
            "True Positive Rate",
            f"fairness_{attribute_name.lower()}_tpr.png"
        ),
        (
            "FPR",
            "False Positive Rate",
            f"fairness_{attribute_name.lower()}_fpr.png"
        )
    ]

    for column, title, filename in metrics:

        plt.figure(figsize=(8, 5))

        plt.bar(
            metrics_df["Group"].astype(str),
            metrics_df[column]
        )

        plt.title(
            f"{title} by {attribute_name}"
        )

        plt.xlabel(
            attribute_name
        )

        plt.ylabel(
            title
        )

        plt.tight_layout()

        output_path = (
            FIGURES_DIR / filename
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Saved chart: {output_path}"
        )


# ============================================================
# AUDIT ATTRIBUTE
# ============================================================

def audit_attribute(
    y_test,
    predictions,
    demographic_test,
    attribute_name
):

    print("\n")
    print("=" * 75)
    print(f"FAIRNESS RESULTS: {attribute_name}")
    print("=" * 75)

    group_values = (
        demographic_test[attribute_name]
    )

    metrics_df = calculate_group_metrics(
        y_test,
        predictions,
        group_values
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    disparities_df = calculate_disparities(
        metrics_df
    )

    print("\nDisparities:")

    print(
        disparities_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save group results
    # --------------------------------------------------------

    results_with_attribute = metrics_df.copy()

    results_with_attribute.insert(
        0,
        "Attribute",
        attribute_name
    )

    # --------------------------------------------------------
    # Save disparity results
    # --------------------------------------------------------

    disparity_with_attribute = (
        disparities_df.copy()
    )

    disparity_with_attribute.insert(
        0,
        "Attribute",
        attribute_name
    )

    # --------------------------------------------------------
    # Create charts
    # --------------------------------------------------------

    create_fairness_charts(
        metrics_df,
        attribute_name
    )

    return (
        results_with_attribute,
        disparity_with_attribute
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Create exact test split
    # --------------------------------------------------------

    (
        X_test,
        y_test,
        demographic_test
    ) = create_test_split(df)

    print("\nTest dataset:")
    print(X_test.shape)

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Apply final threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    print("\nFinal threshold:")
    print(FINAL_THRESHOLD)

    print("\nPrediction distribution:")

    print(
        pd.Series(predictions)
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Audit SEX
    # --------------------------------------------------------

    (
        sex_results,
        sex_disparities
    ) = audit_attribute(
        y_test,
        predictions,
        demographic_test,
        "SEX"
    )

    # --------------------------------------------------------
    # Audit AGE_GROUP
    # --------------------------------------------------------

    (
        age_results,
        age_disparities
    ) = audit_attribute(
        y_test,
        predictions,
        demographic_test,
        "AGE_GROUP"
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    all_results = pd.concat(
        [
            sex_results,
            age_results
        ],
        ignore_index=True
    )

    all_disparities = pd.concat(
        [
            sex_disparities,
            age_disparities
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    all_results.to_csv(
        AUDIT_RESULTS_PATH,
        index=False
    )

    all_disparities.to_csv(
        DISPARITY_RESULTS_PATH,
        index=False
    )

    print("\n")
    print("=" * 75)
    print("FAIRNESS AUDIT COMPLETED")
    print("=" * 75)

    print("\nAudit results saved to:")
    print(AUDIT_RESULTS_PATH)

    print("\nDisparity results saved to:")
    print(DISPARITY_RESULTS_PATH)

    print("\nImportant:")
    print(
        "Fairness metrics measure differences between "
        "groups. They do not by themselves establish "
        "whether discrimination occurred."
    )