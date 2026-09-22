import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Credit Default Risk Scorer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_gradient_boosting_model.pkl"
)

FAIRNESS_PATH = (
    BASE_DIR
    / "reports"
    / "fairness_audit_results.csv"
)

FAIRNESS_DISPARITY_PATH = (
    BASE_DIR
    / "reports"
    / "fairness_disparity_results.csv"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

FINAL_THRESHOLD = 0.35

MODEL_ACCURACY = 0.8108
MODEL_PRECISION = 0.5871
MODEL_RECALL = 0.4879
MODEL_F1 = 0.5329
MODEL_ROC_AUC = 0.7919

TRAINING_RECORDS = 23972
TEST_RECORDS = 5993
FEATURE_COUNT = 33


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# LOAD FAIRNESS RESULTS
# ============================================================

@st.cache_data
def load_fairness_results():

    if not FAIRNESS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        FAIRNESS_PATH
    )


@st.cache_data
def load_fairness_disparities():

    if not FAIRNESS_DISPARITY_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        FAIRNESS_DISPARITY_PATH
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_customer_features(
    limit_bal,
    sex,
    education,
    marriage,
    age,
    pay_0,
    pay_2,
    pay_3,
    pay_4,
    pay_5,
    pay_6,
    bill_amt1,
    bill_amt2,
    bill_amt3,
    bill_amt4,
    bill_amt5,
    bill_amt6,
    pay_amt1,
    pay_amt2,
    pay_amt3,
    pay_amt4,
    pay_amt5,
    pay_amt6
):

    bill_values = [
        bill_amt1,
        bill_amt2,
        bill_amt3,
        bill_amt4,
        bill_amt5,
        bill_amt6
    ]

    payment_values = [
        pay_amt1,
        pay_amt2,
        pay_amt3,
        pay_amt4,
        pay_amt5,
        pay_amt6
    ]

    pay_status_values = [
        pay_0,
        pay_2,
        pay_3,
        pay_4,
        pay_5,
        pay_6
    ]

    avg_bill_amt = np.mean(
        bill_values
    )

    avg_pay_amt = np.mean(
        payment_values
    )

    max_bill_amt = np.max(
        bill_values
    )

    max_pay_amt = np.max(
        payment_values
    )

    if limit_bal != 0:

        credit_utilization = (
            avg_bill_amt /
            limit_bal
        )

    else:

        credit_utilization = 0

    total_payment = np.sum(
        payment_values
    )

    total_bill = np.sum(
        bill_values
    )

    if total_bill != 0:

        payment_to_bill_ratio = (
            total_payment /
            total_bill
        )

    else:

        payment_to_bill_ratio = 0

    avg_payment_status = np.mean(
        pay_status_values
    )

    num_delayed_payments = np.sum(
        np.array(pay_status_values) > 0
    )

    data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,

        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,

        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,

        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6,

        "AVG_BILL_AMT": avg_bill_amt,
        "AVG_PAY_AMT": avg_pay_amt,
        "MAX_BILL_AMT": max_bill_amt,
        "MAX_PAY_AMT": max_pay_amt,
        "CREDIT_UTILIZATION": credit_utilization,
        "TOTAL_PAYMENT": total_payment,
        "TOTAL_BILL": total_bill,
        "PAYMENT_TO_BILL_RATIO": payment_to_bill_ratio,
        "AVG_PAYMENT_STATUS": avg_payment_status,
        "NUM_DELAYED_PAYMENTS": num_delayed_payments
    }

    return pd.DataFrame(
        [data]
    )


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(probability):

    if probability < 0.20:

        return (
            "Low Risk",
            "The predicted probability is below 20%."
        )

    elif probability < 0.35:

        return (
            "Moderate Risk",
            "The predicted probability is below the model threshold."
        )

    elif probability < 0.60:

        return (
            "High Risk",
            "The predicted probability is above the model threshold."
        )

    else:

        return (
            "Very High Risk",
            "The predicted probability is substantially above the model threshold."
        )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_explanation(
    model,
    input_data
):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        input_data
    )

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[0]

    return shap_values[0]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "💳 Loan Risk Scorer"
    )

    st.markdown(
        """
        ### Navigation

        Use the sections below to explore:

        - Risk Assessment
        - Explainability
        - Fairness Audit
        - Model Information
        """
    )

    st.divider()

    st.info(
        """
        **Final Model**

        Gradient Boosting Classifier

        **Threshold:** 0.35

        **ROC-AUC:** 0.7919
        """
    )

    st.divider()

    st.caption(
        "Educational ML project — not a real lending decision system."
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "💳 Loan / Credit Default Risk Scorer"
)

st.markdown(
    """
    **Machine Learning + Explainable AI + Fairness Auditing**

    This application estimates the probability that a customer
    may default on a credit payment and provides explanations
    for the model's prediction.
    """
)

st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Risk Assessment",
        "🔍 Explainability",
        "⚖️ Fairness Audit",
        "📊 Model Information"
    ]
)


# ============================================================
# TAB 1 — RISK ASSESSMENT
# ============================================================

with tab1:

    st.header(
        "Customer Risk Assessment"
    )

    st.write(
        "Enter customer financial and repayment information."
    )

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    st.subheader(
        "Customer Profile"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        limit_bal = st.number_input(
            "Credit Limit",
            min_value=1000.0,
            max_value=1000000.0,
            value=50000.0,
            step=5000.0
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            options=[1, 2],
            format_func=lambda x:
                "Male" if x == 1 else "Female"
        )

    with col3:

        education = st.selectbox(
            "Education",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate School",
                2: "University",
                3: "High School",
                4: "Other"
            }[x]
        )

    with col4:

        marriage = st.selectbox(
            "Marriage",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Married",
                2: "Single",
                3: "Other"
            }[x]
        )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    # --------------------------------------------------------
    # REPAYMENT HISTORY
    # --------------------------------------------------------

    st.subheader(
        "Repayment History"
    )

    st.caption(
        "Use the dataset's repayment status encoding."
    )

    pay_col1, pay_col2, pay_col3 = st.columns(3)

    with pay_col1:

        pay_0 = st.number_input(
            "PAY_0",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_2 = st.number_input(
            "PAY_2",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

    with pay_col2:

        pay_3 = st.number_input(
            "PAY_3",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_4 = st.number_input(
            "PAY_4",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

    with pay_col3:

        pay_5 = st.number_input(
            "PAY_5",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_6 = st.number_input(
            "PAY_6",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

    # --------------------------------------------------------
    # BILL AMOUNTS
    # --------------------------------------------------------

    st.subheader(
        "Bill Amounts"
    )

    bill_col1, bill_col2, bill_col3 = st.columns(3)

    with bill_col1:

        bill_amt1 = st.number_input(
            "BILL_AMT1",
            value=33243.0,
            step=1000.0
        )

        bill_amt2 = st.number_input(
            "BILL_AMT2",
            value=32500.0,
            step=1000.0
        )

    with bill_col2:

        bill_amt3 = st.number_input(
            "BILL_AMT3",
            value=32000.0,
            step=1000.0
        )

        bill_amt4 = st.number_input(
            "BILL_AMT4",
            value=2000.0,
            step=1000.0
        )

    with bill_col3:

        bill_amt5 = st.number_input(
            "BILL_AMT5",
            value=1800.0,
            step=1000.0
        )

        bill_amt6 = st.number_input(
            "BILL_AMT6",
            value=1500.0,
            step=1000.0
        )

    # --------------------------------------------------------
    # PAYMENT AMOUNTS
    # --------------------------------------------------------

    st.subheader(
        "Payment Amounts"
    )

    payment_col1, payment_col2, payment_col3 = st.columns(3)

    with payment_col1:

        pay_amt1 = st.number_input(
            "PAY_AMT1",
            value=1000.0,
            step=500.0
        )

        pay_amt2 = st.number_input(
            "PAY_AMT2",
            value=1000.0,
            step=500.0
        )

    with payment_col2:

        pay_amt3 = st.number_input(
            "PAY_AMT3",
            value=0.0,
            step=500.0
        )

        pay_amt4 = st.number_input(
            "PAY_AMT4",
            value=0.0,
            step=500.0
        )

    with payment_col3:

        pay_amt5 = st.number_input(
            "PAY_AMT5",
            value=1000.0,
            step=500.0
        )

        pay_amt6 = st.number_input(
            "PAY_AMT6",
            value=1000.0,
            step=500.0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Assess Credit Risk",
        type="primary",
        width="stretch"
    )

    if predict_button:

        model = load_model()

        input_data = create_customer_features(
            limit_bal,
            sex,
            education,
            marriage,
            age,
            pay_0,
            pay_2,
            pay_3,
            pay_4,
            pay_5,
            pay_6,
            bill_amt1,
            bill_amt2,
            bill_amt3,
            bill_amt4,
            bill_amt5,
            bill_amt6,
            pay_amt1,
            pay_amt2,
            pay_amt3,
            pay_amt4,
            pay_amt5,
            pay_amt6
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = int(
            probability >= FINAL_THRESHOLD
        )

        risk_band, risk_description = (
            get_risk_band(
                probability
            )
        )

        # Save in session state
        st.session_state[
            "input_data"
        ] = input_data

        st.session_state[
            "probability"
        ] = probability

        st.session_state[
            "prediction"
        ] = prediction

        st.session_state[
            "risk_band"
        ] = risk_band

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Risk Assessment Result"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Default Probability",
                f"{probability:.2%}"
            )

        with result_col2:

            st.metric(
                "Decision Threshold",
                f"{FINAL_THRESHOLD:.0%}"
            )

        with result_col3:

            st.metric(
                "Prediction",
                "DEFAULT"
                if prediction == 1
                else "NON-DEFAULT"
            )

        st.progress(
            min(probability, 1.0)
        )

        if prediction == 1:

            st.error(
                f"⚠️ **{risk_band}**\n\n"
                f"{risk_description}"
            )

        else:

            st.success(
                f"✅ **{risk_band}**\n\n"
                f"{risk_description}"
            )

        st.warning(
            """
            This prediction is generated by a machine learning
            model for educational and demonstration purposes.
            It should not be used as the sole basis for an actual
            lending or credit decision.
            """
        )


# ============================================================
# TAB 2 — EXPLAINABILITY
# ============================================================

with tab2:

    st.header(
        "🔍 Model Explainability"
    )

    if (
        "input_data"
        not in st.session_state
    ):

        st.info(
            "Run a risk assessment first to see the SHAP explanation."
        )

    else:

        model = load_model()

        input_data = (
            st.session_state[
                "input_data"
            ]
        )

        probability = (
            st.session_state[
                "probability"
            ]
        )

        st.metric(
            "Current Default Probability",
            f"{probability:.2%}"
        )

        st.write(
            """
            SHAP shows how individual features contributed to
            this particular prediction.
            """
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_values = get_shap_explanation(
            model,
            input_data
        )

        explanation_df = pd.DataFrame(
            {
                "Feature": input_data.columns,
                "SHAP Value": shap_values,
                "Feature Value": input_data.iloc[0].values
            }
        )

        explanation_df[
            "Absolute SHAP"
        ] = explanation_df[
            "SHAP Value"
        ].abs()

        explanation_df = (
            explanation_df
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
            .head(10)
        )

        st.subheader(
            "Top Prediction Factors"
        )

        chart_df = (
            explanation_df[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ]
            .set_index("Feature")
        )

        st.bar_chart(
            chart_df,
            horizontal=True
        )

        st.dataframe(
            explanation_df[
                [
                    "Feature",
                    "SHAP Value",
                    "Feature Value"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        st.info(
            """
            **How to interpret SHAP values**

            Positive SHAP values push the prediction toward
            default.

            Negative SHAP values push the prediction toward
            non-default.

            The larger the absolute value, the stronger the
            contribution of that feature for this prediction.
            """
        )


# ============================================================
# TAB 3 — FAIRNESS AUDIT
# ============================================================

with tab3:

    st.header(
        "⚖️ Fairness Audit"
    )

    st.write(
        """
        The fairness audit measures whether model outcomes
        differ across selected demographic groups.
        """
    )

    fairness_df = load_fairness_results()

    disparity_df = (
        load_fairness_disparities()
    )

    if fairness_df.empty:

        st.warning(
            "Fairness audit results were not found. "
            "Run fairness_audit.py first."
        )

    else:

        # ----------------------------------------------------
        # Audit results
        # ----------------------------------------------------

        st.subheader(
            "Group-Level Metrics"
        )

        display_columns = [
            "Attribute",
            "Group",
            "Samples",
            "Actual_Default_Rate",
            "Predicted_Default_Rate",
            "TPR",
            "FPR",
            "Precision"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in fairness_df.columns
        ]

        st.dataframe(
            fairness_df[
                available_columns
            ],
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # Disparities
        # ----------------------------------------------------

        if not disparity_df.empty:

            st.subheader(
                "Measured Disparities"
            )

            st.dataframe(
                disparity_df,
                width="stretch",
                hide_index=True
            )

        st.info(
            """
            **Important:** A difference between group metrics
            is a measured disparity. It does not, by itself,
            establish that discrimination occurred.

            Fairness assessment depends on the application,
            context, protected attributes, legal requirements,
            data quality, and the fairness criteria selected.
            """
        )


# ============================================================
# TAB 4 — MODEL INFORMATION
# ============================================================

with tab4:

    st.header(
        "📊 Model Information"
    )

    st.subheader(
        "Final Model"
    )

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:

        st.metric(
            "Algorithm",
            "Gradient Boosting"
        )

        st.metric(
            "Training Records",
            f"{TRAINING_RECORDS:,}"
        )

    with model_col2:

        st.metric(
            "Test Records",
            f"{TEST_RECORDS:,}"
        )

        st.metric(
            "Features",
            FEATURE_COUNT
        )

    with model_col3:

        st.metric(
            "Threshold",
            f"{FINAL_THRESHOLD:.0%}"
        )

        st.metric(
            "ROC-AUC",
            f"{MODEL_ROC_AUC:.4f}"
        )

    st.divider()

    st.subheader(
        "Performance"
    )

    performance_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Value": [
                MODEL_ACCURACY,
                MODEL_PRECISION,
                MODEL_RECALL,
                MODEL_F1,
                MODEL_ROC_AUC
            ]
        }
    )

    performance_display = (
        performance_df.copy()
    )

    performance_display[
        "Value"
    ] = performance_display[
        "Value"
    ].map(
        lambda x: f"{x:.4f}"
    )

    st.dataframe(
        performance_display,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Project Pipeline"
    )

    st.markdown(
        """
        ```text
        Raw Credit Dataset
                ↓
        Data Cleaning
                ↓
        Feature Engineering
                ↓
        Train / Validation / Test Split
                ↓
        Gradient Boosting
                ↓
        Threshold Selection
                ↓
        Final Threshold = 0.35
                ↓
        Final Model
                ↓
        ┌──────────┬──────────┬─────────────┐
        ↓          ↓          ↓
        SHAP      LIME    Fairness Audit
        ↓          ↓          ↓
        └──────────┴──────────┴─────────────┘
                       ↓
                  Streamlit App
        ```
        """
    )

    st.divider()

    st.subheader(
        "Technologies"
    )

    st.write(
        """
        **Python • Pandas • NumPy • Scikit-learn • SHAP • LIME •
        Matplotlib • Streamlit • Joblib**
        """
    )

    st.warning(
        """
        **Disclaimer**

        This application is a machine learning demonstration
        created for educational and portfolio purposes.

        It is not a financial service and should not be used
        as the sole basis for approving or rejecting real
        credit applications.
        """
    )