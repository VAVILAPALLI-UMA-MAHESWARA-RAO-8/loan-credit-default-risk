# 💳 Loan / Credit Default Risk Scorer with Explainable AI & Fairness Auditing

An end-to-end machine learning system that predicts **credit card default risk** and provides **Explainable AI (XAI)** and **fairness auditing** to make model behavior more transparent.

The project compares multiple machine learning models, performs threshold optimization, evaluates the final model on an untouched test set, explains predictions using SHAP and LIME, audits demographic group differences, and provides an interactive Streamlit application.

---

## 🚀 Key Features

* Credit default risk prediction
* Data cleaning and preprocessing
* Financial and repayment behavior feature engineering
* Logistic Regression, Random Forest, and Gradient Boosting comparison
* Classification threshold optimization
* Final model training using train + validation data
* SHAP global and individual explanations
* LIME individual prediction explanations
* Fairness auditing across `SEX` and `AGE_GROUP`
* Fairness mitigation experiment
* Interactive Streamlit dashboard
* Responsible AI considerations and model limitations

---

## 🎯 Problem Statement

Credit risk models are commonly used to estimate whether a customer may default on a payment.

However, evaluating a model only by accuracy is not sufficient. A practical credit-risk system should also consider:

* How well the model identifies potential defaults
* How individual predictions can be explained
* Whether model outcomes differ across demographic groups
* How the model behaves when the classification threshold changes

This project addresses these requirements by combining **machine learning, explainability, and fairness auditing** into a single application.

---

## 🏗️ System Architecture

```text
                    Credit Card Dataset
                            │
                            ▼
                     Data Cleaning
                            │
                            ▼
                  Feature Engineering
                            │
                            ▼
                Train / Validation / Test
                            │
                            ▼
                    Model Training
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
       Logistic        Random Forest   Gradient Boosting
       Regression
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    Model Comparison
                            │
                            ▼
                  Threshold Optimization
                            │
                            ▼
                      Final Model
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
           SHAP            LIME      Fairness Audit
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    Streamlit Application
```

---

# 📊 Dataset

This project uses the **Default of Credit Card Clients** dataset.

The dataset contains information about credit card customers, including:

* Credit limit
* Sex
* Education
* Marriage
* Age
* Repayment status
* Bill amounts
* Payment amounts
* Default status

### Dataset statistics

| Description            |     Value |
| ---------------------- | --------: |
| Raw records            |    30,000 |
| Records after cleaning |    29,965 |
| Input features         |        33 |
| Test records           |     5,993 |
| Target                 | `default` |

### Target

```text
0 → Non-default
1 → Default
```

---

# ⚙️ Feature Engineering

The project creates additional features to capture financial and repayment behavior.

### Engineered Features

```text
AVG_BILL_AMT
AVG_PAY_AMT
MAX_BILL_AMT
MAX_PAY_AMT
CREDIT_UTILIZATION
TOTAL_PAYMENT
TOTAL_BILL
PAYMENT_TO_BILL_RATIO
AVG_PAYMENT_STATUS
NUM_DELAYED_PAYMENTS
AGE_GROUP
```

Examples:

### Credit Utilization

```text
Average Bill Amount / Credit Limit
```

This represents how much of the available credit is being utilized.

### Number of Delayed Payments

Counts the number of repayment-status variables indicating delayed payment behavior.

---

# 🤖 Machine Learning Models

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Baseline Model Comparison

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Gradient Boosting   |   81.76% |    66.91% | 34.77% | 45.76% |  77.69% |
| Random Forest       |   77.31% |    48.92% | 57.99% | 53.07% |  77.36% |
| Logistic Regression |   81.11% |    67.08% | 28.73% | 40.23% |  74.87% |

The baseline comparison was followed by threshold analysis and final model training.

---

# 🎚️ Threshold Optimization

Instead of automatically using a classification threshold of `0.50`, multiple thresholds were evaluated on the validation dataset.

The selected threshold was:

```text
0.35
```

The threshold was selected based on the validation F1 score.

After selecting the threshold:

```text
Training + Validation
        ↓
Final Model
        ↓
Untouched Test Set
```

The test set was not used to select the threshold.

---

# 🏆 Final Model

The final model is a:

**Gradient Boosting Classifier**

Configuration:

```text
n_estimators = 200
learning_rate = 0.05
max_depth = 3
random_state = 42
```

The final model was trained using:

```text
23,972 records
```

and evaluated on:

```text
5,993 untouched test records
```

---

# 📈 Final Model Performance

| Metric                   |      Score |
| ------------------------ | ---------: |
| Accuracy                 | **81.08%** |
| Precision                | **58.71%** |
| Recall                   | **48.79%** |
| F1 Score                 | **53.29%** |
| ROC-AUC                  | **79.19%** |
| Classification Threshold |   **0.35** |

### Confusion Matrix

```text
                 Predicted
                 Non-Default   Default

Actual Non-Default    4212       455
Actual Default         679       647
```

### Interpretation

* **True Negatives:** 4,212
* **False Positives:** 455
* **False Negatives:** 679
* **True Positives:** 647

---

# 🔍 Explainable AI

The project uses two complementary Explainable AI techniques:

## SHAP

SHAP is used to understand:

* Global feature importance
* Individual prediction contributions
* Features pushing predictions toward default
* Features pushing predictions toward non-default

### Top Global SHAP Features

| Rank | Feature                 |
| ---: | ----------------------- |
|    1 | `PAY_0`                 |
|    2 | `NUM_DELAYED_PAYMENTS`  |
|    3 | `CREDIT_UTILIZATION`    |
|    4 | `BILL_AMT1`             |
|    5 | `LIMIT_BAL`             |
|    6 | `MAX_BILL_AMT`          |
|    7 | `AVG_PAYMENT_STATUS`    |
|    8 | `PAYMENT_TO_BILL_RATIO` |
|    9 | `PAY_AMT3`              |
|   10 | `PAY_AMT1`              |

These results indicate that repayment behavior and credit utilization are important contributors to the model's predictions.

---

## LIME

LIME is used to explain an individual prediction by approximating the model locally around a specific customer.

Example:

```text
Default Probability: 11.45%
Threshold: 35%
Prediction: NON-DEFAULT
```

The explanation identifies the features that contributed positively or negatively to this particular prediction.

---

# ⚖️ Fairness Auditing

The model was audited across:

* `SEX`
* `AGE_GROUP`

The following group-level metrics were evaluated:

* Predicted Default Rate
* True Positive Rate (TPR)
* False Positive Rate (FPR)
* Precision

Absolute disparities between groups were also calculated.

### Why Fairness Auditing?

A model can have good overall performance while producing different outcomes across demographic groups.

Therefore, this project treats fairness as a separate evaluation dimension rather than assuming that overall model performance automatically implies fairness.

### Important Note

A measured difference between demographic groups does **not by itself establish discrimination**.

Fairness interpretation depends on:

* The application context
* The fairness definition being used
* Data quality
* Protected attributes
* Error costs
* Legal and regulatory requirements

---

# 🧪 Fairness Mitigation Experiment

An additional Gradient Boosting model was trained after removing:

```text
SEX
AGE
```

The purpose was to investigate whether removing demographic variables automatically eliminates fairness disparities.

The experiment demonstrated an important Responsible AI concept:

> Removing demographic variables does not necessarily guarantee fairness.

Other financial and behavioral variables can contain information correlated with demographic characteristics.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard.

### 🎯 Risk Assessment

Users can enter:

* Credit limit
* Age
* Education
* Marriage
* Repayment history
* Bill amounts
* Payment amounts

The application calculates the engineered features automatically and generates a default probability.

---

### 🔍 Explainability

The application displays SHAP-based feature contributions for the current prediction.

---

### ⚖️ Fairness Audit

The dashboard displays:

* Group-level fairness metrics
* Measured disparities
* SEX audit
* AGE_GROUP audit

---

### 📊 Model Information

The application displays:

* Final model
* Dataset size
* Number of features
* Classification threshold
* Performance metrics
* Technology stack

---

# 📂 Project Structure

```text
loan-credit-default-risk/
│
├── data/
│   ├── default_of_credit_card_clients.xls
│   ├── credit_card_default_clean.csv
│   └── credit_card_default_features.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── gradient_boosting_original.pkl
│   ├── gradient_boosting_demographic_reduced.pkl
│   └── final_gradient_boosting_model.pkl
│
├── reports/
│   ├── figures/
│   ├── fairness_audit_results.csv
│   ├── fairness_disparity_results.csv
│   ├── final_lime_individual_customer.csv
│   └── final_lime_individual_customer.html
│
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── train_random_forest.py
│   ├── train_gradient_boosting.py
│   ├── model_comparison.py
│   ├── threshold_analysis.py
│   ├── shap_analysis.py
│   ├── lime_analysis.py
│   ├── fairness_audit.py
│   ├── fairness_mitigation.py
│   ├── threshold_fairness_analysis.py
│   └── finalize_model.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Technology Stack

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| Python       | Programming language        |
| Pandas       | Data processing             |
| NumPy        | Numerical computation       |
| Scikit-learn | Machine learning            |
| SHAP         | Model explainability        |
| LIME         | Local explainability        |
| Matplotlib   | Visualization               |
| Streamlit    | Interactive web application |
| Joblib       | Model serialization         |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd loan-credit-default-risk
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open at:

```text
http://localhost:8501
```

---

# 🔄 Run the ML Pipeline

### Data Cleaning

```bash
python src/data_cleaning.py
```

### Feature Engineering

```bash
python src/feature_engineering.py
```

### Model Training

```bash
python src/train_model.py
```

### Random Forest

```bash
python src/train_random_forest.py
```

### Gradient Boosting

```bash
python src/train_gradient_boosting.py
```

### Model Comparison

```bash
python src/model_comparison.py
```

### Threshold Analysis

```bash
python src/threshold_analysis.py
```

### Final Model

```bash
python src/finalize_model.py
```

### SHAP Analysis

```bash
python src/shap_analysis.py
```

### LIME Analysis

```bash
python src/lime_analysis.py
```

### Fairness Audit

```bash
python src/fairness_audit.py
```

### Fairness Mitigation Experiment

```bash
python src/fairness_mitigation.py
```

---

# 🔐 Responsible AI Considerations

This project incorporates several Responsible AI practices:

* Separate training, validation, and test datasets
* Validation-based threshold selection
* Final evaluation on an untouched test set
* Individual prediction explanations
* Global model explainability
* Demographic fairness auditing
* Fairness mitigation experimentation
* Transparent reporting of limitations

---

# ⚠️ Limitations

This project is intended for **educational and portfolio purposes**.

It is not a production lending system.

Important limitations include:

* Historical datasets may contain existing biases.
* Dataset characteristics may not represent current lending populations.
* Fairness metrics can provide different conclusions depending on the selected definition.
* Removing demographic features does not guarantee fairness.
* The selected threshold depends on the chosen optimization objective.
* The model has not been validated for real-world lending decisions.
* Predictions should not be treated as financial advice.

---

# 🔮 Future Improvements

Possible extensions include:

* Probability calibration
* Cost-sensitive learning
* Additional fairness metrics
* Advanced fairness mitigation techniques
* Model monitoring
* Data drift detection
* Automated fairness monitoring
* REST API deployment
* Cloud deployment
* Model versioning
* Authentication and access control

---

# 👨‍💻 Author

**Uma Maheswara Rao**

B.Tech — Computer Science Engineering
Artificial Intelligence & Machine Learning

---

# ⭐ Project Summary

This project demonstrates an end-to-end approach to building a responsible machine learning system:

```text
Data
 ↓
Cleaning
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Threshold Optimization
 ↓
Final Model
 ↓
SHAP + LIME
 ↓
Fairness Audit
 ↓
Streamlit Application
```

The project demonstrates practical skills in **Machine Learning, Explainable AI, Responsible AI, Python, Scikit-learn, SHAP, LIME, and Streamlit**.

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only.

The predictions generated by this application should not be used as the sole basis for approving or rejecting real credit applications.
