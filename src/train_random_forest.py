import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "credit_card_default_features.csv"

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# PREPARE DATA
# ============================================================

df = df.drop(columns=["AGE_GROUP"])

X = df.drop(columns=["default"])
y = df["default"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print("RANDOM FOREST MODEL TRAINING")
print("=" * 60)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)