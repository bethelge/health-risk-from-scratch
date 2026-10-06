import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from models.naive_bayes_scratch import GaussianNaiveBayesScratch

ROOT = Path(__file__).resolve().parent.parent

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "has_disease"

train_df = pd.read_csv(ROOT / "data/processed/train.csv")
test_df = pd.read_csv(ROOT / "data/processed/test.csv")

X_train = train_df[FEATURES].to_numpy()
y_train = train_df[TARGET].to_numpy()
X_test = test_df[FEATURES].to_numpy()
y_test = test_df[TARGET].to_numpy()

# --- Your from-scratch model ---
scratch_model = GaussianNaiveBayesScratch()
scratch_model.fit(X_train, y_train)
scratch_preds = scratch_model.predict(X_test)
scratch_acc = accuracy_score(y_test, scratch_preds)

# --- scikit-learn's reference implementation ---
sklearn_model = GaussianNB()
sklearn_model.fit(X_train, y_train)
sklearn_preds = sklearn_model.predict(X_test)
sklearn_acc = accuracy_score(y_test, sklearn_preds)

print(f"Scratch  accuracy: {scratch_acc:.4f}")
print(f"sklearn  accuracy: {sklearn_acc:.4f}")
print(f"Difference:        {abs(scratch_acc - sklearn_acc):.4f}")

# --- Compare predictions directly, not just accuracy ---
agreement = (scratch_preds == sklearn_preds).mean()
print(f"\nPrediction agreement (scratch vs sklearn): {agreement:.4f}")
disagreements = np.where(scratch_preds != sklearn_preds)[0]
print(f"Number of patients where predictions differ: {len(disagreements)}")

# --- Compare learned parameters directly ---
# sklearn stores means in theta_, variances in var_
print("\nMean absolute difference in learned means (per class, per feature):")
print(np.abs(scratch_model.mean_ - sklearn_model.theta_).mean())
print("\nMean absolute difference in learned variances:")
print(np.abs(scratch_model.var_ - sklearn_model.var_).mean())
