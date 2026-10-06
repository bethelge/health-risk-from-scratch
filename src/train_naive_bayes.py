import pandas as pd
import numpy as np
from pathlib import Path
from models.naive_bayes_scratch import GaussianNaiveBayesScratch

ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "has_disease"

train_df = pd.read_csv(ROOT / "data/processed/train.csv")
test_df = pd.read_csv(ROOT / "data/processed/test.csv")

X_train = train_df[FEATURES].to_numpy()
y_train = train_df[TARGET].to_numpy()
X_test = test_df[FEATURES].to_numpy()
y_test = test_df[TARGET].to_numpy()

model = GaussianNaiveBayesScratch()
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f}")

print("\nLearned priors:", dict(zip(model.classes_, model.priors_)))
print("\nFirst 10 predictions:", preds[:10])
print("First 10 actual:     ", y_test[:10])
