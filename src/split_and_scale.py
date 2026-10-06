import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/processed/heart_clean.csv")
OUT_DIR = Path("data/processed")

df = pd.read_csv(DATA)

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "has_disease"

# --- Stratified split: preserve the disease/no-disease ratio in both sets ---
def stratified_split(df, target_col, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls].sample(frac=1, random_state=seed)
        n_test = int(len(cls_df) * test_size)
        test_parts.append(cls_df.iloc[:n_test])
        train_parts.append(cls_df.iloc[n_test:])
    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df

train_df, test_df = stratified_split(df, TARGET)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
print("Train class balance:\n", train_df[TARGET].value_counts(normalize=True))
print("Test class balance:\n", test_df[TARGET].value_counts(normalize=True))

# --- Scale: fit on train only, apply to both ---
means = train_df[FEATURES].mean()
stds = train_df[FEATURES].std().replace(0, 1)  # guard divide-by-zero

train_scaled = train_df.copy()
test_scaled = test_df.copy()
train_scaled[FEATURES] = (train_df[FEATURES] - means) / stds
test_scaled[FEATURES] = (test_df[FEATURES] - means) / stds

train_scaled.to_csv(OUT_DIR / "train.csv", index=False)
test_scaled.to_csv(OUT_DIR / "test.csv", index=False)

# Save the scaling parameters too — you'll need them later to scale any new patient the same way
scale_params = pd.DataFrame({"mean": means, "std": stds})
scale_params.to_csv(OUT_DIR / "scale_params.csv")

print("\nSaved train.csv, test.csv, and scale_params.csv to data/processed/")
print("\nSample of scaled training data:")
print(train_scaled.head())    