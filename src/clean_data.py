import pandas as pd
from pathlib import Path

RAW = Path("data/raw/heart.csv")
OUT = Path("data/processed/heart_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, encoding="utf-8-sig")

# Drop the duplicate row we found in Step 2
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate row(s)")

# Fix the flipped label: rename to has_disease, where 1 = disease
df["has_disease"] = 1 - df["target"]
df = df.drop(columns=["target"])

print("\nNew label counts:")
print(df["has_disease"].value_counts())

print("\nCorrelation with corrected label (should now be POSITIVE for exang, oldpeak, ca, thal):")
print(df.corr()["has_disease"].drop("has_disease").sort_values())

df.to_csv(OUT, index=False)
print(f"\nSaved cleaned data to {OUT}")