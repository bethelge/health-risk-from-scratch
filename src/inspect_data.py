import pandas as pd

df = pd.read_csv("data/raw/heart.csv", encoding="utf-8-sig")

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nTarget counts:\n", df["target"].value_counts())
print("\nCorrelation with target:\n",
      df.corr()["target"].drop("target").sort_values())