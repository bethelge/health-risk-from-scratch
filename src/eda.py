import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

DATA = Path("data/processed/heart_clean.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# --- 1. Class balance ---
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="has_disease", data=df, hue="has_disease", palette="Set2", legend=False, ax=ax)
ax.set_title("Class Balance: Disease (1) vs No Disease (0)")
ax.set_xlabel("has_disease")
for i, v in enumerate(df["has_disease"].value_counts().sort_index()):
    ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_balance.png", dpi=150)
plt.close()
print("Saved class_balance.png")

# --- 2. Correlation heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap (corrected label)")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
plt.close()
print("Saved correlation_heatmap.png")

# --- 3. Feature distributions split by disease status ---
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(continuous_features):
    sns.kdeplot(data=df, x=feat, hue="has_disease", fill=True, alpha=0.4,
                ax=axes[i], palette="Set2", common_norm=False)
    axes[i].set_title(f"{feat} by disease status")
for j in range(len(continuous_features), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_distributions.png", dpi=150)
plt.close()
print("Saved feature_distributions.png")

# --- 4. Summary stats ---
print("\nSummary statistics:")
print(df.describe().T)