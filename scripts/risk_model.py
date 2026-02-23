from pathlib import Path
import pandas as pd
from clean_data import clean_cap_data

# -----------------------
# 1. Load Data
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "cardinals_page.html"

tables = pd.read_html(data_path)
df = tables[0]

# -----------------------
# 2. Clean Data
# -----------------------

df = clean_cap_data(df)

# Ensure Age is numeric
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# -----------------------
# 3. Risk Metrics
# -----------------------

# Dead Cap Risk (contract rigidity)
df["dead_cap_ratio"] = df["Dead Cap"].abs() / df["Cap Hit"]

# Cash Exposure Risk
df["cash_ratio"] = df["Cash Total"] / df["Cap Hit"]

# Age Risk (starts at age 30)
df["age_risk"] = ((df["Age"] - 29).clip(lower=0)) * 0.05

# Composite Risk Score (weighted model)
df["risk_score"] = (
    df["dead_cap_ratio"] * 0.5 +
    df["age_risk"] * 0.3 +
    df["cash_ratio"] * 0.2
)

# Remove infinite / NaN values
df = df.replace([float("inf"), -float("inf")], pd.NA)
df = df.dropna(subset=["risk_score"])

# Sort by highest risk
df = df.sort_values(by="risk_score", ascending=False)

# -----------------------
# 4. Output Results
# -----------------------

print("\nTop 10 Highest Risk Contracts:\n")
print(df[["Player (59)", "Cap Hit", "Dead Cap", "Age", "risk_score"]].head(10))

# -----------------------
# 5. Team-Level Risk
# -----------------------

total_dead_cap = df["Dead Cap"].abs().sum()
total_cap_hit = df["Cap Hit"].sum()

team_dead_cap_ratio = total_dead_cap / total_cap_hit

print("\nTeam Dead Cap Exposure Ratio:", round(team_dead_cap_ratio, 3))