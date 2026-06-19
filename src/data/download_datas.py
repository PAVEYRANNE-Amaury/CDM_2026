import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. LOAD DATASET INTERNATIONAL RESULTS
# =========================================================
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

df = pd.read_csv(url)

print(df.head())

# =========================================================
# 2. CLEAN + FILTER POST 2022
# =========================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df[df["date"].dt.year >= 2020].copy()

# enlever lignes invalides
df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

# =========================================================
# 3. CREATE TARGET (important pour ML)
# =========================================================

def result(row):
    if row["home_score"] > row["away_score"]:
        return 1   # home win
    elif row["home_score"] < row["away_score"]:
        return -1  # away win
    else:
        return 0   # draw

df["result"] = df.apply(result, axis=1)

# =========================================================
# 4. SAVE CLEAN DATASET
# =========================================================

df.to_csv(DATA_DIR / "international_matches_post2020.csv", index=False)

print("Dataset shape:", df.shape)
print(df.head())