import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import os

# =========================================================
# LOAD DATA
# =========================================================
DATA_PATH = "data/raw/international_matches_post2020.csv"
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score", "date"])

df["date"] = pd.to_datetime(df["date"])

# =========================================================
# TIME WEIGHTS (EXPONENTIAL DECAY)
# =========================================================
today = pd.Timestamp.today()

# age in days -> convert to months
age_days = (today - df["date"]).dt.days.values
age_months = age_days / 30.44

# half-life ~ 24 months (stable for international football)
tau = 24.0

df["weight"] = np.exp(-age_months / tau)

# normalize weights (important for stability)
df["weight"] /= df["weight"].mean()

# =========================================================
# TEAM INDEX
# =========================================================
teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
team_idx = {team: i for i, team in enumerate(teams)}
n_teams = len(teams)

# =========================================================
# PREBUILD MATRICES
# =========================================================
home_idx = df["home_team"].map(team_idx).values
away_idx = df["away_team"].map(team_idx).values

home_goals = df["home_score"].values
away_goals = df["away_score"].values

weights = df["weight"].values

# =========================================================
# PARAM VECTOR
# =========================================================
n_params = 2 * n_teams
x0 = np.zeros(n_params)

# =========================================================
# PARAM EXTRACT
# =========================================================
def get_params(x):
    attack = x[:n_teams]
    defense = x[n_teams:]
    return attack, defense

# =========================================================
# WEIGHTED NLL
# =========================================================
def nll(x):
    attack, defense = get_params(x)

    eps = 1e-9

    lam_home = np.exp(attack[home_idx] + defense[away_idx])
    lam_away = np.exp(attack[away_idx] + defense[home_idx])

    log_lik = (
        home_goals * np.log(lam_home + eps) - lam_home +
        away_goals * np.log(lam_away + eps) - lam_away
    )

    # weighted likelihood
    return -np.sum(weights * log_lik)

# =========================================================
# CALLBACK PROGRESS
# =========================================================
pbar = None

def callback(xk):
    global pbar
    if pbar:
        pbar.update(1)

# =========================================================
# TRAIN
# =========================================================
def fit_model(max_iter=50):
    global pbar
    pbar = tqdm(total=max_iter, desc="Training Poisson Model (time-weighted)")

    result = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False},
        callback=callback
    )

    pbar.close()
    return result.x

# =========================================================
# RANKINGS
# =========================================================
def get_ratings(x):
    attack, defense = get_params(x)

    return pd.DataFrame({
        "team": teams,
        "attack": attack,
        "defense": defense
    }).sort_values("attack", ascending=False)

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":

    print("Start training...")
    x_opt = fit_model(max_iter=50)
    print("Done optimization")

    ratings = get_ratings(x_opt)
    print(ratings.head(10))

    os.makedirs("results", exist_ok=True)
    output_path = "results/model_ratings_poisson_timeweighted.csv"
    ratings.to_csv(output_path, index=False)

    print(f"Saved ratings to {output_path}")