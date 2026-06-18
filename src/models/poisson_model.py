import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import os

# =========================================================
# LOAD DATA
# =========================================================
DATA_PATH = "data/raw/international_matches_post2022.csv"
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

# =========================================================
# TEAM INDEX
# =========================================================
teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
team_idx = {team: i for i, team in enumerate(teams)}
n_teams = len(teams)

# =========================================================
# PREBUILD MATRICES (VERY IMPORTANT)
# =========================================================
home_idx = df["home_team"].map(team_idx).values
away_idx = df["away_team"].map(team_idx).values

home_goals = df["home_score"].values
away_goals = df["away_score"].values

neutral = df["neutral"].fillna(False).astype(int).values

n_matches = len(df)

# =========================================================
# PARAM VECTOR
# =========================================================
n_params = 2 * n_teams + 1
x0 = np.zeros(n_params)

# =========================================================
# PARAM EXTRACT
# =========================================================
def get_params(x):
    attack = x[:n_teams]
    defense = x[n_teams:2*n_teams]
    home_adv = x[-1]
    return attack, defense, home_adv

# =========================================================
# VECTORISED NLL
# =========================================================
def nll(x):
    attack, defense, home_adv = get_params(x)

    # home advantage conditionnel
    ha = home_adv * (1 - neutral)

    # expected goals
    lam_home = np.exp(attack[home_idx] + defense[away_idx] + ha)
    lam_away = np.exp(attack[away_idx] + defense[home_idx])

    eps = 1e-9

    loss = np.sum(
        home_goals * np.log(lam_home + eps) - lam_home +
        away_goals * np.log(lam_away + eps) - lam_away
    )

    return -loss

# =========================================================
# CALLBACK FOR PROGRESS BAR
# =========================================================
pbar = None

def callback(xk):
    global pbar
    if pbar:
        pbar.update(1)

# =========================================================
# TRAIN MODEL
# =========================================================
def fit_model(max_iter=50):

    global pbar
    pbar = tqdm(total=max_iter, desc="Training Poisson Model")

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
# EXTRACT RATINGS
# =========================================================
def get_ratings(x):

    attack, defense, home_adv = get_params(x)

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

    attack, defense, home_adv = get_params(x_opt)

    print("HOME ADVANTAGE GLOBAL:", home_adv)

    # sauvegarde
    os.makedirs("results", exist_ok=True)

    output_path = "results/model_ratings_poisson.csv"
    ratings.to_csv(output_path, index=False)

    print(f"Saved ratings to {output_path}")