import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# =========================================================
# LOAD DATA
# =========================================================
matches = pd.read_csv("data/raw/international_matches_post2020.csv")
ratings = pd.read_csv("results/model_ratings_poisson_timeweighted.csv")

matches["date"] = pd.to_datetime(matches["date"])

# =========================================================
# MODEL
# =========================================================
teams = ratings["team"].values
team_idx = {t: i for i, t in enumerate(teams)}

attack = ratings["attack"].values
defense = ratings["defense"].values

# =========================================================
# GROUPS
# =========================================================
groups = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# =========================================================
# MATCH SIMULATION
# =========================================================
def sim_match(home, away):
    h = team_idx[home]
    a = team_idx[away]

    lam_home = np.exp(attack[h] + defense[a])
    lam_away = np.exp(attack[a] + defense[h])

    return np.random.poisson(lam_home), np.random.poisson(lam_away)


def sim_match_knockout(team1, team2):
    h = team_idx[team1]
    a = team_idx[team2]

    lam1 = np.exp(attack[h] + defense[a])
    lam2 = np.exp(attack[a] + defense[h])

    g1 = np.random.poisson(lam1)
    g2 = np.random.poisson(lam2)

    if g1 == g2:
        g1 += np.random.binomial(1, 0.5)
        g2 += np.random.binomial(1, 0.5)

    return team1 if g1 > g2 else team2

# =========================================================
# GROUP STANDINGS
# =========================================================
def init_table():
    return pd.DataFrame(columns=["pts", "gf", "ga", "gd"]).astype(float)


def update_table(table, team, gf, ga):
    if team not in table.index:
        table.loc[team] = [0, 0, 0, 0]

    table.loc[team, "gf"] += gf
    table.loc[team, "ga"] += ga
    table.loc[team, "gd"] = table.loc[team, "gf"] - table.loc[team, "ga"]

    if gf > ga:
        table.loc[team, "pts"] += 3
    elif gf == ga:
        table.loc[team, "pts"] += 1

    return table

# =========================================================
# GROUPS
# =========================================================
def build_groups():
    group_tables = {}

    for g, teams_g in groups.items():
        table = init_table()

        for i in range(len(teams_g)):
            for j in range(i + 1, len(teams_g)):
                h, a = teams_g[i], teams_g[j]

                gf, ga = sim_match(h, a)

                table = update_table(table, h, gf, ga)
                table = update_table(table, a, ga, gf)

        group_tables[g] = table.sort_values(["pts", "gd", "gf"], ascending=False)

    return group_tables

# =========================================================
# RO32 BUILDER
# =========================================================
def get_ro32(group_tables):

    first, second, third = {}, {}, {}

    for g, t in group_tables.items():
        first[g] = t.index[0]
        second[g] = t.index[1]
        third[g] = t.index[2]

    third_list = pd.DataFrame([
        {
            "team": third[g],
            "group": g,
            "pts": group_tables[g].loc[third[g], "pts"],
            "gd": group_tables[g].loc[third[g], "gd"],
            "gf": group_tables[g].loc[third[g], "gf"]
        }
        for g in groups
    ])

    best_thirds = third_list.sort_values(
        ["pts", "gd", "gf"],
        ascending=False
    ).head(8)["team"].tolist()

    pool = best_thirds.copy()

    def pick():
        return pool.pop(0)

    ro32 = []

    def add(a, b):
        ro32.append((a, b))

    add(first["E"], pick())
    add(first["I"], pick())
    add(second["A"], second["B"])
    add(first["F"], first["C"])
    add(second["K"], second["L"])
    add(first["H"], second["J"])
    add(first["D"], pick())
    add(first["G"], pick())
    add(first["C"], second["F"])
    add(second["E"], second["I"])
    add(first["A"], pick())
    add(first["L"], pick())
    add(first["J"], second["H"])
    add(second["D"], second["G"])
    add(first["B"], pick())
    add(first["K"], pick())

    return ro32

# =========================================================
# TOURNAMENT
# =========================================================
def simulate_tournament(ro32):

    r16 = [sim_match_knockout(a, b) for a, b in ro32]
    qf = [sim_match_knockout(r16[i], r16[i+1]) for i in range(0, 16, 2)]
    sf = [sim_match_knockout(qf[i], qf[i+1]) for i in range(0, 8, 2)]

    final_winner = sim_match_knockout(sf[0], sf[1])

    return final_winner

# =========================================================
# MONTE CARLO
# =========================================================
def monte_carlo(n_sim=1000):

    wins = {}

    for _ in tqdm(range(n_sim)):

        group_tables = build_groups()
        ro32 = get_ro32(group_tables)
        winner = simulate_tournament(ro32)

        wins[winner] = wins.get(winner, 0) + 1

    return (
        pd.DataFrame([
            {"team": t, "win_prob": w / n_sim}
            for t, w in wins.items()
        ])
        .sort_values("win_prob", ascending=False)
    )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=10000)
    args = parser.parse_args()

    print(f"Running Monte Carlo with {args.n_sim} simulations...")

    probs = monte_carlo(n_sim=args.n_sim)

    print("\n===== WIN PROBABILITIES =====")
    print(probs.head(20))