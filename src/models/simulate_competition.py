import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# =========================================================
# LOAD DATA
# =========================================================
ratings = pd.read_csv("results/model_ratings_poisson_timeweighted.csv")

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
# FAST POISSON SIMULATION (VECTOR FRIENDLY)
# =========================================================
def sim_match_fast(h_idx, a_idx):
    lam_h = np.exp(attack[h_idx] + defense[a_idx])
    lam_a = np.exp(attack[a_idx] + defense[h_idx])

    return (
        np.random.poisson(lam_h, size=lam_h.shape if hasattr(lam_h, "shape") else ()),
        np.random.poisson(lam_a, size=lam_a.shape if hasattr(lam_a, "shape") else ())
    )


# =========================================================
# GROUP SIMULATION (OPTIMIZED)
# =========================================================
def simulate_group(group_teams):
    n = len(group_teams)

    idx = [team_idx[t] for t in group_teams]

    pts = np.zeros(n)
    gf = np.zeros(n)
    ga = np.zeros(n)

    # 6 matches fixed
    for i in range(n):
        for j in range(i + 1, n):

            h, a = idx[i], idx[j]

            g1 = np.random.poisson(np.exp(attack[h] + defense[a]))
            g2 = np.random.poisson(np.exp(attack[a] + defense[h]))

            gf[i] += g1
            ga[i] += g2

            gf[j] += g2
            ga[j] += g1

            if g1 > g2:
                pts[i] += 3
            elif g2 > g1:
                pts[j] += 3
            else:
                pts[i] += 1
                pts[j] += 1

    gd = gf - ga

    order = np.lexsort((-gf, -gd, -pts))  # FIFA tie-break approx

    return (
        np.array(group_teams)[order],
        pts[order],
        gf[order],
        ga[order],
        gd[order],
    )


# =========================================================
# FULL GROUP PHASE
# =========================================================
def build_groups():
    first, second, third = {}, {}, {}
    third_stats = []

    for g, teams_g in groups.items():

        ranked, pts, gf, ga, gd = simulate_group(teams_g)

        first[g] = ranked[0]
        second[g] = ranked[1]
        third[g] = ranked[2]

        third_stats.append({
            "team": ranked[2],
            "group": g,
            "pts": pts[2],
            "gf": gf[2],
            "gd": gd[2]
        })

    return first, second, third_stats


# =========================================================
# BEST THIRD (GLOBAL UNIQUE)
# =========================================================
def select_best_thirds(third_stats):
    df = pd.DataFrame(third_stats)

    best = df.sort_values(
        ["pts", "gd", "gf"],
        ascending=False
    ).head(8)["team"].tolist()

    return best


# =========================================================
# RO32 BUILDER (IDENTIQUE LOGIQUE MAIS SAFE)
# =========================================================
def get_ro32(first, second, best_thirds):

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
# KNOCKOUT SIM (SIMPLE BUT FAST)
# =========================================================
def sim_knockout(a, b):
    ia, ib = team_idx[a], team_idx[b]

    g1 = np.random.poisson(np.exp(attack[ia] + defense[ib]))
    g2 = np.random.poisson(np.exp(attack[ib] + defense[ia]))

    if g1 == g2:
        g1 += np.random.binomial(1, 0.5)

    return a if g1 > g2 else b


def simulate_tournament(ro32):

    r16 = [sim_knockout(a, b) for a, b in ro32]
    qf = [sim_knockout(r16[i], r16[i+1]) for i in range(0, 16, 2)]
    sf = [sim_knockout(qf[i], qf[i+1]) for i in range(0, 8, 2)]
    final = sim_knockout(sf[0], sf[1])

    return final


# =========================================================
# MONTE CARLO
# =========================================================
def monte_carlo(n_sim=1000):

    wins = {}

    for _ in tqdm(range(n_sim)):

        first, second, third_stats = build_groups()
        best_thirds = select_best_thirds(third_stats)
        ro32 = get_ro32(first, second, best_thirds)
        winner = simulate_tournament(ro32)

        wins[winner] = wins.get(winner, 0) + 1

    return pd.DataFrame([
        {"team": t, "win_prob": v / n_sim}
        for t, v in wins.items()
    ]).sort_values("win_prob", ascending=False)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=10000)
    args = parser.parse_args()

    print(f"Running Monte Carlo with {args.n_sim} simulations...")

    results = monte_carlo(n_sim=args.n_sim)

    print("\n===== WIN PROBABILITIES =====")
    print(results.head(20))