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

teams = ratings["team"].values
team_idx = {t: i for i, t in enumerate(teams)}

attack = ratings["attack"].values
defense = ratings["defense"].values


# =========================================================
# WORLD CUP TEAMS
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

WC_TEAMS = set([t for g in groups.values() for t in g])


# =========================================================
# FILTER WORLD CUP MATCHES (REAL DATA)
# =========================================================
def get_wc_matches():
    df = matches.copy()

    # keep only WC teams
    df = df[
        df["home_team"].isin(WC_TEAMS) &
        df["away_team"].isin(WC_TEAMS)
    ]

    # keep tournament containing WC / qualifiers / finals logic
    # (robust filter, adjust if needed)
    wc_keywords = ["World Cup", "WC", "FIFA"]
    df = df[df["tournament"].str.contains("|".join(wc_keywords), case=False, na=False)]

    return df


# =========================================================
# POISSON SIMULATION (FALLBACK ONLY)
# =========================================================
def sim_match(home, away):
    h = team_idx[home]
    a = team_idx[away]

    lam_h = np.exp(attack[h] + defense[a])
    lam_a = np.exp(attack[a] + defense[h])

    return np.random.poisson(lam_h), np.random.poisson(lam_a)


# =========================================================
# GROUP SIMULATION WITH REAL MATCH INTEGRATION
# =========================================================
def simulate_group(group_teams, wc_matches):

    n = len(group_teams)
    idx = {t: i for i, t in enumerate(group_teams)}

    pts = np.zeros(n)
    gf = np.zeros(n)
    ga = np.zeros(n)

    played = set()

    # =====================================================
    # 1. REAL MATCHES FIRST
    # =====================================================
    real = wc_matches[
        wc_matches["home_team"].isin(group_teams) &
        wc_matches["away_team"].isin(group_teams)
    ]

    for _, r in real.iterrows():

        h, a = r["home_team"], r["away_team"]
        key = tuple(sorted((h, a)))

        if key in played:
            continue

        i, j = idx[h], idx[a]

        g1 = int(r["home_score"])
        g2 = int(r["away_score"])

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

        played.add(key)

    # =====================================================
    # 2. SIMULATE MISSING MATCHES
    # =====================================================
    for i in range(n):
        for j in range(i + 1, n):

            h, a = group_teams[i], group_teams[j]
            key = tuple(sorted((h, a)))

            if key in played:
                continue

            g1, g2 = sim_match(h, a)

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

    order = np.lexsort((-gf, -gd, -pts))

    return (
        np.array(group_teams)[order],
        pts[order],
        gf[order],
        ga[order],
        gd[order],
    )


# =========================================================
# FULL GROUP STAGE
# =========================================================
def build_groups(wc_matches):

    first, second, third = {}, {}, {}
    third_stats = []

    for g, teams_g in groups.items():

        ranked, pts, gf, ga, gd = simulate_group(teams_g, wc_matches)

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
# BEST THIRD
# =========================================================
def select_best_thirds(third_stats):

    df = pd.DataFrame(third_stats)

    return df.sort_values(
        ["pts", "gd", "gf"],
        ascending=False
    ).head(8)["team"].tolist()


# =========================================================
# RO32 (UNCHANGED LOGIC)
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
# KNOCKOUT
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

    return sim_knockout(sf[0], sf[1])


# =========================================================
# MONTE CARLO
# =========================================================
def monte_carlo(n_sim=1000):

    wc_matches = get_wc_matches()
    wins = {}

    for _ in tqdm(range(n_sim)):

        first, second, third_stats = build_groups(wc_matches)
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