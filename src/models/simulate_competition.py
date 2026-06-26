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
# WORLD CUP STRUCTURE
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
team_to_group = {team: g for g, teams_list in groups.items() for team in teams_list}


# =========================================================
# PREPROCESSING REAL MATCH (SÉCURISÉ)
# =========================================================
def build_real_state(wc_matches):
    wc_matches = wc_matches.sort_values("date")

    group_results = {}
    ko_results = {}
    encounter_counts = {}

    for _, r in wc_matches.iterrows():
        h, a = r["home_team"], r["away_team"]
        
        if h not in WC_TEAMS or a not in WC_TEAMS:
            continue

        key = tuple(sorted((h, a)))
        encounter_counts[key] = encounter_counts.get(key, 0) + 1

        g_h = team_to_group[h]
        g_a = team_to_group[a]

        # RÈGLE 1 : Même poule + 1er match chronologique = Match de poule
        if g_h == g_a and encounter_counts[key] == 1:
            # CORRIGÉ : Structure en dict pour éviter l'inversion alphabétique
            group_results[key] = {
                "home": h,
                "away": a,
                "home_score": int(r["home_score"]),
                "away_score": int(r["away_score"])
            }
            
        # RÈGLE 2 : Poules différentes OU 2ème match dans la même poule = Phase finale
        else:
            winner_col = None
            for col in ["winner", "shootout_winner", "winning_team"]:
                if col in r and pd.notna(r[col]):
                    winner_col = r[col]
                    break
            
            ko_results[key] = {
                "home": h,
                "away": a,
                "home_score": int(r["home_score"]),
                "away_score": int(r["away_score"]),
                "winner": winner_col
            }

    return group_results, ko_results


# =========================================================
# FILTER WORLD CUP MATCHES ONLY
# =========================================================
def get_wc_matches(start_date=None, end_date=None):
    df = matches.copy()
    df = df[df["tournament"] == "FIFA World Cup"].copy()

    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df.sort_values("date")


# =========================================================
# RO32
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
    add(first["F"], second["C"])  
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
# POISSON SIM
# =========================================================
def sim_match(home, away):
    h = team_idx[home]
    a = team_idx[away]

    lam_h = np.exp(attack[h] + defense[a])
    lam_a = np.exp(attack[a] + defense[h])

    return np.random.poisson(lam_h), np.random.poisson(lam_a)


# =========================================================
# GROUP SIMULATION (CORRIGÉ)
# =========================================================
def simulate_group(group_teams, group_results):
    n = len(group_teams)
    idx = {t: i for i, t in enumerate(group_teams)}

    pts = np.zeros(n)
    gf = np.zeros(n)
    ga = np.zeros(n)

    played = set()

    # -----------------------------------------------------
    # 1. INTEGRATION DES MATCHS REELS REUSSITE
    # -----------------------------------------------------
    for key, match_data in group_results.items():
        h, a = match_data["home"], match_data["away"]
        g1, g2 = match_data["home_score"], match_data["away_score"]

        if h in group_teams and a in group_teams:
            i, j = idx[h], idx[a]

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

    # -----------------------------------------------------
    # 2. SIMULATION DES MATCHS RESTANTS
    # -----------------------------------------------------
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


def select_best_thirds(third_stats):
    df = pd.DataFrame(third_stats)
    return df.sort_values(
        ["pts", "gd", "gf"],
        ascending=False
    ).head(8)["team"].tolist()


def simulate_knockout_match(t1, t2):
    i, j = team_idx[t1], team_idx[t2]
    g1 = np.random.poisson(np.exp(attack[i] + defense[j]))
    g2 = np.random.poisson(np.exp(attack[j] + defense[i]))

    if g1 == g2:
        return t1 if np.random.rand() < 0.5 else t2
    return t1 if g1 > g2 else t2


# =========================================================
# KNOCKOUT WITH REAL RESULT SUPPORT
# =========================================================
def sim_knockout(t1, t2, ko_results):
    key = tuple(sorted((t1, t2)))

    if key in ko_results:
        match_data = ko_results[key]
        h, a = match_data["home"], match_data["away"]
        g1, g2 = match_data["home_score"], match_data["away_score"]

        if g1 > g2:
            return h
        elif g2 > g1:
            return a
        else:
            if match_data["winner"] is not None:
                return match_data["winner"]
            return t1 if np.random.rand() < 0.5 else t2

    return simulate_knockout_match(t1, t2)


# =========================================================
# TOURNAMENT
# =========================================================
def simulate_tournament(ro32, ko_results):
    # 1. Seizièmes de finale -> 16 équipes qualifiées pour les 8èmes
    r16 = [sim_knockout(a, b, ko_results) for a, b in ro32]
    
    # 2. Huitièmes de finale -> 8 équipes qualifiées pour les Quarts
    qf = [sim_knockout(r16[i], r16[i+1], ko_results) for i in range(0, 16, 2)]
    
    # 3. Quarts de finale -> 4 équipes qualifiées pour les Demi-finales
    sf = [sim_knockout(qf[i], qf[i+1], ko_results) for i in range(0, 8, 2)]

    # 4. CORRECTION : On joue les deux demi-finales pour trouver les deux finalistes
    finalist_1 = sim_knockout(sf[0], sf[1], ko_results)  # Match entre le vainqueur du Quart 1 et 2
    finalist_2 = sim_knockout(sf[2], sf[3], ko_results)  # Match entre le vainqueur du Quart 3 et 4

    # 5. La grande finale
    final = sim_knockout(finalist_1, finalist_2, ko_results)
    return final


# =========================================================
# MONTE CARLO
# =========================================================
def monte_carlo(n_sim=1000):
    wc_matches = get_wc_matches("2026-06-01", "2026-07-31")
    group_results, ko_results = build_real_state(wc_matches)

    wins = {}

    for _ in tqdm(range(n_sim)):
        first, second, third_stats = build_groups(group_results)

        best_thirds = select_best_thirds(third_stats)
        ro32 = get_ro32(first, second, best_thirds)

        winner = simulate_tournament(ro32, ko_results)
        wins[winner] = wins.get(winner, 0) + 1

    return pd.DataFrame([
        {"team": t, "win_prob": v / n_sim}
        for t, v in wins.items()
    ]).sort_values("win_prob", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=10000)
    args = parser.parse_args()

    print(f"Running Monte Carlo with {args.n_sim} simulations...")
    results = monte_carlo(args.n_sim)

    print("\n===== WIN PROBABILITIES =====")
    print(results.head(32))