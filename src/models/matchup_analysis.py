import numpy as np
import pandas as pd
from scipy.special import gammaln


class MatchupAnalyzer:
    def __init__(self, ratings_df):
        self.teams = ratings_df["team"].values
        self.team_idx = {t: i for i, t in enumerate(self.teams)}

        self.attack = ratings_df["attack"].values
        self.defense = ratings_df["defense"].values

    # =========================================================
    # EXPECTED GOALS
    # =========================================================
    def expected_goals(self, team1, team2):
        i = self.team_idx[team1]
        j = self.team_idx[team2]

        lam1 = np.exp(self.attack[i] + self.defense[j])
        lam2 = np.exp(self.attack[j] + self.defense[i])

        return lam1, lam2

    # =========================================================
    # POISSON PMF (STABLE)
    # =========================================================
    @staticmethod
    def poisson_pmf(k, lam):
        return np.exp(k * np.log(lam) - lam - gammaln(k + 1))

    # =========================================================
    # JOINT SCORE MATRIX
    # =========================================================
    def score_distribution(self, team1, team2, max_goals=7):
        lam1, lam2 = self.expected_goals(team1, team2)

        p1 = np.array([self.poisson_pmf(k, lam1) for k in range(max_goals + 1)])
        p2 = np.array([self.poisson_pmf(k, lam2) for k in range(max_goals + 1)])

        joint = np.outer(p1, p2)
        joint /= joint.sum()

        return joint

    # =========================================================
    # 1X2 PROBABILITIES
    # =========================================================
    def match_outcome_probs(self, team1, team2, max_goals=7):
        joint = self.score_distribution(team1, team2, max_goals)

        win1 = 0.0
        draw = 0.0
        win2 = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = joint[i, j]

                if i > j:
                    win1 += p
                elif i == j:
                    draw += p
                else:
                    win2 += p

        return {
            "team1_win": win1,
            "draw": draw,
            "team2_win": win2
        }

    # =========================================================
    # TOP SCORELINES
    # =========================================================
    def top_scorelines(self, team1, team2, top_k=5, max_goals=7):
        joint = self.score_distribution(team1, team2, max_goals)

        results = []

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                results.append(((i, j), joint[i, j]))

        results.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(
            [(f"{s[0]}-{s[1]}", p) for s, p in results[:top_k]],
            columns=["score", "probability"]
        )


# =========================================================
# RUN EXAMPLE
# =========================================================
if __name__ == "__main__":

    ratings = pd.read_csv("results/model_ratings_poisson_timeweighted.csv")

    analyzer = MatchupAnalyzer(ratings)

    t1 = "France"
    t2 = "Brazil"

    print(f"\n=== MATCH ANALYSIS: {t1} vs {t2} ===\n")

    print("Top scorelines:")
    print(analyzer.top_scorelines(t1, t2))

    print("\n1X2 probabilities:")
    probs = analyzer.match_outcome_probs(t1, t2)

    print(f"{t1} win: {probs['team1_win']:.3f}")
    print(f"Draw    : {probs['draw']:.3f}")
    print(f"{t2} win: {probs['team2_win']:.3f}")