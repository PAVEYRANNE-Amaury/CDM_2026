import subprocess
import sys
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
N_MC = 10000  # nombre de simulations Monte Carlo
RUN_DOWNLOAD = True
RUN_FIT = True
RUN_SIMULATION = True

PYTHON = sys.executable

# =========================================================
# HELPERS
# =========================================================
def run_script(script_path, n_sim=None):
    cmd = [sys.executable, script_path]

    if n_sim is not None:
        cmd += ["--n_sim", str(n_sim)]

    print("\n============================================================")
    print("RUNNING:", " ".join(cmd))
    print("============================================================\n")

    subprocess.run(cmd, check=True)


# =========================================================
# PIPELINE
# =========================================================
def main():

    start = datetime.now()

    # 1. DOWNLOAD DATA
    if RUN_DOWNLOAD:
        run_script("src/data/download_datas.py")

    # 2. FIT POISSON MODEL
    if RUN_FIT:
        run_script("src/models/poisson_model.py")

    # 3. MONTE CARLO SIMULATION
    if RUN_SIMULATION:
        run_script(
            "src/models/simulate_competition.py",
            n_sim=N_MC
        )

    end = datetime.now()

    print("\n" + "="*60)
    print("PIPELINE DONE")
    print(f"TIME: {end - start}")
    print("="*60)


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    main()
