"""Reproduce every result and figure of the project with one command.

    python run_all.py            # full run (about 5-10 minutes on one CPU core)
    python run_all.py --quick    # small smoke-test configuration (about 1 minute)
    python run_all.py --plots-only   # re-draw figures / tables from results/results.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from wwdm import export, plotting
from wwdm.experiments import Config, log, run_all

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "results.pkl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small, fast configuration (smoke test)")
    ap.add_argument("--plots-only", action="store_true", help="skip simulations, reuse cached results")
    args = ap.parse_args()

    t0 = time.time()
    if args.plots_only:
        log(f"Loading cached results from {CACHE}")
        R = pickle.load(open(CACHE, "rb"))
    else:
        R = run_all(Config(quick=args.quick))
        CACHE.parent.mkdir(exist_ok=True)
        pickle.dump(R, open(CACHE, "wb"))
    log("Writing CSV tables and summary.json ...")
    export.write_all(R, ROOT / "results")
    log("Drawing figures ...")
    plotting.make_all(R)
    log(f"Done in {time.time() - t0:.0f} s. Figures: figures/   Tables: results/")


if __name__ == "__main__":
    main()
