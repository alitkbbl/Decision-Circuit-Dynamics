"""Write human-readable result tables (CSV) and summary.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .experiments import METRICS


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def _write_csv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_tables(R, res_dir: Path):
    res_dir.mkdir(exist_ok=True)
    B = R["baseline"]
    km = B["k_main"]
    S = B["summary"]
    rows = [[c] + [S[m][i, km] for m in ("n_dec", "acc", "p_nodec", "p_prem", "rt_mean", "rt_corr", "rt_err", "rt_sd")]
            for i, c in enumerate(B["coh"])]
    _write_csv(res_dir / "baseline_summary.csv",
               ["coherence_pct", "n_decided", "accuracy", "p_timeout", "p_premature", "rt_mean_s",
                "rt_correct_s", "rt_error_s", "rt_sd_s"], rows)
    for name, key in (("sweep_J12_sigma.csv", "sweep"), ("sweep_J11_sigma.csv", "sweep11")):
        G = R[key]
        km_ = R["k_th"]
        rows = [[x, s, c] + [G[m][i, j, k, km_] for m in METRICS]
                for i, x in enumerate(G["xs"]) for j, s in enumerate(G["sigmas"]) for k, c in enumerate(G["cohs"])]
        _write_csv(res_dir / name, [G["xname"], "sigma", "coherence_pct"] + list(METRICS), rows)
    geo = R["geo"]
    _write_csv(res_dir / "geometry_J12.csv",
               ["J12", "lambda_u", "lambda_s", "d_to_separatrix", "d_start_saddle", "n_stable", "n_saddle",
                "lambda_resting", "d_resting_to_saddle", "attractor_rate_Hz"],
               [[geo["xs"][i], geo["lam_u"][i], geo["lam_s"][i], geo["d"][i], geo["d_saddle"][i], geo["n_stable"][i],
                 geo["n_saddle"][i], geo["lam_sp"][i], geo["d_barrier"][i], geo["r_att"][i]]
                for i in range(len(geo["xs"]))])
    G2 = R["grid2d"]
    rows = [[G2["J11s"][i], G2["J12s"][j], G2["sigmas"][k], int(G2["keep"][i, j]), G2["g_lam_u"][i, j], G2["g_d"][i, j]]
            + [G2[m][i, j, k] for m in METRICS]
            for i in range(len(G2["J11s"])) for j in range(len(G2["J12s"])) for k in range(len(G2["sigmas"]))]
    _write_csv(res_dir / "grid2d_J11_J12.csv", ["J11", "J12", "sigma", "retained_WTA", "lambda_u", "d"] + list(METRICS), rows)
    h1 = R["h1"]
    summ = dict(
        calibration=B["cal"], weibull_alpha_beta=B["weibull"],
        bifurcations_J12={str(c): [(v, a, b) for v, a, b in t] for c, t in R["bif"]["trans"].items()},
        bifurcations_J12_no_stimulus=[(v, a, b) for v, a, b in R["bif"]["trans_off"]],
        bifurcations_J11={str(c): [(v, a, b) for v, a, b in t] for c, t in R["bif"]["trans_J11"].items()},
        H1={t: {m: {k: v for k, v in r.items() if k != "pred"} for m, r in h1[t].items() if m != "y"}
            for t in ("log_rt", "logit_err")},
        H1_n=h1["n"], H1_corr_lambda_d=h1["corr_lam_d"],
        H2_J12={k: v for k, v in R["h2"].items() if k != "bif"},
        H2_J11=dict(R["h2_j11"]),
        regime_counts={str(f): np.bincount(l.ravel(), minlength=4).tolist() for f, l in R["sweep"]["regimes"].items()},
        landscape_transitions=R["landscape"]["n_switch"],
        grid2d=dict(analysis={k: v for k, v in R["h1_2d"].items() if k not in ("mask", "models")},
                    models={m: {t: {k: v for k, v in r.items() if k not in ("pred",)} for t, r in d.items()}
                            for m, d in R["h1_2d"]["models"].items()}),
        premature={k: ({m: {kk: vv for kk, vv in r.items() if kk != "pred"} for m, r in v.items()} if k == "models" else v)
                   for k, v in R["prem"].items() if k not in ("obs", "sigma_rows")},
        resting_state_stability=R["rest"],
        trapping={k: v for k, v in R["trap"].items() if k not in ("lead_rate_timeouts", "J11_grid", "r_att_curve")},
    )
    with open(res_dir / "summary.json", "w") as f:
        json.dump(_clean(summ), f, indent=1)


def write_all(R, res_dir: Path):
    """Write the CSV tables and summary.json (plain data exports; nothing is written into the report)."""
    write_tables(R, res_dir)
