"""Write human-readable tables (CSV / JSON) and the LaTeX macros used by the report."""
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
    )
    with open(res_dir / "summary.json", "w") as f:
        json.dump(_clean(summ), f, indent=1)


def _f(x, n=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def write_latex(R, rep_dir: Path):
    """Macros and small tables included by report.tex, so that all numbers come from the code."""
    rep_dir.mkdir(exist_ok=True)
    B, h1, h2, h2b, cfg = R["baseline"], R["h1"], R["h2"], R["h2_j11"], R["cfg"]
    km = B["k_main"]
    S = B["summary"]
    L = []

    def mac(n, v):
        L.append(f"\\newcommand{{\\{n}}}{{{v}}}")

    mac("rstar", _f(B["cal"]["r_star"], 1))
    mac("thetaHz", _f(B["cal"]["theta"], 1))
    mac("wbAlpha", _f(B["weibull"][0], 1))
    mac("wbBeta", _f(B["weibull"][1], 2))
    mac("nBase", str(cfg.n_baseline))
    mac("nSweep", str(cfg.n_sweep))
    mac("nGrid", str(cfg.n_grid))
    mac("HoneN", str(h1["n"]))
    for t, nm in (("log_rt", "RT"), ("logit_err", "Err")):
        for m, mn in (("geo", "Geo"), ("param", "Par"), ("sigma", "Sig"), ("lam_sigma", "LamSig"), ("geo_plus", "GeoPlus")):
            mac(f"Hone{nm}{mn}R", _f(h1[t][m]["r2"], 2))
            mac(f"Hone{nm}{mn}CV", _f(h1[t][m]["cv_r2"], 2))
    if "prem_vs_barrier" in h2:
        mac("PremBarrierR", _f(h2["prem_vs_barrier"]["r2"], 2))
        mac("PremBarrierRho", _f(h2["prem_vs_barrier_rho"], 2))
        mac("PremBarrierN", str(h2["prem_vs_barrier"]["n"]))
    jb = h2b["j11_boundary"]
    mac("JelevenTheta", _f(h2b["j11_theta"], 3))
    mac("JelevenBoundLo", _f(np.nanmin(jb), 3))
    mac("JelevenBoundHi", _f(np.nanmax(jb), 3))
    if "shift" in h2b:
        mac("JelevenShiftSlope", _f(h2b["shift"]["slope"], 2))
        mac("JelevenShiftR", _f(h2b["shift"]["r2"], 2))
    for c, tag in ((0.0, "Zero"), (12.8, "Hi")):
        for i, (v, _, _) in enumerate(R["bif"]["trans"][c]):
            mac(f"bifJ{tag}{chr(65 + i)}", _f(v, 4))
    for i, (v, _, _) in enumerate(R["bif"]["trans_J11"][0.0]):
        mac(f"bifEleven{chr(65 + i)}", _f(v, 4))
    for f, l in R["sweep"]["regimes"].items():
        cnt = np.bincount(l.ravel(), minlength=4)
        tag = {60: "Sixty", 75: "SeventyFive", 90: "Ninety"}.get(int(round(f * 100)), "X")
        mac(f"regNormal{tag}", str(cnt[0]))
        mac(f"regImp{tag}", str(cnt[1]))
    ns = R["landscape"]["n_switch"]
    for sg, tag in ((0.05, "A"), (0.055, "B"), (0.06, "C"), (0.07, "D")):
        mac(f"nTrans{tag}", str(ns.get(sg, "n/a")))
    mac("landT", f"{R['landscape']['T']:.0f}")
    (rep_dir / "generated_numbers.tex").write_text("\n".join(L) + "\n")

    rows = []
    for i, c in enumerate(B["coh"]):
        err = _f(1000 * S["rt_err"][i, km], 0) if S["n_err"][i, km] >= 5 else "--"
        rows.append(f"{c:g} & {_f(S['acc'][i, km], 3)} & {_f(1000 * S['rt_corr'][i, km], 0)} & {err} & "
                    f"{_f(100 * S['p_nodec'][i, km], 1)} \\\\")
    (rep_dir / "table_baseline.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    rows = []
    names = {"sigma": r"$\sigma$", "param": r"$J_{12},\sigma$", "lam_sigma": r"$\log\lambda_u,\sigma$",
             "geo": r"$\log\lambda_u,d,\sigma$", "geo_plus": r"$\log\lambda_u,d,d_{\rm sad},\sigma$"}
    for m, nm in names.items():
        rows.append(f"{nm} & {_f(h1['log_rt'][m]['r2'], 2)} & {_f(h1['log_rt'][m]['cv_r2'], 2)} & "
                    f"{_f(h1['logit_err'][m]['r2'], 2)} & {_f(h1['logit_err'][m]['cv_r2'], 2)} \\\\")
    (rep_dir / "table_h1.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def write_all(R, res_dir: Path, rep_dir: Path):
    write_tables(R, res_dir)
    write_latex(R, rep_dir)
