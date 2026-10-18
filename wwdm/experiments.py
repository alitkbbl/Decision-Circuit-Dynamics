"""Numerical experiments: baseline behaviour, parameter sweeps, regimes, H1/H2 tests."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import analysis as an
from .model import Params, long_run, run_trials, simulate_traces

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

COH_LEVELS = (0.0, 3.2, 6.4, 12.8, 25.6, 51.2)      # % (as in Roitman & Shadlen 2002)


@dataclass
class Config:
    quick: bool = False
    seed: int = 2026
    f_main: float = 0.75                      # decision threshold = f * attractor rate
    f_grid: tuple = (0.6, 0.75, 0.9)          # thresholds evaluated simultaneously in sweeps
    f_baseline: tuple = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
    dt: float = 5e-4
    t_pre: float = 0.5
    t_max: float = 2.0
    n_baseline: int = 1000
    n_sweep: int = 500
    n_grid: int = 15
    # regime criteria (preset, applied at 12.8 % coherence)
    crit_nodec: float = 0.20                  # indecisive if timeout rate above this
    crit_prem: float = 0.10                   # impulsive if premature-response rate above this
    crit_acc: float = 0.65                    # impulsive if accuracy below this
    J12_range: tuple = (0.015, 0.085)
    J11_range: tuple = (0.215, 0.285)
    sigma_range: tuple = (0.004, 0.080)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.quick:
            self.n_baseline, self.n_sweep, self.n_grid = 200, 100, 7

    @property
    def J12_grid(self):
        return np.linspace(*self.J12_range, self.n_grid)

    @property
    def J11_grid(self):
        return np.linspace(*self.J11_range, self.n_grid)

    @property
    def sigma_grid(self):
        return np.linspace(*self.sigma_range, self.n_grid)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Trial summaries
# --------------------------------------------------------------------------- #

METRICS = ("n_dec", "acc", "p_nodec", "p_prem", "rt_mean", "rt_corr", "rt_err", "rt_med",
           "rt_sd", "r_other")


def summarize(choice, rt, prem, r_other) -> dict:
    """Per-threshold summary statistics.  Inputs have shape (n, K); outputs shape (K,).

    ``decided``  = a threshold crossing after stimulus onset;
    ``p_nodec``  = fraction of trials with no crossing before the 2 s deadline;
    ``p_prem``   = fraction of trials that crossed *before* stimulus onset;
    accuracy is the fraction of decided trials choosing population 1 (the favoured one).
    """
    dec = (choice > 0) & ~prem
    corr = dec & (choice == 1)
    err = dec & (choice == 2)
    n_dec = dec.sum(0)
    with np.errstate(all="ignore"):
        rt_d = np.where(dec, rt, np.nan)
        out = dict(
            n_dec=n_dec.astype(float),
            acc=np.where(n_dec > 0, corr.sum(0) / np.maximum(n_dec, 1), np.nan),
            p_nodec=(choice == 0).mean(0),
            p_prem=prem.mean(0),
            rt_mean=np.nanmean(rt_d, 0),
            rt_corr=np.nanmean(np.where(corr, rt, np.nan), 0),
            rt_err=np.nanmean(np.where(err, rt, np.nan), 0),
            rt_med=np.nanmedian(rt_d, 0),
            rt_sd=np.nanstd(rt_d, 0),
            r_other=np.nanmean(np.where(dec, r_other, np.nan), 0),
        )
    out["n_corr"] = corr.sum(0).astype(float)
    out["n_err"] = err.sum(0).astype(float)
    return out


def run_condition(p: Params, coh: float, n: int, seed: int, thetas, cfg: Config,
                  dt: float | None = None, s0=None):
    """Run ``n`` trials at one coherence; returns raw arrays (n, K)."""
    s0 = an.spontaneous_state(p) if s0 is None else s0
    return run_trials(p.to_array(), float(coh), int(n), int(seed), dt or cfg.dt, cfg.t_pre,
                      cfg.t_max, np.asarray(thetas, float), float(s0[0]), float(s0[1]))


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #

def calibrate(p: Params, cfg: Config) -> dict:
    r_star = an.attractor_rate(p, 0.0)
    return dict(r_star=r_star, theta=cfg.f_main * r_star,
                thetas={f: f * r_star for f in cfg.f_baseline})


def run_baseline(p: Params, cfg: Config) -> dict:
    """Psychometric / chronometric data at baseline parameters (all coherences)."""
    cal = calibrate(p, cfg)
    fs = list(cfg.f_baseline)
    thetas = np.array([f * cal["r_star"] for f in fs])
    s0 = an.spontaneous_state(p)
    k_main = fs.index(cfg.f_main)
    res = dict(coh=np.array(COH_LEVELS), f=np.array(fs), cal=cal, s0=s0, k_main=k_main,
               summary={m: [] for m in METRICS + ("n_corr", "n_err")}, raw={})
    for i, c in enumerate(COH_LEVELS):
        ch, rt, pr, ro = run_condition(p, c, cfg.n_baseline, cfg.seed + 100000 * (i + 1), thetas,
                                       cfg, s0=s0)
        s = summarize(ch, rt, pr, ro)
        for m in s:
            res["summary"][m].append(s[m])
        res["raw"][c] = dict(choice=ch[:, k_main].copy(), rt=rt[:, k_main].copy(),
                             prem=pr[:, k_main].copy())
    res["summary"] = {m: np.array(v) for m, v in res["summary"].items()}      # (n_coh, K)
    km = k_main
    S = res["summary"]
    res["weibull"] = an.fit_weibull(res["coh"], S["n_corr"][:, km], S["n_dec"][:, km])
    return res


def run_traces(p: Params, cfg: Config, coh: float = 12.8, n: int = 40) -> dict:
    cal = calibrate(p, cfg)
    t, tr, out = simulate_traces(p, coh, n, cfg.seed + 7, cal["theta"], dt=cfg.dt,
                                 t_pre=cfg.t_pre, t_max=cfg.t_max)
    return dict(t=t, traces=tr, coh=coh, theta=cal["theta"], **out)


def run_dt_sensitivity(p: Params, cfg: Config) -> dict:
    """Baseline behaviour for several integration time steps (same threshold)."""
    cal = calibrate(p, cfg)
    s0 = an.spontaneous_state(p)
    dts = (1e-3, 5e-4, 2.5e-4, 1e-4)
    n = max(cfg.n_baseline // (1 if not cfg.quick else 2), 100)
    out = dict(dts=np.array(dts), coh=np.array(COH_LEVELS), acc=[], rt=[], nodec=[])
    for dt in dts:
        a, r, nd = [], [], []
        for i, c in enumerate(COH_LEVELS):
            ch, rt, pr, ro = run_condition(p, c, n, cfg.seed + 555 + 1000 * i, [cal["theta"]], cfg,
                                           dt=dt, s0=s0)
            s = summarize(ch, rt, pr, ro)
            a.append((s["acc"][0], s["n_corr"][0], s["n_dec"][0]))
            r.append(s["rt_mean"][0])
            nd.append(s["p_nodec"][0])
        out["acc"].append(a)
        out["rt"].append(r)
        out["nodec"].append(nd)
    out = {k: (np.array(v) if isinstance(v, list) else v) for k, v in out.items()}
    out["n"] = n
    return out


# --------------------------------------------------------------------------- #
# Parameter sweeps
# --------------------------------------------------------------------------- #

def run_grid(base: Params, xname: str, xs, sigmas, cohs, n: int, thetas, cfg: Config,
             seed0: int) -> dict:
    """Behaviour over a (parameter ``xname``, sigma) grid.

    Output arrays have shape (n_x, n_sigma, n_coh, K); ``K`` is the number of decision
    thresholds evaluated simultaneously.
    """
    nx, ns, nc, K = len(xs), len(sigmas), len(cohs), len(thetas)
    G = {m: np.full((nx, ns, nc, K), np.nan) for m in METRICS + ("n_corr", "n_err")}
    starts = []
    t0 = time.time()
    for i, x in enumerate(xs):
        p_x = base.with_(**{xname: float(x)})
        s0 = an.spontaneous_state(p_x)
        starts.append(s0)
        for j, sg in enumerate(sigmas):
            p = p_x.with_(sigma=float(sg))
            for k, c in enumerate(cohs):
                seed = seed0 + ((i * ns + j) * nc + k) * 1000
                ch, rt, pr, ro = run_condition(p, c, n, seed, thetas, cfg, s0=s0)
                s = summarize(ch, rt, pr, ro)
                for m in G:
                    G[m][i, j, k, :] = s[m]
        if (i + 1) % max(1, nx // 5) == 0:
            log(f"   {xname} {i + 1}/{nx} done ({time.time() - t0:.0f}s)")
    G.update(xname=xname, xs=np.asarray(xs), sigmas=np.asarray(sigmas), cohs=np.asarray(cohs),
             thetas=np.asarray(thetas), n=n, starts=np.array(starts))
    return G


def classify_regimes(G: dict, cfg: Config, k_th: int, coh12: float = 12.8) -> np.ndarray:
    """Operational regime labels on the (x, sigma) grid.

    0 = normal, 1 = impulsive, 2 = indecisive, 3 = other (e.g. RT does not fall with coherence).
    Criteria (preset, see ``Config``) are evaluated at 12.8 % coherence.
    """
    kc12 = int(np.argmin(np.abs(G["cohs"] - coh12)))
    kc0 = int(np.argmin(np.abs(G["cohs"] - 0.0)))
    acc, nod = G["acc"][:, :, kc12, k_th], G["p_nodec"][:, :, kc12, k_th]
    prem = G["p_prem"][:, :, kc12, k_th]
    rt12, rt0 = G["rt_mean"][:, :, kc12, k_th], G["rt_mean"][:, :, kc0, k_th]
    lab = np.zeros(acc.shape, int)
    lab[(rt12 >= rt0) | np.isnan(rt12)] = 3
    lab[(acc < cfg.crit_acc) | (prem > cfg.crit_prem)] = 1
    lab[nod > cfg.crit_nodec] = 2
    return lab


def _crossing(xs, ys, level, direction):
    """First linear-interpolated crossing of ``level``; direction 'down' or 'up'."""
    for a in range(len(xs) - 1):
        y0, y1 = ys[a], ys[a + 1]
        if np.isnan(y0) or np.isnan(y1):
            continue
        hit = (y0 >= level > y1) if direction == "down" else (y0 <= level < y1)
        if hit:
            return float(xs[a] + (level - y0) * (xs[a + 1] - xs[a]) / (y1 - y0))
    return np.nan


# --------------------------------------------------------------------------- #
# Geometry of the (J12) sweep and the hypotheses
# --------------------------------------------------------------------------- #

def geometry_scan(base: Params, xname: str, xs, coh: float = 12.8) -> dict:
    """Deterministic descriptors for each value of ``xname`` (they do not depend on sigma)."""
    rows = dict(lam_u=[], lam_s=[], d=[], d_saddle=[], n_stable=[], n_saddle=[], lam_sp=[],
                d_barrier=[], r_att=[], lam_u0=[])
    polys, saddles = [], []
    for x in xs:
        p = base.with_(**{xname: float(x)})
        g = an.geometry(p, coh)
        g0 = an.geometry(p, 0.0)
        sg = an.spontaneous_geometry(p)
        for k, v in dict(lam_u=g["lam_u"], lam_s=g["lam_s"], d=g["d"], d_saddle=g["d_saddle"],
                         n_stable=g["n_stable"], n_saddle=g["n_saddle"], lam_sp=sg["lam_sp"],
                         d_barrier=sg["d_barrier"], r_att=an.attractor_rate(p, 0.0),
                         lam_u0=g0["lam_u"]).items():
            rows[k].append(v)
        polys.append(g["poly"])
        saddles.append(g["saddle"])
    out = {k: np.array(v, float) for k, v in rows.items()}
    out["polys"] = polys
    out["saddles"] = np.array(saddles)
    out["xs"] = np.asarray(xs)
    return out


def emp_logit(k, n):
    return np.log((k + 0.5) / (n - k + 0.5))


def analyze_h1(G: dict, geo: dict, k_th: int, coh12: float = 12.8) -> dict:
    """H1: are RT and error rate explained by (lambda_u, d, sigma)?  OLS across grid cells."""
    kc = int(np.argmin(np.abs(G["cohs"] - coh12)))
    nx, ns = len(G["xs"]), len(G["sigmas"])
    X_, S_ = np.meshgrid(G["xs"], G["sigmas"], indexing="ij")
    rt = G["rt_mean"][:, :, kc, k_th]
    nd = G["n_dec"][:, :, kc, k_th]
    err = G["n_err"][:, :, kc, k_th]
    ok = (G["p_prem"][:, :, kc, k_th] < 0.05) & (G["p_nodec"][:, :, kc, k_th] < 0.05) & (nd >= 50)
    ok &= np.isfinite(geo["lam_u"])[:, None] & np.isfinite(geo["d"])[:, None]
    ok &= np.isfinite(rt)
    lam = np.repeat(geo["lam_u"][:, None], ns, 1)
    d = np.repeat(geo["d"][:, None], ns, 1)
    dsad = np.repeat(geo["d_saddle"][:, None], ns, 1)
    grp = np.repeat(np.arange(nx)[:, None], ns, 1)
    y_rt = np.log(rt)
    y_er = err / np.maximum(nd, 1)
    out = dict(mask=ok, n=int(ok.sum()), xs=G["xs"], sigmas=G["sigmas"])
    sel = ok.ravel()
    feats = {
        "geo": (np.column_stack([np.log(lam.ravel()), d.ravel(), S_.ravel()]), ["log(lambda_u)", "d", "sigma"]),
        "param": (np.column_stack([X_.ravel(), S_.ravel()]), [G["xname"], "sigma"]),
        "sigma": (S_.ravel()[:, None], ["sigma"]),
        "geo_plus": (np.column_stack([np.log(lam.ravel()), d.ravel(), dsad.ravel(), S_.ravel()]),
                     ["log(lambda_u)", "d", "d_saddle", "sigma"]),
        "lam_sigma": (np.column_stack([np.log(lam.ravel()), S_.ravel()]), ["log(lambda_u)", "sigma"]),
        "geo_nosigma": (np.column_stack([np.log(lam.ravel()), d.ravel()]), ["log(lambda_u)", "d"]),
    }
    for tname, y in (("log_rt", y_rt), ("logit_err", y_er)):
        out[tname] = {}
        for mname, (X, names) in feats.items():
            if tname == "log_rt":          # OLS on log mean RT
                out[tname][mname] = an.ols(X[sel], y.ravel()[sel], names, groups=grp.ravel()[sel])
            else:                          # binomial GLM on error counts (McFadden pseudo-R^2)
                out[tname][mname] = an.binom_glm(X[sel], err.ravel()[sel], nd.ravel()[sel], names,
                                                 groups=grp.ravel()[sel])
        out[tname]["y"] = y
    out["corr_lam_d"] = float(np.corrcoef(lam.ravel()[sel], d.ravel()[sel])[0, 1])
    return out


def analyze_h2_j12(G: dict, geo: dict, bif: dict, cfg: Config, k_th: int) -> dict:
    """H2 for the (J12, sigma) sweep: where do behavioural boundaries lie relative to the
    deterministic bifurcations, and how does noise move them?"""
    kc = int(np.argmin(np.abs(G["cohs"] - 12.8)))
    xs, sg = G["xs"], G["sigmas"]
    s_acc = np.array([_crossing(sg, G["acc"][i, :, kc, k_th], cfg.crit_acc, "down") for i in range(len(xs))])
    s_prem = np.array([_crossing(sg, G["p_prem"][i, :, kc, k_th], cfg.crit_prem, "up") for i in range(len(xs))])
    out = dict(sigma_acc=s_acc, sigma_prem=s_prem, xs=xs)
    for name, y in (("acc", s_acc), ("prem", s_prem)):
        m = np.isfinite(y)
        if m.sum() >= 4:
            r = stats.spearmanr(xs[m], y[m])
            out[f"spearman_{name}"] = (float(r.statistic), float(r.pvalue))
            lin = stats.linregress(xs[m], y[m])
            out[f"slope_{name}"] = (float(lin.slope), float(lin.intercept), float(lin.rvalue ** 2))
    m = np.isfinite(s_prem) & np.isfinite(geo["d_barrier"])
    if m.sum() >= 4:
        lin = stats.linregress(geo["d_barrier"][m], s_prem[m])
        out["prem_vs_barrier"] = dict(slope=float(lin.slope), intercept=float(lin.intercept),
                                      r2=float(lin.rvalue ** 2), n=int(m.sum()))
        out["prem_vs_barrier_rho"] = float(stats.spearmanr(geo["d_barrier"][m], s_prem[m]).statistic)
    # Is there a discontinuity of the boundary at the bifurcation values?  Compare the mean
    # absolute second difference of the boundary at bifurcation-straddling vs other J12 values.
    out["bif"] = bif
    return out


def analyze_h2_j11(G: dict, cfg: Config, k_th: int, theta: float, p0: Params, bif_j11: list) -> dict:
    """H2 for the (J11, sigma) sweep: indecisive boundary vs. deterministic thresholds."""
    kc = int(np.argmin(np.abs(G["cohs"] - 12.8)))
    xs, sg = G["xs"], G["sigmas"]
    # scan from strong to weak recurrent excitation: the timeout rate rises through the criterion
    j11_c = np.array([_crossing(xs[::-1], G["p_nodec"][::-1, j, kc, k_th], cfg.crit_nodec, "up")
                      for j in range(len(sg))])
    # deterministic J11 at which the choice-attractor rate equals the criterion theta
    grid = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 61)
    rates = np.array([an.attractor_rate(p0.with_(J11=float(g)), 0.0) for g in grid])
    j11_theta = np.nan
    for a in range(len(grid) - 1):
        if np.isfinite(rates[a]) and np.isfinite(rates[a + 1]) and rates[a] < theta <= rates[a + 1]:
            j11_theta = float(grid[a] + (theta - rates[a]) * (grid[a + 1] - grid[a]) / (rates[a + 1] - rates[a]))
    out = dict(sigmas=sg, j11_boundary=j11_c, j11_theta=j11_theta, bif_j11=bif_j11, theta=theta)
    m = np.isfinite(j11_c)
    if m.sum() >= 4:
        lin = stats.linregress(sg[m], j11_c[m])
        out["shift"] = dict(slope=float(lin.slope), intercept=float(lin.intercept),
                            r2=float(lin.rvalue ** 2), n=int(m.sum()))
    return out


# --------------------------------------------------------------------------- #
# Empirical landscape
# --------------------------------------------------------------------------- #

def run_landscape(p: Params, cfg: Config) -> dict:
    """Empirical quasi-potential  U = -log P  from simulation (non-gradient system)."""
    out = {}
    cal = calibrate(p, cfg)
    # (a) decision-trial ensemble at baseline noise, 0 % coherence: occupancy after onset
    n = 150 if not cfg.quick else 40
    t, tr, _ = simulate_traces(p, 0.0, n, cfg.seed + 999, cal["theta"], dt=cfg.dt,
                               t_pre=cfg.t_pre, t_max=cfg.t_max)
    sel = (t >= 0) & (t <= 1.5)
    out["ensemble"] = tr[:, sel, :].reshape(-1, 2)
    # (b) stationary long runs at elevated noise, 0 % coherence (stimulus on)
    T = 20000.0 if not cfg.quick else 2000.0
    out["long"] = {}
    for sg in (0.05, 0.055, 0.06, 0.07):
        X = long_run(p.with_(sigma=sg), 0.0, T, cfg.seed + int(sg * 1000), s0=(0.05, 0.65),
                     every=100, burn_time=2.0)
        out["long"][sg] = X
        d = X[:, 0] - X[:, 1]
        st = np.where(d > 0.35, 1, np.where(d < -0.35, -1, 0))
        st = st[st != 0]
        sw = int(np.sum(np.diff(st) != 0))          # genuine transitions between choice states
        out.setdefault("n_switch", {})[sg] = sw
        log(f"   landscape sigma={sg}: {len(X)} samples, {sw} choice-to-choice transitions")
    out["T"] = T
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_all(cfg: Config) -> dict:
    p = Params()
    R = dict(cfg=cfg, params=p)
    log("Calibration and baseline ...")
    R["baseline"] = run_baseline(p, cfg)
    cal = R["baseline"]["cal"]
    log(f"   attractor rate r* = {cal['r_star']:.2f} Hz, theta = {cal['theta']:.2f} Hz")
    R["traces"] = run_traces(p, cfg)
    log("dt sensitivity ...")
    R["dt_sens"] = run_dt_sensitivity(p, cfg)

    thetas = np.array([f * cal["r_star"] for f in cfg.f_grid])
    k_th = list(cfg.f_grid).index(cfg.f_main)
    R["k_th"] = k_th
    cohs = (0.0, 12.8)

    log("Deterministic bifurcation analysis ...")
    fine = np.linspace(0.0, 0.14, 141 if not cfg.quick else 57)
    R["bif"] = dict(
        J12_grid=fine,
        scan={c: an.bifurcation_scan(p, np.linspace(0.0, 0.14, 281 if not cfg.quick else 71), c, True)
              for c in (0.0, 12.8)},
        scan_off=an.bifurcation_scan(p, np.linspace(0.0, 0.14, 281 if not cfg.quick else 71), 0.0, False),
        trans={c: an.locate_transitions(p, fine, c, True) for c in (0.0, 12.8)},
        trans_off=an.locate_transitions(p, fine, 0.0, False),
        coh_axis=np.linspace(0, 51.2, 33 if not cfg.quick else 13),
        J12_axis=np.linspace(0.0, 0.12, 97 if not cfg.quick else 31),
    )
    R["bif"]["count_map"] = an.stable_count_map(p, R["bif"]["J12_axis"], R["bif"]["coh_axis"])
    fine11 = np.linspace(0.20, 0.30, 101 if not cfg.quick else 41)
    R["bif"]["trans_J11"] = {c: an.locate_transitions(p, fine11, c, True, param="J11") for c in (0.0, 12.8)}

    log(f"Main sweep (J12 x sigma): {cfg.n_grid}x{cfg.n_grid} cells, {cfg.n_sweep} trials, 2 coherences ...")
    R["sweep"] = run_grid(p, "J12", cfg.J12_grid, cfg.sigma_grid, cohs, cfg.n_sweep, thetas, cfg,
                          cfg.seed + 10_000_000)
    R["sweep"]["regimes"] = {f: classify_regimes(R["sweep"], cfg, k) for k, f in enumerate(cfg.f_grid)}
    log("Geometry of the J12 sweep ...")
    R["geo"] = geometry_scan(p, "J12", cfg.J12_grid, 12.8)
    log("Hypothesis tests ...")
    R["h1"] = analyze_h1(R["sweep"], R["geo"], k_th)
    bif_all = dict(R["bif"]["trans"])
    R["h2"] = analyze_h2_j12(R["sweep"], R["geo"], bif_all, cfg, k_th)

    log("Supplementary sweep (J11 x sigma) ...")
    R["sweep11"] = run_grid(p, "J11", cfg.J11_grid, cfg.sigma_grid, cohs, max(cfg.n_sweep // 2, 100),
                            thetas, cfg, cfg.seed + 20_000_000)
    R["sweep11"]["regimes"] = {f: classify_regimes(R["sweep11"], cfg, k) for k, f in enumerate(cfg.f_grid)}
    R["h2_j11"] = analyze_h2_j11(R["sweep11"], cfg, k_th, cal["theta"], p, R["bif"]["trans_J11"][0.0])

    log("Empirical landscape ...")
    R["landscape"] = run_landscape(p, cfg)
    return R
