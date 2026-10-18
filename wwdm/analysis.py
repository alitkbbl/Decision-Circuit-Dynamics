"""Dynamical-systems analysis and statistics for the Wong-Wang decision circuit."""
from __future__ import annotations

import numba as nb
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .model import (P_A, P_B, P_D, P_GAMMA, P_I0, P_J11, P_J12, P_JEXT, P_MU0, P_TAU,
                    Params, _H, jacobian, rate_from_S, vector_field)

STABLE, SADDLE, UNSTABLE = 0, 1, 2
KIND_NAMES = {STABLE: "stable", SADDLE: "saddle", UNSTABLE: "unstable"}

# --------------------------------------------------------------------------- #
# Fast deterministic flow (Numba) -- used for basins and the spontaneous state
# --------------------------------------------------------------------------- #


@nb.njit(cache=True)
def _flow(par, coh, stim, s01, s02, T, dt):
    tau, gam, a, b, d = par[P_TAU], par[P_GAMMA], par[P_A], par[P_B], par[P_D]
    j11, j12, i0 = par[P_J11], par[P_J12], par[P_I0]
    ext = par[P_JEXT] * par[P_MU0]
    e1 = ext * (1.0 + coh / 100.0) if stim else 0.0
    e2 = ext * (1.0 - coh / 100.0) if stim else 0.0
    S1, S2 = s01, s02
    for _ in range(int(T / dt)):
        x1 = j11 * S1 - j12 * S2 + i0 + e1
        x2 = j11 * S2 - j12 * S1 + i0 + e2
        S1n = S1 + dt * (-S1 / tau + (1.0 - S1) * gam * _H(x1, a, b, d))
        S2n = S2 + dt * (-S2 / tau + (1.0 - S2) * gam * _H(x2, a, b, d))
        S1 = min(max(S1n, 0.0), 1.0)
        S2 = min(max(S2n, 0.0), 1.0)
    return S1, S2


@nb.njit(cache=True)
def _basin_grid(par, coh, stim, g1, g2, T, dt):
    n1, n2 = g1.shape[0], g2.shape[0]
    out = np.zeros((n2, n1))
    for i in range(n1):
        for j in range(n2):
            s1, s2 = _flow(par, coh, stim, g1[i], g2[j], T, dt)
            out[j, i] = s1 - s2
    return out


def basin_grid(p: Params, coh: float, n: int = 140, T: float = 6.0, dt: float = 2e-3,
               stim: bool = True):
    """Final value of S1 - S2 of the deterministic flow from a grid of initial states.

    Returns ``(g1, g2, D)`` with ``D[j, i]`` the end point of the trajectory started at
    ``(g1[i], g2[j])``; sign(D) identifies the basin (positive: population 1 wins).
    """
    g = np.linspace(0.0, 0.95, n)
    return g, g, _basin_grid(p.to_array(), float(coh), stim, g, g, T, dt)


def flow_end(p: Params, coh: float, S0, stim: bool = True, T: float = 6.0, dt: float = 2e-3):
    return np.array(_flow(p.to_array(), float(coh), stim, float(S0[0]), float(S0[1]), T, dt))


# --------------------------------------------------------------------------- #
# Fixed points
# --------------------------------------------------------------------------- #

def _start_grid() -> np.ndarray:
    lo = np.linspace(0.0, 0.12, 7)
    hi = np.linspace(0.16, 0.95, 11)
    g = np.concatenate([lo, hi])
    A, B = np.meshgrid(g, g)
    return np.column_stack([A.ravel(), B.ravel()])


_STARTS = _start_grid()


def _newton(starts, p, coh, stim, iters=80, tol=1e-9):
    S = starts.copy()
    for _ in range(iters):
        f1, f2 = vector_field(S[:, 0], S[:, 1], p, coh, stim)
        J = jacobian(S[:, 0], S[:, 1], p, coh, stim)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        d1 = (J[:, 1, 1] * f1 - J[:, 0, 1] * f2) / det
        d2 = (-J[:, 1, 0] * f1 + J[:, 0, 0] * f2) / det
        nrm = np.maximum(np.hypot(d1, d2), 1e-300)
        scale = np.minimum(1.0, 0.15 / nrm)               # damped Newton
        S = np.clip(S - np.column_stack([d1 * scale, d2 * scale]), 0.0, 1.0)
    f1, f2 = vector_field(S[:, 0], S[:, 1], p, coh, stim)
    ok = (np.hypot(f1, f2) < tol) & (S > 1e-9).all(1) & (S < 1 - 1e-9).all(1)
    return S[ok]


def find_fixed_points(p: Params, coh: float = 0.0, stim: bool = True):
    """All fixed points in the unit square, classified by Jacobian eigenvalues.

    Returns a list of dicts sorted by ``S1 - S2`` with keys ``S`` (2,), ``eig`` (2,
    real parts, descending), ``kind`` (STABLE / SADDLE / UNSTABLE), ``rate`` (Hz).
    """
    sols = _newton(_STARTS, p, coh, stim)
    if len(sols) == 0:
        return []
    sols = np.unique(np.round(sols, 6), axis=0)
    out = []
    for s in sols:
        s = _newton(s[None, :], p, coh, stim, iters=10)
        s = s[0] if len(s) else sols[0]
        J = jacobian(s[0], s[1], p, coh, stim)
        w = np.linalg.eigvals(J)
        re = np.sort(w.real)[::-1]
        kind = STABLE if re[0] < 0 else (SADDLE if re[1] < 0 else UNSTABLE)
        out.append(dict(S=s, eig=re, imag=np.abs(w.imag).max(), kind=kind,
                        rate=rate_from_S(s, p)))
    out.sort(key=lambda r: (r["S"][0] - r["S"][1], r["S"][0] + r["S"][1]))
    return out


def spontaneous_state(p: Params) -> np.ndarray:
    """Stimulus-free resting state: lowest-activity stable fixed point of the stim-off system."""
    fps = [f for f in find_fixed_points(p, 0.0, stim=False) if f["kind"] == STABLE]
    if fps:
        return min(fps, key=lambda f: f["S"].sum())["S"].copy()
    return flow_end(p, 0.0, (0.01, 0.01), stim=False, T=5.0, dt=5e-3)


def attractor_rate(p: Params, coh: float = 0.0, min_asym: float = 5.0) -> float:
    """Firing rate [Hz] of the winning population in a *choice* attractor (stimulus on).

    Only stable states with |r1 - r2| > ``min_asym`` Hz count as choice attractors; NaN if
    there is none.
    """
    fps = [f for f in find_fixed_points(p, coh, True)
           if f["kind"] == STABLE and abs(f["rate"][0] - f["rate"][1]) > min_asym]
    if not fps:
        return float("nan")
    return float(max(max(f["rate"]) for f in fps))


def spontaneous_geometry(p: Params) -> dict:
    """Geometry of the stimulus-free system around the resting state.

    ``lam_sp``: slowest (least negative) eigenvalue at the resting state [1/s];
    ``d_barrier``: distance (S units) from the resting state to the nearest saddle, i.e. a
    deterministic proxy for the barrier that noise must overcome to escape prematurely.
    """
    start = spontaneous_state(p)
    fps = find_fixed_points(p, 0.0, stim=False)
    J = jacobian(start[0], start[1], p, 0.0, False)
    lam = float(np.linalg.eigvals(J).real.max())
    sad = [f for f in fps if f["kind"] == SADDLE]
    d = min((np.linalg.norm(f["S"] - start) for f in sad), default=np.nan)
    return dict(start=start, lam_sp=lam, d_barrier=float(d))


# --------------------------------------------------------------------------- #
# Separatrix and geometric quantities
# --------------------------------------------------------------------------- #

def separatrix(p: Params, coh: float, saddle=None, t_max: float = 80.0, eps: float = 1e-5):
    """Stable manifold of the decision saddle by backward integration.

    Returns an (N, 2) polyline passing through the saddle, or ``None`` if there is no
    saddle.  Integration stops at the boundary of the unit square or near a stationary point.
    """
    if saddle is None:
        sad = [f for f in find_fixed_points(p, coh, True) if f["kind"] == SADDLE]
        if not sad:
            return None
        saddle = min(sad, key=lambda f: abs(f["S"][0] - f["S"][1]))["S"]
    saddle = np.asarray(saddle, float)
    J = jacobian(saddle[0], saddle[1], p, coh, True)
    w, v = np.linalg.eig(J)
    vs = np.real(v[:, np.argmin(w.real)])
    vs /= np.linalg.norm(vs)

    def rhs(_, y):
        f1, f2 = vector_field(y[0], y[1], p, coh, True)
        return [-float(f1), -float(f2)]

    def leave(_, y):
        return min(y[0] + 0.01, y[1] + 0.01, 1.01 - y[0], 1.01 - y[1])

    def slow(_, y):
        f1, f2 = vector_field(y[0], y[1], p, coh, True)
        return np.hypot(float(f1), float(f2)) - 1e-2

    leave.terminal = slow.terminal = True
    leave.direction = slow.direction = -1
    branches = []
    for sgn in (+1.0, -1.0):
        sol = solve_ivp(rhs, (0, t_max), saddle + sgn * eps * vs, events=[leave, slow],
                        rtol=1e-9, atol=1e-12, max_step=0.05)
        branches.append(sol.y.T)
    return np.vstack([branches[1][::-1], saddle[None, :], branches[0]])


def dist_to_polyline(q, poly) -> float:
    """Euclidean distance from point ``q`` to a polyline."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.sqrt(((proj - q) ** 2).sum(1)).min())


def geometry(p: Params, coh: float = 12.8) -> dict:
    """Deterministic geometric descriptors of the stimulus-on phase plane.

    * ``lam_u``  : unstable eigenvalue of the decision saddle [1/s]
    * ``d``      : signed distance (in S units) from the pre-stimulus resting state to the
                   separatrix; positive if the resting state lies in the basin of the
                   favoured population 1 (i.e. the deterministic choice is correct).
    * ``n_stable``, ``n_saddle`` : number of stable fixed points / saddles.
    """
    start = spontaneous_state(p)
    fps = find_fixed_points(p, coh, True)
    stable = [f for f in fps if f["kind"] == STABLE]
    sad = [f for f in fps if f["kind"] == SADDLE]
    out = dict(start=start, n_stable=len(stable), n_saddle=len(sad), lam_u=np.nan, lam_s=np.nan,
               d=np.nan, saddle=np.full(2, np.nan), d_saddle=np.nan, poly=None)
    if not sad:
        return out
    s = min(sad, key=lambda f: np.linalg.norm(f["S"] - start))
    out["lam_u"], out["lam_s"] = float(s["eig"][0]), float(s["eig"][1])
    out["saddle"] = s["S"]
    out["d_saddle"] = float(np.linalg.norm(s["S"] - start))
    poly = separatrix(p, coh, saddle=s["S"])
    out["poly"] = poly
    if poly is not None:
        dist = dist_to_polyline(start, poly)
        end = flow_end(p, coh, start)
        out["d"] = dist if end[0] > end[1] else -dist
    return out


# --------------------------------------------------------------------------- #
# Bifurcation analysis
# --------------------------------------------------------------------------- #

def bifurcation_scan(p: Params, J12_vals, coh: float, stim: bool = True,
                     param: str = "J12") -> dict:
    """Fixed points along a sweep of ``param`` (default J12): all of them, with stability."""
    rows = []
    for j in J12_vals:
        for f in find_fixed_points(p.with_(**{param: float(j)}), coh, stim):
            rows.append((j, f["S"][0], f["S"][1], f["kind"], f["eig"][0], f["eig"][1]))
    a = np.array(rows) if rows else np.zeros((0, 6))
    return dict(J12=a[:, 0], S1=a[:, 1], S2=a[:, 2], kind=a[:, 3].astype(int),
                eig1=a[:, 4], eig2=a[:, 5])


def _signature(p: Params, coh: float, stim: bool):
    fps = find_fixed_points(p, coh, stim)
    return (sum(f["kind"] == STABLE for f in fps), sum(f["kind"] == SADDLE for f in fps))


def locate_transitions(p: Params, J12_grid, coh: float, stim: bool = True, n_bisect: int = 18,
                       param: str = "J12"):
    """Values of ``param`` (default J12) at which the number of (stable, saddle) fixed
    points changes.  Returns a list of ``(value, signature_before, signature_after)``.
    """
    sigs = [_signature(p.with_(**{param: float(j)}), coh, stim) for j in J12_grid]
    out = []
    for i in range(len(J12_grid) - 1):
        if sigs[i] != sigs[i + 1]:
            lo, hi = float(J12_grid[i]), float(J12_grid[i + 1])
            slo = sigs[i]
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                if _signature(p.with_(**{param: mid}), coh, stim) == slo:
                    lo = mid
                else:
                    hi = mid
            out.append((0.5 * (lo + hi), sigs[i], sigs[i + 1]))
    return out


def stable_count_map(p: Params, J12_vals, coh_vals) -> np.ndarray:
    """Number of stable fixed points (stimulus on) over the (coherence, J12) plane."""
    out = np.zeros((len(coh_vals), len(J12_vals)), int)
    for i, c in enumerate(coh_vals):
        for j, J in enumerate(J12_vals):
            out[i, j] = _signature(p.with_(J12=float(J)), float(c), True)[0]
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def wilson(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    n_ = np.maximum(n, 1)
    ph = k / n_
    den = 1 + z ** 2 / n_
    c = (ph + z ** 2 / (2 * n_)) / den
    h = z * np.sqrt(ph * (1 - ph) / n_ + z ** 2 / (4 * n_ ** 2)) / den
    return c - h, c + h


def weibull(c, alpha, beta):
    return 1.0 - 0.5 * np.exp(-(np.asarray(c, float) / alpha) ** beta)


def fit_weibull(coh, k_correct, n):
    """Maximum-likelihood Weibull psychometric fit P = 1 - 0.5 exp(-(c/alpha)^beta)."""
    coh, k, n = (np.asarray(v, float) for v in (coh, k_correct, n))
    m = coh > 0

    def nll(th):
        pr = np.clip(weibull(coh[m], np.exp(th[0]), np.exp(th[1])), 1e-9, 1 - 1e-9)
        return -(k[m] * np.log(pr) + (n[m] - k[m]) * np.log(1 - pr)).sum()

    best = min((minimize(nll, x0, method="Nelder-Mead") for x0 in
                ([np.log(8), np.log(1.5)], [np.log(3), np.log(1.0)], [np.log(20), np.log(2.5)])),
               key=lambda r: r.fun)
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def ols(X, y, names, groups=None):
    """Ordinary least squares with standardised coefficients and drop-one importance.

    ``X`` is (n, k) without intercept.  ``groups`` (optional) makes the 5-fold cross-validation
    hold out whole groups (e.g. all cells sharing one J12).  Returns a dict with R^2, adjusted R^2, CV R^2,
    raw and standardised coefficients, t-values, and the loss in R^2 when each regressor is
    removed (``drop_r2``).
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd > 0, sd, 1)
    A = np.column_stack([np.ones(n), Z])
    ys = y.std() if y.std() > 0 else 1.0

    def fit(A_, y_):
        beta, *_ = np.linalg.lstsq(A_, y_, rcond=None)
        return beta

    beta = fit(A, y)
    res = y - A @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res ** 2).sum() / ss_tot
    dof = max(n - k - 1, 1)
    s2 = (res ** 2).sum() / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    tvals = beta / np.sqrt(np.diag(cov))
    drops = {}
    for j in range(k):
        keep = [0] + [i + 1 for i in range(k) if i != j]
        b_ = fit(A[:, keep], y)
        r_ = y - A[:, keep] @ b_
        drops[names[j]] = float(r2 - (1 - (r_ ** 2).sum() / ss_tot))
    if groups is None:
        folds = np.random.default_rng(0).permutation(n) % 5
    else:                                  # leave-group-out: whole groups are held out together
        _, gi = np.unique(np.asarray(groups), return_inverse=True)
        folds = gi % 5
    pred = np.empty(n)
    for f in np.unique(folds):
        tr = folds != f
        pred[~tr] = A[~tr] @ fit(A[tr], y[tr])
    cv = 1 - ((y - pred) ** 2).sum() / ss_tot
    return dict(n=int(n), r2=float(r2), adj_r2=float(1 - (1 - r2) * (n - 1) / dof), cv_r2=float(cv),
                std_coef=dict(zip(names, (beta[1:] / ys).tolist())),
                coef=dict(zip(names, (beta[1:] / np.where(sd > 0, sd, 1)).tolist())),
                intercept=float(beta[0] - (beta[1:] / np.where(sd > 0, sd, 1) * mu).sum()),
                t=dict(zip(names, tvals[1:].tolist())), drop_r2=drops, pred=A @ beta)


def _binom_dev(k, n, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(k > 0, k * np.log(k / (n * p)), 0.0)
        b = np.where(n - k > 0, (n - k) * np.log((n - k) / (n * (1 - p))), 0.0)
    return 2.0 * np.sum(a + b)


def _irls(A, k, n, iters=60):
    pbar = np.clip(k.sum() / n.sum(), 1e-6, 1 - 1e-6)
    beta = np.zeros(A.shape[1])
    beta[0] = np.log(pbar / (1 - pbar))
    for _ in range(iters):
        eta = np.clip(A @ beta, -30, 30)
        pr = 1 / (1 + np.exp(-eta))
        w = n * pr * (1 - pr) + 1e-9
        z = eta + (k - n * pr) / w
        Aw = A * w[:, None]
        new = np.linalg.solve(A.T @ Aw + 1e-6 * np.eye(A.shape[1]), Aw.T @ z)
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    return beta


def binom_glm(X, k, n, names, groups=None):
    """Logistic (binomial) regression of counts ``k`` out of ``n`` on standardised ``X``.

    Returns McFadden pseudo-R^2 (``r2``), leave-group-out cross-validated pseudo-R^2
    (``cv_r2``, based on held-out deviance), standardised coefficients (log-odds per SD),
    drop-one importance and fitted probabilities ``pred``.
    """
    X, k, n = np.asarray(X, float), np.asarray(k, float), np.asarray(n, float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A = np.column_stack([np.ones(len(k)), (X - mu) / sd])
    beta = _irls(A, k, n)
    pr = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
    pbar = k.sum() / n.sum()
    dev0 = _binom_dev(k, n, np.full_like(pr, pbar))
    r2 = 1 - _binom_dev(k, n, pr) / dev0
    drops = {}
    for j in range(X.shape[1]):
        keep = [0] + [i + 1 for i in range(X.shape[1]) if i != j]
        b_ = _irls(A[:, keep], k, n)
        p_ = 1 / (1 + np.exp(-np.clip(A[:, keep] @ b_, -30, 30)))
        drops[names[j]] = float(r2 - (1 - _binom_dev(k, n, p_) / dev0))
    if groups is None:
        folds = np.random.default_rng(0).permutation(len(k)) % 5
    else:
        _, gi = np.unique(np.asarray(groups), return_inverse=True)
        folds = gi % 5
    d_m = d_0 = 0.0
    for f in np.unique(folds):
        te = folds == f
        b_ = _irls(A[~te], k[~te], n[~te])
        p_ = 1 / (1 + np.exp(-np.clip(A[te] @ b_, -30, 30)))
        d_m += _binom_dev(k[te], n[te], p_)
        d_0 += _binom_dev(k[te], n[te], np.full(te.sum(), k[~te].sum() / n[~te].sum()))
    return dict(n=int(len(k)), r2=float(r2), cv_r2=float(1 - d_m / d_0), std_coef=dict(zip(names, beta[1:].tolist())),
                coef=dict(zip(names, (beta[1:] / sd).tolist())), drop_r2=drops, pred=pr, adj_r2=float(r2))
