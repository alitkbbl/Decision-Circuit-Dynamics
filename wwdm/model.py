"""Reduced two-variable Wong-Wang (2006) decision circuit.

Contents
--------
* ``Params``            : model parameters (seconds, nA, Hz).
* ``H``, ``vector_field``, ``jacobian`` : deterministic model (NumPy, vectorised).
* ``run_trials``        : Monte-Carlo simulator (Numba, one fixed seed per trial).
* ``simulate_traces``   : full single-trial trajectories (for plotting).
* ``long_run``          : one long stimulus-on trajectory (empirical landscape).

Units: time in seconds, currents in nA, rates in Hz.  ``gamma`` is dimensionless
(0.641 here corresponds to 0.641/1000 ms^-1 Hz^-1 in the original paper).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numba as nb
import numpy as np

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Params:
    """Baseline parameters of Wong & Wang (2006)."""

    tau_s: float = 0.100      # NMDA decay time constant [s]
    gamma: float = 0.641      # kinetic parameter [dimensionless]
    a: float = 270.0          # input-output gain [Hz/nA]
    b: float = 108.0          # input-output offset [Hz]
    d: float = 0.154          # input-output curvature [s]
    J11: float = 0.2609       # self-excitation  J_N,11 = J_N,22 [nA]
    J12: float = 0.0497       # cross-inhibition J_N,12 = J_N,21 [nA]
    I0: float = 0.3255        # background input [nA]
    J_ext: float = 5.2e-4     # external coupling J_A,ext [nA/Hz]
    mu0: float = 30.0         # stimulus strength [Hz]
    sigma: float = 0.02       # OU noise amplitude [nA]
    tau_ampa: float = 0.002   # OU correlation time [s]

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def to_array(self) -> np.ndarray:
        return np.array([self.tau_s, self.gamma, self.a, self.b, self.d, self.J11,
                         self.J12, self.I0, self.J_ext, self.mu0, self.sigma,
                         self.tau_ampa], dtype=np.float64)

    @property
    def stim_current(self) -> float:
        """J_ext * mu0 [nA]: total stimulus drive per population at 0 % coherence."""
        return self.J_ext * self.mu0


# index map for the Numba parameter vector
(P_TAU, P_GAMMA, P_A, P_B, P_D, P_J11, P_J12, P_I0, P_JEXT, P_MU0, P_SIG, P_TAUA) = range(12)

# --------------------------------------------------------------------------- #
# Deterministic model (NumPy)
# --------------------------------------------------------------------------- #


def H(x, p: Params):
    """Input-output function H(x) = (a x - b) / (1 - exp(-d (a x - b))) [Hz]."""
    y = p.a * np.asarray(x, dtype=float) - p.b
    z = np.clip(p.d * y, -600.0, 600.0)
    small = np.abs(z) < 1e-7
    zs = np.where(small, 1.0, z)
    return np.where(small, 1.0 / p.d + 0.5 * y, y / (1.0 - np.exp(-zs)))


def dH_dx(x, p: Params):
    """Derivative of H with respect to x [Hz/nA]."""
    y = p.a * np.asarray(x, dtype=float) - p.b
    z = np.clip(p.d * y, -600.0, 600.0)
    small = np.abs(z) < 1e-5
    zs = np.where(small, 1.0, z)
    e = np.exp(-zs)
    dy = ((1.0 - e) - zs * e) / (1.0 - e) ** 2   # dH/dy
    dy_small = 0.5 + p.d * y / 6.0
    return p.a * np.where(small, dy_small, dy)


def stimulus_inputs(p: Params, coh: float, stim: bool = True):
    """External currents (I_ext1, I_ext2) [nA] for signed coherence ``coh`` (%)."""
    if not stim:
        return 0.0, 0.0
    return p.stim_current * (1 + coh / 100.0), p.stim_current * (1 - coh / 100.0)


def total_inputs(S1, S2, p: Params, coh: float = 0.0, stim: bool = True):
    e1, e2 = stimulus_inputs(p, coh, stim)
    x1 = p.J11 * S1 - p.J12 * S2 + p.I0 + e1
    x2 = p.J11 * S2 - p.J12 * S1 + p.I0 + e2
    return x1, x2


def vector_field(S1, S2, p: Params, coh: float = 0.0, stim: bool = True):
    """Deterministic right-hand side (dS1/dt, dS2/dt) [1/s]."""
    x1, x2 = total_inputs(S1, S2, p, coh, stim)
    f1 = -S1 / p.tau_s + (1 - S1) * p.gamma * H(x1, p)
    f2 = -S2 / p.tau_s + (1 - S2) * p.gamma * H(x2, p)
    return f1, f2


def jacobian(S1, S2, p: Params, coh: float = 0.0, stim: bool = True):
    """Analytic Jacobian; returns array of shape (..., 2, 2)."""
    S1 = np.asarray(S1, float)
    S2 = np.asarray(S2, float)
    x1, x2 = total_inputs(S1, S2, p, coh, stim)
    h1, h2 = H(x1, p), H(x2, p)
    g1, g2 = dH_dx(x1, p), dH_dx(x2, p)
    J = np.empty(np.broadcast(S1, S2).shape + (2, 2))
    J[..., 0, 0] = -1 / p.tau_s - p.gamma * h1 + (1 - S1) * p.gamma * g1 * p.J11
    J[..., 0, 1] = -(1 - S1) * p.gamma * g1 * p.J12
    J[..., 1, 0] = -(1 - S2) * p.gamma * g2 * p.J12
    J[..., 1, 1] = -1 / p.tau_s - p.gamma * h2 + (1 - S2) * p.gamma * g2 * p.J11
    return J


def rate_from_S(S, p: Params):
    """Rate-equivalent of a gating variable, r = S / (gamma tau_s (1 - S)) [Hz].

    At every fixed point this equals the firing rate H(x*).  During the dynamics it is a
    smooth (NMDA-filtered) read-out that is insensitive to the 2 ms input noise.
    """
    S = np.asarray(S, float)
    return S / (p.gamma * p.tau_s * (1 - S))


def relax(p: Params, coh: float = 0.0, stim: bool = True, S0=(0.01, 0.01),
          T: float = 15.0, dt: float = 0.002):
    """Deterministic forward-Euler relaxation; returns the final state."""
    S1, S2 = float(S0[0]), float(S0[1])
    for _ in range(int(T / dt)):
        f1, f2 = vector_field(S1, S2, p, coh, stim)
        S1 = min(max(S1 + dt * float(f1), 0.0), 1.0)
        S2 = min(max(S2 + dt * float(f2), 0.0), 1.0)
    return np.array([S1, S2])


# --------------------------------------------------------------------------- #
# Stochastic simulator (Numba)
# --------------------------------------------------------------------------- #

@nb.njit(cache=True, inline="always")
def _H(x, a, b, d):
    y = a * x - b
    z = d * y
    if abs(z) < 1e-7:
        return 1.0 / d + 0.5 * y
    if z < -600.0:
        z = -600.0
    return y / (1.0 - np.exp(-z))


@nb.njit(cache=True)
def _trial(par, coh, dt, t_pre, t_max, thetas, s01, s02, stop, rec_every, trace,
           choice, rt, prem, r_other):
    """One trial of the stochastic model.

    Euler scheme for S_i; exact (Gaussian) transition for the OU input noise.  A decision
    at threshold k is the first time a rate-equivalent r_i = S_i / (gamma tau (1 - S_i))
    exceeds ``thetas[k]``.  A crossing before stimulus onset is flagged as premature.
    Outputs are written in place: ``choice[k]`` (0 none, 1 pop 1, 2 pop 2), ``rt[k]``
    (s after stimulus onset), ``prem[k]`` and ``r_other[k]`` (rate of the losing pop.).
    """
    tau, gam, a, b, d = par[P_TAU], par[P_GAMMA], par[P_A], par[P_B], par[P_D]
    j11, j12, i0 = par[P_J11], par[P_J12], par[P_I0]
    ext = par[P_JEXT] * par[P_MU0]
    sig, taua = par[P_SIG], par[P_TAUA]
    sstat = sig / np.sqrt(2.0)                  # stationary std of the OU input
    rho = np.exp(-dt / taua)
    amp = sstat * np.sqrt(1.0 - rho * rho)
    n_steps = int((t_pre + t_max) / dt)
    K = thetas.shape[0]
    S1, S2 = s01, s02
    n1 = sstat * np.random.randn()
    n2 = sstat * np.random.randn()
    e1 = ext * (1.0 + coh / 100.0)
    e2 = ext * (1.0 - coh / 100.0)
    n_done = 0
    kc = gam * tau
    for k in range(K):
        choice[k] = 0
        rt[k] = np.nan
        prem[k] = False
        r_other[k] = np.nan
    for step in range(n_steps):
        t = step * dt
        x1 = j11 * S1 - j12 * S2 + i0 + n1
        x2 = j11 * S2 - j12 * S1 + i0 + n2
        if t >= t_pre:
            x1 += e1
            x2 += e2
        h1 = _H(x1, a, b, d)
        h2 = _H(x2, a, b, d)
        S1n = S1 + dt * (-S1 / tau + (1.0 - S1) * gam * h1)
        S2n = S2 + dt * (-S2 / tau + (1.0 - S2) * gam * h2)
        S1 = min(max(S1n, 0.0), 1.0)
        S2 = min(max(S2n, 0.0), 1.0)
        n1 = n1 * rho + amp * np.random.randn()
        n2 = n2 * rho + amp * np.random.randn()
        if rec_every > 0 and step % rec_every == 0:
            idx = step // rec_every
            if idx < trace.shape[0]:
                trace[idx, 0] = S1
                trace[idx, 1] = S2
        r1 = S1 / (kc * (1.0 - S1 + 1e-12))
        r2 = S2 / (kc * (1.0 - S2 + 1e-12))
        for k in range(K):
            if choice[k] == 0:
                if r1 >= thetas[k] or r2 >= thetas[k]:
                    if r1 >= r2:
                        choice[k] = 1
                        r_other[k] = r2
                    else:
                        choice[k] = 2
                        r_other[k] = r1
                    rt[k] = t + dt - t_pre
                    prem[k] = t < t_pre
                    n_done += 1
        if stop and n_done == K:
            break
    return S1, S2


@nb.njit(cache=True)
def run_trials(par, coh, n, seed, dt, t_pre, t_max, thetas, s01, s02):
    """Run ``n`` independent trials; trial ``i`` is seeded with ``seed + i`` (reproducible).

    Returns arrays ``choice``, ``rt``, ``prem``, ``r_other`` of shape ``(n, K)``.
    """
    K = thetas.shape[0]
    choice = np.zeros((n, K), np.int8)
    rt = np.full((n, K), np.nan)
    prem = np.zeros((n, K), np.bool_)
    r_other = np.full((n, K), np.nan)
    dummy = np.zeros((1, 2))
    for i in range(n):
        np.random.seed(seed + i)
        _trial(par, coh, dt, t_pre, t_max, thetas, s01, s02, True, 0, dummy,
               choice[i], rt[i], prem[i], r_other[i])
    return choice, rt, prem, r_other


@nb.njit(cache=True)
def _seeded_trial(par, coh, dt, t_pre, t_max, thetas, s0, seed, rec_every, trace,
                  choice, rt, prem, r_other):
    np.random.seed(seed)
    _trial(par, coh, dt, t_pre, t_max, thetas, s0[0], s0[1], False, rec_every, trace,
           choice, rt, prem, r_other)


def simulate_traces(p: Params, coh: float, n: int, seed: int, theta: float, dt: float = 5e-4,
                    t_pre: float = 0.5, t_max: float = 2.0, s0=None, rec_every: int = 4):
    """Full trajectories (no early stopping) for ``n`` trials.

    Returns ``t`` (s, relative to stimulus onset), ``traces`` (n, T, 2) and a dict with the
    decision outcome of each trial.
    """
    from .analysis import spontaneous_state
    if s0 is None:
        s0 = spontaneous_state(p)
    s0 = np.asarray(s0, float)
    par = p.to_array()
    n_rec = int((t_pre + t_max) / dt) // rec_every + 1
    traces = np.full((n, n_rec, 2), np.nan)
    thetas = np.array([theta])
    ch = np.zeros((n, 1), np.int8)
    rt = np.full((n, 1), np.nan)
    pr = np.zeros((n, 1), np.bool_)
    ro = np.full((n, 1), np.nan)
    for i in range(n):
        _seeded_trial(par, float(coh), dt, t_pre, t_max, thetas, s0, seed + i, rec_every,
                      traces[i], ch[i], rt[i], pr[i], ro[i])
    t = np.arange(n_rec) * rec_every * dt - t_pre
    return t, traces, dict(choice=ch[:, 0], rt=rt[:, 0], premature=pr[:, 0])


@nb.njit(cache=True)
def _long_run(par, coh, dt, n_steps, seed, s01, s02, every, burn):
    np.random.seed(seed)
    tau, gam, a, b, d = par[P_TAU], par[P_GAMMA], par[P_A], par[P_B], par[P_D]
    j11, j12, i0 = par[P_J11], par[P_J12], par[P_I0]
    ext = par[P_JEXT] * par[P_MU0]
    sig, taua = par[P_SIG], par[P_TAUA]
    sstat = sig / np.sqrt(2.0)
    rho = np.exp(-dt / taua)
    amp = sstat * np.sqrt(1.0 - rho * rho)
    n1 = sstat * np.random.randn()
    n2 = sstat * np.random.randn()
    e1 = ext * (1.0 + coh / 100.0)
    e2 = ext * (1.0 - coh / 100.0)
    S1, S2 = s01, s02
    n_out = (n_steps - burn) // every
    out = np.empty((n_out, 2))
    for step in range(n_steps):
        x1 = j11 * S1 - j12 * S2 + i0 + n1 + e1
        x2 = j11 * S2 - j12 * S1 + i0 + n2 + e2
        S1 = min(max(S1 + dt * (-S1 / tau + (1.0 - S1) * gam * _H(x1, a, b, d)), 0.0), 1.0)
        S2 = min(max(S2 + dt * (-S2 / tau + (1.0 - S2) * gam * _H(x2, a, b, d)), 0.0), 1.0)
        n1 = n1 * rho + amp * np.random.randn()
        n2 = n2 * rho + amp * np.random.randn()
        if step >= burn and (step - burn) % every == 0:
            j = (step - burn) // every
            if j < n_out:
                out[j, 0] = S1
                out[j, 1] = S2
    return out


def long_run(p: Params, coh: float, T: float, seed: int, s0=(0.1, 0.1), dt: float = 5e-4,
             every: int = 100, burn_time: float = 1.0) -> np.ndarray:
    """One long stimulus-on trajectory sampled every ``every`` steps; returns (N, 2)."""
    n_steps = int(T / dt)
    return _long_run(p.to_array(), float(coh), dt, n_steps, int(seed), float(s0[0]),
                     float(s0[1]), every, int(burn_time / dt))
