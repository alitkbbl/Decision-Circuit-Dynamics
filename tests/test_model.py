"""Unit tests: run with `pytest -q` from the project root."""
import numpy as np
import pytest

from wwdm import analysis as an
from wwdm.model import Params, H, dH_dx, jacobian, run_trials, vector_field

P = Params()


def test_H_limits():
    # H(x) -> 1/d at a x = b ; ~0 far below ; ~ a x - b far above
    assert H(P.b / P.a, P) == pytest.approx(1 / P.d, rel=1e-6)
    assert H(0.0, P) < 1e-3
    x = 2.0
    assert H(x, P) == pytest.approx(P.a * x - P.b, rel=1e-6)


def test_dH_matches_finite_difference():
    for x in (0.2, 0.3255, 0.4, 0.5):
        h = 1e-6
        fd = (H(x + h, P) - H(x - h, P)) / (2 * h)
        assert dH_dx(x, P) == pytest.approx(fd, rel=1e-5)


def test_jacobian_matches_finite_difference():
    S = np.array([0.3, 0.45])
    J = jacobian(S[0], S[1], P, 12.8, True)
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        fp = np.array(vector_field(*(S + e), P, 12.8, True))
        fm = np.array(vector_field(*(S - e), P, 12.8, True))
        assert np.allclose(J[:, j], (fp - fm) / (2 * h), rtol=1e-4, atol=1e-6)


def test_baseline_fixed_points_match_wong_wang():
    fps = an.find_fixed_points(P, 0.0, stim=True)
    kinds = sorted(f["kind"] for f in fps)
    assert kinds == [an.STABLE, an.STABLE, an.SADDLE]
    winner = max(max(f["rate"]) for f in fps)
    assert 28 < winner < 32                       # ~30 Hz attractor in the original paper
    for f in fps:                                  # residuals vanish
        assert np.hypot(*vector_field(*f["S"], P, 0.0, True)) < 1e-8
    sad = [f for f in fps if f["kind"] == an.SADDLE][0]
    assert sad["S"][0] == pytest.approx(sad["S"][1], abs=1e-6)   # symmetric saddle at 0 % coherence


def test_symmetry_under_population_swap():
    S = np.array([0.2, 0.5])
    f12 = np.array(vector_field(S[0], S[1], P, 10.0, True))
    f21 = np.array(vector_field(S[1], S[0], P, -10.0, True))
    assert np.allclose(f12, f21[::-1])


def test_separatrix_is_basin_boundary():
    poly = an.separatrix(P, 12.8)
    i = len(poly) // 3
    t = poly[i + 1] - poly[i - 1]
    n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
    a = an.flow_end(P, 12.8, poly[i] + 1e-3 * n)
    b = an.flow_end(P, 12.8, poly[i] - 1e-3 * n)
    assert np.sign(a[0] - a[1]) != np.sign(b[0] - b[1])


def test_simulation_reproducible_and_sensible():
    s0 = an.spontaneous_state(P)
    th = np.array([0.75 * an.attractor_rate(P, 0.0)])
    args = (P.to_array(), 25.6, 300, 7, 5e-4, 0.5, 2.0, th, s0[0], s0[1])
    a, b = run_trials(*args), run_trials(*args)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1], equal_nan=True)
    dec = a[0][:, 0] > 0
    assert dec.mean() > 0.99
    assert (a[0][dec, 0] == 1).mean() > 0.95      # high coherence -> mostly correct


def test_noise_free_deterministic_choice():
    s0 = an.spontaneous_state(P.with_(sigma=0.0))
    th = np.array([0.75 * an.attractor_rate(P, 0.0)])
    ch, rt, pr, _ = run_trials(P.with_(sigma=0.0).to_array(), 12.8, 3, 1, 5e-4, 0.5, 2.0, th, s0[0], s0[1])
    assert (ch[:, 0] == 1).all() and np.allclose(rt[:, 0], rt[0, 0])


def test_wilson_and_weibull():
    lo, hi = an.wilson(50, 100)
    assert lo < 0.5 < hi
    assert an.weibull(1e6, 5, 2) == pytest.approx(1.0)
    assert an.weibull(1e-9, 5, 2) == pytest.approx(0.5)
