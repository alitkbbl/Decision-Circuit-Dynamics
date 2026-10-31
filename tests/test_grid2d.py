"""Tests for the joint (J11, J12) analysis, statistics helpers and WTA check."""
import numpy as np
import pytest

from wwdm import analysis as an
from wwdm import experiments as ex
from wwdm.model import Params

P = Params()


def test_wta_check_baseline_and_failure():
    theta = 0.75 * an.attractor_rate(P, 0.0)
    assert an.is_bistable_wta(P, theta)                          # baseline is a symmetric WTA circuit
    assert not an.is_bistable_wta(P.with_(J12=0.015), theta)     # co-activation: no choice attractors
    assert not an.is_bistable_wta(P.with_(J11=0.20), theta)      # weak recurrent excitation


def test_gradient_angle_parallel_and_orthogonal():
    x = np.linspace(0, 1, 9)
    y = np.linspace(0, 1, 9)
    X, Y = np.meshgrid(x, y, indexing="ij")
    assert np.nanmax(an.gradient_angles(x, y, X + Y, 2 * (X + Y))) < 1e-3        # parallel contours
    ang = an.gradient_angles(x, y, X, Y)                                          # orthogonal
    assert np.allclose(ang, 90, atol=1e-3)


def test_ols_recovers_linear_model_and_reports_metrics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = 1.0 + 2 * X[:, 0] - 0.5 * X[:, 1] + 0.05 * rng.normal(size=200)
    grp = np.repeat(np.arange(40), 5)
    r = an.ols(X, y, ["a", "b"], groups=grp, shuffle_groups=True)
    assert r["r2"] > 0.99 and r["cv_r2"] > 0.99
    assert r["rmse"] == pytest.approx(0.05, abs=0.02)
    assert r["cv_rmse"] >= r["rmse"] * 0.9
    r0 = an.ols(X[:, :1] * 0 + rng.normal(size=(200, 1)), y, ["noise"], groups=grp, shuffle_groups=True)
    assert r0["aic"] > r["aic"]                                  # the true model has the lower AIC


def test_binom_glm_recovers_logistic_effect():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    n = np.full(300, 100.0)
    k = rng.binomial(100, 1 / (1 + np.exp(-(0.3 + 1.5 * x))))
    r = an.binom_glm(x[:, None], k, n, ["x"], groups=np.arange(300) // 3, shuffle_groups=True)
    assert r["coef"]["x"] == pytest.approx(1.5, abs=0.15)
    assert r["r2"] > 0.5 and r["cv_r2"] > 0.5
    assert np.all((r["pred"] > 0) & (r["pred"] < 1))


@pytest.fixture(scope="module")
def grid():
    cfg = ex.Config(quick=True)
    cal = ex.calibrate(P, cfg)
    return cfg, ex.run_2d(P, cfg, cal["theta"])


def test_run_2d_shapes_and_retention(grid):
    cfg, G = grid
    n = cfg.n_2d
    assert G["keep"].shape == (n, n) and G["keep"].all()          # the chosen region is bistable WTA
    assert G["acc"].shape == (n, n, len(cfg.sigmas_2d))
    assert np.all(np.isfinite(G["g_lam_u"])) and np.all(G["g_lam_u"] > 0)
    assert np.all(G["g_lam_sp"] < 0)                              # resting state stable everywhere
    assert np.all(G["acc"][G["keep"]] > 0.5)


def test_analyze_2d_structure(grid):
    _, G = grid
    H = ex.analyze_2d(G)
    for m in ("sigma", "J12", "J11", "lam", "geo", "J11J12", "quad", "geo_quad", "lam_J", "d"):
        for t in ("rt", "acc"):
            r = H["models"][m][t]
            assert {"r2", "cv_r2", "rmse", "cv_rmse", "aic", "daic"} <= set(r)
            assert r["daic"] >= -1e-9
    # richer circuit model never fits worse in-sample than its nested sub-model
    assert H["models"]["J11J12"]["rt"]["r2"] >= H["models"]["J11"]["rt"]["r2"] - 1e-9
    assert 0 <= H["geometry"]["median_angle"] <= 90
