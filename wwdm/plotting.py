"""Publication-quality figures (all figures of the report are produced here)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import analysis as an
from .experiments import _crossing
from .model import rate_from_S, vector_field

# Okabe-Ito colour-blind-safe palette
BLUE, VERM, GREEN, ORANGE, SKY, PINK, YELLOW, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#7F7F7F")
C_POP1, C_POP2 = BLUE, VERM
REGIME_COL = [GREEN, VERM, BLUE, "#BBBBBB"]
REGIME_NAME = ["Normal", "Impulsive", "Indecisive", "Other"]

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_FORMATS = ("png", "pdf")


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
        "axes.grid": False, "lines.linewidth": 1.8, "lines.markersize": 5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "figure.constrained_layout.use": True, "pdf.fonttype": 42,
    })


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in FIG_FORMATS:
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)


def panel_label(ax, s, dx=-36, dy=5):
    """Bold panel letter placed (in points) above-left of the axes, clear of the title."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontsize=13, fontweight="bold", va="bottom", ha="left", annotation_clip=False)


def edges(x):
    x = np.asarray(x, float)
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])


def plot_fps(ax, fps, ms=7):
    for f in fps:
        x, y = f["S"]
        if f["kind"] == an.STABLE:
            ax.plot(x, y, "o", color="k", ms=ms, zorder=6)
        elif f["kind"] == an.SADDLE:
            ax.plot(x, y, "D", mfc="white", mec="k", mew=1.4, ms=ms, zorder=6)
        else:
            ax.plot(x, y, "o", mfc="white", mec="k", mew=1.4, ms=ms, zorder=6)


def nullclines(ax, p, coh, stim, lim=0.85, n=500):
    g = np.linspace(0.0, lim, n)
    A, B = np.meshgrid(g, g)
    f1, f2 = vector_field(A, B, p, coh, stim)
    ax.contour(A, B, f1, levels=[0], colors=[C_POP1], linewidths=1.7, zorder=4)
    ax.contour(A, B, f2, levels=[0], colors=[C_POP2], linewidths=1.7, zorder=4)


def streams(ax, p, coh, stim, lim=0.85):
    g = np.linspace(0.0, lim, 40)
    A, B = np.meshgrid(g, g)
    f1, f2 = vector_field(A, B, p, coh, stim)
    ax.streamplot(g, g, f1, f2, color="#9A9A9A", density=1.0, linewidth=0.55, arrowsize=0.7, zorder=2)


def _phase_axes(ax, lim=0.85):
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$S_1$ (population 1)")
    ax.set_ylabel(r"$S_2$ (population 2)")


# --------------------------------------------------------------------------- #
# Fig 1 - phase portraits
# --------------------------------------------------------------------------- #

def fig_phase_portrait(R):
    from .model import simulate_traces
    p, cfg = R["params"], R["cfg"]
    theta = R["baseline"]["cal"]["theta"]
    lim = 0.85
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 5.0))
    cases = [(0.0, False, "Resting state (no stimulus)"),
             (0.0, True, "Stimulus on, 0 % coherence"),
             (12.8, True, "Stimulus on, 12.8 % coherence")]
    for ax, (coh, stim, title), lab in zip(axs, cases, "abc"):
        g1, g2, D = an.basin_grid(p, coh, n=130, stim=stim)
        if stim:
            ax.contourf(g1, g2, D, levels=[-2, 0, 2], colors=["#F6D5C5", "#CFE3F3"], zorder=0)
        else:
            ax.contourf(g1, g2, D, levels=[-2, -0.3, 0.3, 2], colors=["#F6D5C5", "#E3E3E3", "#CFE3F3"], zorder=0)
        streams(ax, p, coh, stim, lim)
        nullclines(ax, p, coh, stim, lim)
        fps = an.find_fixed_points(p, coh, stim)
        plot_fps(ax, fps)
        if stim:
            poly = an.separatrix(p, coh)
            if poly is not None:
                ax.plot(poly[:, 0], poly[:, 1], "k--", lw=1.8, zorder=5)
        s0 = an.spontaneous_state(p)
        ax.plot(*s0, "*", color=GREEN, ms=15, mec="k", mew=0.8, zorder=7)
        if lab == "c":
            t, tr, out = simulate_traces(p, coh, 40, cfg.seed + 31, theta, dt=cfg.dt, t_pre=cfg.t_pre,
                                         t_max=cfg.t_max)
            sel = t >= 0
            for k in range(tr.shape[0]):
                ch = out["choice"][k]
                col = C_POP1 if ch == 1 else C_POP2
                T = tr[k][sel]
                ax.plot(T[:, 0], T[:, 1], color=col, lw=0.9, alpha=0.75, zorder=3)
        _phase_axes(ax, lim)
        ax.set_title(title)
        panel_label(ax, lab)
    handles = [Line2D([], [], color=C_POP1, lw=2, label=r"$dS_1/dt=0$ nullcline"),
               Line2D([], [], color=C_POP2, lw=2, label=r"$dS_2/dt=0$ nullcline"),
               Line2D([], [], color="k", ls="--", lw=1.8, label="separatrix (stable manifold)"),
               Line2D([], [], marker="o", color="k", ls="", label="stable fixed point"),
               Line2D([], [], marker="D", mfc="white", mec="k", color="k", ls="", label="saddle"),
               Line2D([], [], marker="*", color=GREEN, mec="k", ls="", ms=12, label="pre-stimulus state"),
               Patch(fc="#CFE3F3", label="basin: population 1 wins"),
               Patch(fc="#F6D5C5", label="basin: population 2 wins"),
               Patch(fc="#E3E3E3", label="basin: resting state"),
               Line2D([], [], color=C_POP1, lw=1.2, label="noisy trials (12.8 %)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig1_phase_portrait")


# --------------------------------------------------------------------------- #
# Fig 2 - baseline behaviour
# --------------------------------------------------------------------------- #

def fig_behavior(R):
    p, B = R["params"], R["baseline"]
    km, coh = B["k_main"], B["coh"]
    S = B["summary"]
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.0))
    # (a) example traces
    ax = axs[0, 0]
    tr = R["traces"]
    t, T = tr["t"], tr["traces"]
    rates = rate_from_S(T, p)
    for k in range(10):
        ax.plot(t, rates[k, :, 0], color=C_POP1, lw=0.9, alpha=0.55)
        ax.plot(t, rates[k, :, 1], color=C_POP2, lw=0.9, alpha=0.55)
    ax.axhline(tr["theta"], color="k", ls="--", lw=1.2)
    ax.text(1.95, tr["theta"] + 0.7, r"decision threshold $\theta$", ha="right", va="bottom", fontsize=8.5)
    ax.axvline(0, color=GREY, lw=1.0)
    ax.text(0.03, 34, "stimulus on", color=GREY, fontsize=8.5)
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(0, 36)
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel(r"rate-equivalent $r_i=S_i/(\gamma\tau_S(1-S_i))$ (Hz)")
    ax.set_title("Single trials, 12.8 % coherence")
    ax.legend(handles=[Line2D([], [], color=C_POP1, label="population 1 (favoured)"),
                       Line2D([], [], color=C_POP2, label="population 2")], loc="upper left",
              bbox_to_anchor=(0.0, 0.88), frameon=False)
    panel_label(ax, "a")
    # (b) psychometric
    ax = axs[0, 1]
    xplot = np.where(coh > 0, coh, 1.6)
    acc = S["acc"][:, km]
    lo, hi = an.wilson(S["n_corr"][:, km], S["n_dec"][:, km])
    ax.errorbar(xplot, acc, yerr=[acc - lo, hi - acc], fmt="o", color=BLUE, capsize=3, zorder=4, label="simulation (95 % CI)")
    a_, b_ = B["weibull"]
    xx = np.geomspace(2.4, 60, 100)
    ax.plot(xx, an.weibull(xx, a_, b_), color="k", lw=1.5, label=rf"Weibull fit ($\alpha$={a_:.1f} %, $\beta$={b_:.2f})")
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([1.6, 3.2, 6.4, 12.8, 25.6, 51.2])
    ax.set_xticklabels(["0", "3.2", "6.4", "12.8", "25.6", "51.2"])
    ax.minorticks_off()
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("motion coherence (%)")
    ax.set_ylabel("proportion correct")
    ax.set_title("Psychometric function")
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "b")
    # (c) chronometric
    ax = axs[1, 0]
    for key, col, lab, mk in (("rt_corr", C_POP1, "correct", "o"), ("rt_err", C_POP2, "error", "s")):
        y = S[key][:, km]
        sem = np.array([np.nanstd(np.where(
            (B["raw"][c]["choice"] == (1 if key == "rt_corr" else 2)) & ~B["raw"][c]["prem"],
            B["raw"][c]["rt"], np.nan)) / np.sqrt(max(
                np.sum((B["raw"][c]["choice"] == (1 if key == "rt_corr" else 2)) & ~B["raw"][c]["prem"]), 1))
            for c in coh])
        ok = np.isfinite(y) & (S["n_err"][:, km] >= 5 if key == "rt_err" else True)
        ax.errorbar(xplot[ok], 1000 * y[ok], yerr=1000 * sem[ok], fmt=mk + "-", color=col, capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([1.6, 3.2, 6.4, 12.8, 25.6, 51.2])
    ax.set_xticklabels(["0", "3.2", "6.4", "12.8", "25.6", "51.2"])
    ax.minorticks_off()
    ax.set_xlabel("motion coherence (%)")
    ax.set_ylabel("mean reaction time (ms)")
    ax.set_title("Chronometric function (±SEM)")
    ax.legend(frameon=False)
    panel_label(ax, "c")
    # (d) RT distributions
    ax = axs[1, 1]
    c_show = 6.4
    raw = B["raw"][c_show]
    dec = (raw["choice"] > 0) & ~raw["prem"]
    bins = np.linspace(0, 2.0, 41)
    for chv, col, lab in ((1, C_POP1, "correct"), (2, C_POP2, "error")):
        v = raw["rt"][dec & (raw["choice"] == chv)]
        ax.hist(v, bins=bins, density=True, color=col, alpha=0.55, label=f"{lab} (n={len(v)})")
        if len(v):
            ax.axvline(np.mean(v), color=col, lw=1.6, ls="--")
    ax.set_xlabel("reaction time (s)")
    ax.set_ylabel("probability density")
    ax.set_title(f"RT distributions, {c_show} % coherence")
    ax.legend(frameon=False)
    panel_label(ax, "d")
    save(fig, "fig2_behavior")


# --------------------------------------------------------------------------- #
# Fig 3 - bifurcation analysis
# --------------------------------------------------------------------------- #

def _bif_panel(ax, sc, trans, title, j12_base, j11):
    st, sd, un = sc["kind"] == an.STABLE, sc["kind"] == an.SADDLE, sc["kind"] == an.UNSTABLE
    y = sc["S1"] - sc["S2"]
    ax.plot(sc["J12"][un], y[un], ".", color=GREY, ms=2.5)
    ax.plot(sc["J12"][sd], y[sd], ".", color=VERM, ms=2.5)
    ax.plot(sc["J12"][st], y[st], ".", color=BLUE, ms=2.8)
    for v, _, _ in trans:
        ax.axvline(v, color="k", lw=0.8, ls=":")
    ax.axvline(j12_base, color=GREEN, lw=1.4)
    ax.set_xlim(0, 0.14)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xlabel(r"cross-inhibition $J_{12}$ (nA)")
    ax.set_ylabel(r"$S_1-S_2$")
    ax.set_title(title)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    sec.set_xticks([0.0, 0.0497, 0.1])
    sec.set_xticklabels(["", rf"$J_{{11}}/J_{{12}}$={j11 / 0.0497:.1f}", ""], fontsize=8)
    sec.tick_params(length=0)
    sec.spines["top"].set_visible(False)


def fig_bifurcation(R):
    p, bif = R["params"], R["bif"]
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.4))
    _bif_panel(axs[0, 0], bif["scan_off"], bif["trans_off"], "No stimulus", p.J12, p.J11)
    _bif_panel(axs[0, 1], bif["scan"][0.0], bif["trans"][0.0], "Stimulus on, 0 % coherence", p.J12, p.J11)
    _bif_panel(axs[1, 0], bif["scan"][12.8], bif["trans"][12.8], "Stimulus on, 12.8 % coherence", p.J12, p.J11)
    for ax, l in zip(axs.ravel()[:3], "abc"):
        panel_label(ax, l)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="stable"),
                              Line2D([], [], marker="o", ls="", color=VERM, label="saddle"),
                              Line2D([], [], marker="o", ls="", color=GREY, label="unstable node/focus"),
                              Line2D([], [], color="k", ls=":", label="bifurcation"),
                              Line2D([], [], color=GREEN, label="baseline $J_{12}$")],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.45, -0.05))
    # (d) two-parameter map
    ax = axs[1, 1]
    cm = bif["count_map"]
    levels = np.arange(0.5, cm.max() + 1.5)
    cols = ["#F1F1F1", "#CFE3F3", "#9ECAE1", "#4292C6", "#08519C"][: int(cm.max())]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(np.arange(0.5, cm.max() + 1.0, 1.0), cmap.N)
    pc = ax.pcolormesh(edges(bif["J12_axis"]), edges(bif["coh_axis"]), cm, cmap=cmap, norm=norm)
    ax.axvline(p.J12, color=GREEN, lw=1.4)
    cb = fig.colorbar(pc, ax=ax, ticks=range(1, int(cm.max()) + 1), shrink=0.9)
    cb.set_label("number of stable fixed points")
    ax.set_xlabel(r"cross-inhibition $J_{12}$ (nA)")
    ax.set_ylabel("coherence (%)")
    ax.set_title("Two-parameter map (stimulus on)")
    panel_label(ax, "d")
    save(fig, "fig3_bifurcation")


# --------------------------------------------------------------------------- #
# Fig 4 - regime map
# --------------------------------------------------------------------------- #

def _heat(ax, fig, G, Z, cmap, label, vmin=None, vmax=None, base=None, title=""):
    pc = ax.pcolormesh(edges(G["xs"]), edges(G["sigmas"]), Z.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    cb = fig.colorbar(pc, ax=ax, shrink=0.92)
    cb.set_label(label)
    if base is not None:
        ax.plot(*base, "*", color="white", mec="k", mew=0.9, ms=13, zorder=6)
    ax.set_xlabel(rf"cross-inhibition $J_{{12}}$ (nA)" if G["xname"] == "J12" else r"self-excitation $J_{11}$ (nA)")
    ax.set_ylabel(r"noise amplitude $\sigma$ (nA)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(True)


def fig_regime_map(R):
    G, cfg, p, h2 = R["sweep"], R["cfg"], R["params"], R["h2"]
    k, kc = R["k_th"], 1
    base = (p.J12, p.sigma)
    fig, axs = plt.subplots(2, 3, figsize=(16.0, 9.0))
    _heat(axs[0, 0], fig, G, G["acc"][:, :, kc, k], "viridis", "proportion correct", 0.5, 1.0, base, "Accuracy, 12.8 % coherence")
    _heat(axs[0, 1], fig, G, 1000 * G["rt_mean"][:, :, kc, k], "magma_r", "mean RT (ms)", None, None, base, "Mean RT, 12.8 % coherence")
    _heat(axs[0, 2], fig, G, G["p_prem"][:, :, kc, k], "Reds", "fraction of trials", 0,
          max(0.3, np.nanmax(G["p_prem"][:, :, kc, k])), base, "Premature responses (pre-stimulus)")
    _heat(axs[1, 0], fig, G, G["r_other"][:, :, kc, k], "cividis_r", "rate of losing population (Hz)", 0, None, base,
          "Competition: loser rate at decision")
    bif_sn = R["bif"]["trans"][12.8][0][0]
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.axvline(bif_sn, color="white", lw=1.4, ls="--")
    for ax, l in zip(axs.ravel(), "abcdef"):
        panel_label(ax, l)
    # (e) classified regimes with bifurcation curves and boundaries
    ax = axs[1, 1]
    lab = G["regimes"][cfg.f_main]
    ax.pcolormesh(edges(G["xs"]), edges(G["sigmas"]), lab.T, cmap=ListedColormap(REGIME_COL), vmin=-0.5, vmax=3.5, shading="flat")
    xl = (edges(G["xs"])[0], edges(G["xs"])[-1])
    yl = (edges(G["sigmas"])[0], edges(G["sigmas"])[-1])
    ax.fill_betweenx(yl, xl[0], bif_sn, facecolor="none", edgecolor="k", hatch="//", lw=0, alpha=0.35)
    for c, ls in ((0.0, "-"), (12.8, "--")):
        for v, _, _ in R["bif"]["trans"][c]:
            if xl[0] <= v <= xl[1]:
                ax.axvline(v, color="k", ls=ls, lw=1.3)
    ax.plot(h2["xs"], h2["sigma_acc"], "o-", color="white", mec="k", ms=4, lw=1.4)
    ax.plot(h2["xs"], h2["sigma_prem"], "s-", color="white", mec="k", ms=4, lw=1.4)
    ax.plot(*base, "*", color="white", mec="k", mew=0.9, ms=13, zorder=6)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xlabel(r"cross-inhibition $J_{12}$ (nA)")
    ax.set_ylabel(r"noise amplitude $\sigma$ (nA)")
    ax.set_title("Operational regime map")
    ax.spines[["top", "right"]].set_visible(True)
    hs = [Patch(fc=REGIME_COL[i], label=REGIME_NAME[i]) for i in range(4) if np.any(lab == i)]
    hs += [Patch(fc="none", ec="k", hatch="//", label="no choice attractors (deterministic)"),
           Line2D([], [], color="k", ls="-", label="bifurcation, 0 %"),
           Line2D([], [], color="k", ls="--", label="bifurcation, 12.8 %"),
           Line2D([], [], marker="o", color="white", mec="k", label=r"$\sigma$: accuracy $=0.65$"),
           Line2D([], [], marker="s", color="white", mec="k", label=r"$\sigma$: premature $=10\,\%$")]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=8.5)
    # (f) premature-boundary vs deterministic barrier proxy
    ax = axs[1, 2]
    geo = R["geo"]
    m = np.isfinite(h2["sigma_prem"])
    ax.scatter(geo["d_barrier"][m], h2["sigma_prem"][m], c=h2["xs"][m], cmap="viridis", s=48, edgecolor="k", zorder=3)
    if "prem_vs_barrier" in h2:
        v = h2["prem_vs_barrier"]
        xx = np.linspace(geo["d_barrier"][m].min(), geo["d_barrier"][m].max(), 20)
        ax.plot(xx, v["intercept"] + v["slope"] * xx, "k--", lw=1.3, label=rf"linear fit, $R^2$={v['r2']:.2f}")
        ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel("resting state $\\to$ nearest saddle, no stimulus (S units)")
    ax.set_ylabel(r"$\sigma$ at 10 % premature responses")
    ax.set_title("Noise boundary vs. deterministic barrier")
    save(fig, "fig4_regime_map")


# --------------------------------------------------------------------------- #
# Fig 5 - geometry and H1
# --------------------------------------------------------------------------- #

def fig_geometry_h1(R):
    G, geo, h1, p = R["sweep"], R["geo"], R["h1"], R["params"]
    xs = geo["xs"]
    fig = plt.figure(figsize=(14.2, 8.6))
    gs = fig.add_gridspec(2, 3)
    # (a) lambda_u and d
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xs, geo["lam_u"], "o-", color=VERM, label=r"$\lambda_u$ (saddle)")
    ax.set_xlabel(r"cross-inhibition $J_{12}$ (nA)")
    ax.set_ylabel(r"unstable eigenvalue $\lambda_u$ (s$^{-1}$)", color=VERM)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(xs, geo["d"], "s-", color=BLUE)
    ax2.set_ylabel(r"signed distance $d$ to separatrix", color=BLUE)
    ax.set_title("Deterministic geometry, 12.8 %")
    ax.axvline(p.J12, color=GREEN, lw=1.2)
    panel_label(ax, "a")
    # (b) separatrices
    ax = fig.add_subplot(gs[0, 1])
    sel = np.linspace(0, len(xs) - 1, 5).astype(int)
    cmap = plt.get_cmap("viridis")
    for n_, i in enumerate(sel):
        poly = geo["polys"][i]
        if poly is None:
            continue
        col = cmap(n_ / (len(sel) - 1))
        ax.plot(poly[:, 0], poly[:, 1], color=col, lw=1.7, label=rf"$J_{{12}}$={xs[i]:.3f}")
        pp = p.with_(J12=float(xs[i]))
        ax.plot(*an.spontaneous_state(pp), "*", color=col, ms=11, mec="k", mew=0.6)
        ax.plot(*geo["saddles"][i], "D", mfc="white", mec=col, mew=1.5, ms=6)
    ax.plot([0, 0.85], [0, 0.85], color=GREY, lw=0.7, ls=":")
    _phase_axes(ax, 0.85)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Separatrix, saddle ($\\diamond$), start ($\\star$)")
    panel_label(ax, "b")
    # (c) log RT: predicted vs observed
    m = h1["mask"].ravel()
    sg = np.repeat(G["sigmas"][None, :], len(xs), 0).ravel()[m]
    ax = fig.add_subplot(gs[0, 2])
    y, pred = h1["log_rt"]["y"].ravel()[m], h1["log_rt"]["geo"]["pred"]
    sc = ax.scatter(pred, y, c=sg, cmap="plasma", s=26, edgecolor="none")
    lo, hi = min(pred.min(), y.min()), max(pred.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(r"predicted $\log$ mean RT")
    ax.set_ylabel(r"observed $\log$ mean RT")
    ax.set_title(rf"H1, RT: $R^2$={h1['log_rt']['geo']['r2']:.2f} (CV {h1['log_rt']['geo']['cv_r2']:.2f})")
    fig.colorbar(sc, ax=ax, shrink=0.85).set_label(r"$\sigma$ (nA)")
    panel_label(ax, "c")
    # (d) error rate: predicted vs observed probability
    ax = fig.add_subplot(gs[1, 0])
    y, pred = h1["logit_err"]["y"].ravel()[m], h1["logit_err"]["geo"]["pred"]
    ax.scatter(pred, y, c=sg, cmap="plasma", s=26, edgecolor="none")
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=1)
    ax.set_xlabel("predicted error rate")
    ax.set_ylabel("observed error rate")
    ax.set_title(rf"H1, errors: pseudo-$R^2$={h1['logit_err']['geo']['r2']:.2f} (CV {h1['logit_err']['geo']['cv_r2']:.2f})")
    panel_label(ax, "d")
    # (e) model comparison
    ax = fig.add_subplot(gs[1, 1])
    models = [("sigma", r"$\sigma$"), ("param", r"$J_{12},\sigma$"), ("geo", r"$\lambda_u,d,\sigma$"),
              ("geo_plus", r"$\lambda_u,d,d_{sad},\sigma$")]
    w = 0.18
    xpos = np.arange(2)
    for i, (m, nm) in enumerate(models):
        r2 = [h1["log_rt"][m]["r2"], h1["logit_err"][m]["r2"]]
        cv = [h1["log_rt"][m]["cv_r2"], h1["logit_err"][m]["cv_r2"]]
        col = [GREY, SKY, BLUE, "#08306B"][i]
        off = (i - 1.5) * w * 1.1
        ax.bar(xpos + off, r2, w, color=col, label=nm)
        ax.plot(xpos + off, cv, "k_", ms=11, mew=2)
    ax.set_xticks(xpos)
    ax.set_ylabel(r"$R^2$ (bars); leave-$J_{12}$-out CV (ticks)")
    ax.set_title("Predictors compared")
    ax.set_xticklabels([r"$\log$ RT (OLS $R^2$)", "error rate (McFadden)"])
    ax.set_ylim(0, 1.28)
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=8)
    panel_label(ax, "e")
    # (f) standardised coefficients
    ax = fig.add_subplot(gs[1, 2])
    names = ["log(lambda_u)", "d", "sigma"]
    pretty = [r"$\log\lambda_u$", r"$d$", r"$\sigma$"]
    for i, (key, col, nm) in enumerate((("log_rt", BLUE, "log RT (OLS)"), ("logit_err", VERM, "error rate (log-odds)"))):
        vals = [h1[key]["geo"]["std_coef"][n] for n in names]
        ax.bar(np.arange(3) + (i - 0.5) * 0.36, vals, 0.34, color=col, label=nm)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(pretty)
    ax.set_ylabel("standardised coefficient")
    ax.set_title("Geometric model: effect sizes")
    ax.legend(frameon=False)
    panel_label(ax, "f")
    save(fig, "fig5_geometry_h1")


# --------------------------------------------------------------------------- #
# Fig 6 - empirical landscape
# --------------------------------------------------------------------------- #

def _U2(X, bins=90, rng=((0, 0.85), (0, 0.85))):
    H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=bins, range=rng, density=True)
    with np.errstate(divide="ignore"):
        U = -np.log(H.T)
    U -= np.nanmin(U[np.isfinite(U)])
    return xe, ye, U


def _U1(X, bins=70, rng=(-0.8, 0.8)):
    h, e = np.histogram(X[:, 0] - X[:, 1], bins=bins, range=rng, density=True)
    with np.errstate(divide="ignore"):
        u = -np.log(h)
    u -= np.nanmin(u[np.isfinite(u)])
    return 0.5 * (e[1:] + e[:-1]), u


def fig_landscape(R):
    L, p = R["landscape"], R["params"]
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.9))
    vmax = 9
    ax = axs[0]
    xe, ye, U = _U2(L["ensemble"])
    pc = ax.pcolormesh(xe, ye, np.clip(U, 0, vmax), cmap="viridis_r", vmin=0, vmax=vmax)
    fig.colorbar(pc, ax=ax, shrink=0.85).set_label(r"$-\log P$")
    plot_fps(ax, an.find_fixed_points(p, 0.0, True), ms=6)
    ax.set_title(r"Decision trials, $\sigma$=0.02 nA")
    _phase_axes(ax)
    panel_label(ax, "a")
    s_show = 0.06
    ax = axs[1]
    xe, ye, U = _U2(L["long"][s_show])
    pc = ax.pcolormesh(xe, ye, np.clip(U, 0, vmax), cmap="viridis_r", vmin=0, vmax=vmax)
    fig.colorbar(pc, ax=ax, shrink=0.85).set_label(r"$-\log P$")
    plot_fps(ax, an.find_fixed_points(p, 0.0, True), ms=6)
    ax.set_title(rf"Stationary long run, $\sigma$={s_show} nA")
    _phase_axes(ax)
    panel_label(ax, "b")
    ax = axs[2]
    for sg, col in zip(sorted(L["long"]), (BLUE, GREEN, ORANGE, VERM)):
        x, u = _U1(L["long"][sg])
        ax.plot(x, np.clip(u, 0, 14), color=col, label=rf"$\sigma$={sg} ({L['n_switch'][sg]} transitions)")
    x, u = _U1(L["ensemble"])
    ax.plot(x, np.clip(u, 0, 14), color="k", ls="--", lw=1.4, label=r"decision trials, $\sigma$=0.02 (non-stationary)")
    ax.set_xlabel(r"$S_1-S_2$")
    ax.set_ylabel(r"$-\log P$  (1D cut)")
    ax.set_ylim(0, 9)
    ax.set_title("Barrier between choice states")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    panel_label(ax, "c")
    save(fig, "fig6_landscape")


# --------------------------------------------------------------------------- #
# Fig 7 - sensitivity
# --------------------------------------------------------------------------- #

def fig_sensitivity(R):
    B, D, G, cfg = R["baseline"], R["dt_sens"], R["sweep"], R["cfg"]
    coh = B["coh"]
    xplot = np.where(coh > 0, coh, 1.6)
    fig, axs = plt.subplots(2, 3, figsize=(14.4, 8.0))
    cols = plt.get_cmap("cividis")(np.linspace(0.05, 0.9, len(B["f"])))
    ticks = ([1.6, 3.2, 6.4, 12.8, 25.6, 51.2], ["0", "3.2", "6.4", "12.8", "25.6", "51.2"])
    for k, f in enumerate(B["f"]):
        axs[0, 0].plot(xplot, B["summary"]["acc"][:, k], "o-", color=cols[k], ms=3.5, label=f"{f:.2f}")
        axs[0, 1].plot(xplot, 1000 * B["summary"]["rt_mean"][:, k], "o-", color=cols[k], ms=3.5)
        axs[0, 2].plot(xplot, B["summary"]["p_nodec"][:, k] + B["summary"]["p_prem"][:, k], "o-", color=cols[k], ms=3.5)
    axs[0, 0].legend(title=r"threshold $f\,r^*$", frameon=False, fontsize=8, ncol=2)
    for ax, yl, t, l in ((axs[0, 0], "proportion correct", "Threshold: accuracy", "a"),
                         (axs[0, 1], "mean RT (ms)", "Threshold: RT", "b"),
                         (axs[0, 2], "fraction timed out or premature", "Threshold: non-valid trials", "c")):
        ax.set_xscale("log")
        ax.set_xticks(ticks[0])
        ax.set_xticklabels(ticks[1])
        ax.minorticks_off()
        ax.set_xlabel("coherence (%)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        panel_label(ax, l)
    dcol = [GREY, SKY, BLUE, VERM]
    for i, dt in enumerate(D["dts"]):
        axs[1, 0].plot(xplot, D["acc"][i][:, 0], "o-", color=dcol[i], ms=3.5, label=f"{dt * 1000:g} ms")
        axs[1, 1].plot(xplot, 1000 * D["rt"][i], "o-", color=dcol[i], ms=3.5)
    axs[1, 0].legend(title="time step", frameon=False, fontsize=8)
    for ax, yl, t, l in ((axs[1, 0], "proportion correct", r"Integration step $dt$: accuracy", "d"),
                         (axs[1, 1], "mean RT (ms)", r"Integration step $dt$: RT", "e")):
        ax.set_xscale("log")
        ax.set_xticks(ticks[0])
        ax.set_xticklabels(ticks[1])
        ax.minorticks_off()
        ax.set_xlabel("coherence (%)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        panel_label(ax, l)
    ax = axs[1, 2]
    kc = 1
    for k, (f, col) in enumerate(zip(cfg.f_grid, (SKY, BLUE, "#08306B"))):
        xs, sg = G["xs"], G["sigmas"]
        y = [_crossing(sg, G["acc"][i, :, kc, k], cfg.crit_acc, "down") for i in range(len(xs))]
        ax.plot(xs, y, "o-", color=col, label=f"f={f:.2f}")
    ax.set_xlabel(r"cross-inhibition $J_{12}$ (nA)")
    ax.set_ylabel(r"$\sigma$ at which accuracy falls to 0.65")
    ax.set_title("Regime boundary vs. threshold")
    ax.legend(frameon=False)
    panel_label(ax, "f")
    save(fig, "fig7_sensitivity")


# --------------------------------------------------------------------------- #
# Fig 8 - supplementary J11 sweep
# --------------------------------------------------------------------------- #

def fig_j11_sweep(R):
    G, cfg, p, h = R["sweep11"], R["cfg"], R["params"], R["h2_j11"]
    k, kc = R["k_th"], 1
    base = (p.J11, p.sigma)
    fig, axs = plt.subplots(1, 3, figsize=(15.4, 4.9))
    _heat(axs[0], fig, G, G["p_nodec"][:, :, kc, k], "Blues", "no-decision rate", 0, 1, base, "Timeouts, 12.8 % coherence")
    _heat(axs[1], fig, G, 1000 * G["rt_mean"][:, :, kc, k], "magma_r", "mean RT (ms)", None, None, base, "Mean RT, 12.8 % coherence")
    ax = axs[2]
    lab = G["regimes"][cfg.f_main]
    ax.pcolormesh(edges(G["xs"]), edges(G["sigmas"]), lab.T, cmap=ListedColormap(REGIME_COL), vmin=-0.5, vmax=3.5)
    for v, a, b in h["bif_j11"]:
        if G["xs"].min() <= v <= G["xs"].max():
            ax.axvline(v, color="k", lw=1.3)
    if np.isfinite(h["j11_theta"]):
        ax.axvline(h["j11_theta"], color="k", lw=1.3, ls="--")
    ax.plot(h["j11_boundary"], h["sigmas"], "o-", color="white", mec="k", ms=4, lw=1.4)
    ax.plot(*base, "*", color="white", mec="k", mew=0.9, ms=13, zorder=6)
    ax.set_xlim(edges(G["xs"])[0], edges(G["xs"])[-1])
    ax.set_ylim(edges(G["sigmas"])[0], edges(G["sigmas"])[-1])
    ax.set_xlabel(r"self-excitation $J_{11}$ (nA)")
    ax.set_ylabel(r"noise amplitude $\sigma$ (nA)")
    ax.set_title("Operational regime map")
    hs = [Patch(fc=REGIME_COL[i], label=REGIME_NAME[i]) for i in range(4) if np.any(lab == i)]
    hs += [Line2D([], [], color="k", label="choice attractors appear"),
           Line2D([], [], color="k", ls="--", label=r"attractor rate $=\theta$"),
           Line2D([], [], marker="o", color="white", mec="k", label="observed 20 % timeout")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(True)
    for a, l in zip(axs, "abc"):
        panel_label(a, l)
    save(fig, "fig8_j11_sweep")


def make_all(R):
    set_style()
    for fn in (fig_phase_portrait, fig_behavior, fig_bifurcation, fig_regime_map, fig_geometry_h1,
               fig_landscape, fig_sensitivity, fig_j11_sweep):
        fn(R)
