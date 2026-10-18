# 🌀 Decision-Circuit-Dynamics

**Dynamical Origins of Speed–Accuracy Trade-offs and Dysfunctional Decision Regimes in a Reduced Cortical Decision Circuit**

A purely computational bachelor's project on the two-variable decision circuit of
[Wong & Wang (2006)](https://doi.org/10.1523/JNEUROSCI.3733-05.2006): phase-plane geometry,
bifurcations, and noise-resolved behavioral regimes. 🔬

Repository: <https://github.com/alitkbbl/Decision-Circuit-Dynamics>

## 📄 Documents

- 📝 [**Project Proposal**](report/Proposal.pdf): the original proposal with the research question, hypotheses H1/H2, methods and the 12-week work plan.
- 📘 [**Final Report**](report/report.pdf): the full scientific report with model, methods, results, discussion and limitations ([LaTeX source](report/report.tex)).

> The report is built from the proposal: it tests the same hypotheses and lists every deviation from the plan in its *Deviations from the proposal* paragraph.

## 🚀 Quick start

```bash
git clone https://github.com/alitkbbl/Decision-Circuit-Dynamics.git
cd Decision-Circuit-Dynamics
pip install -r requirements.txt
python run_all.py            # everything: ~4-5 min on one CPU core
python run_all.py --quick    # tiny smoke test (WARNING: overwrites figures with noisy ones)
python run_all.py --plots-only   # redraw figures/tables from results/results.pkl
pytest -q                    # unit tests ✅
cd report && pdflatex report.tex && pdflatex report.tex   # rebuild the report (numbers are auto-generated)
```
🎲 All randomness uses fixed seeds (trial `i` is seeded with `seed + i`), so every run reproduces the same numbers.

## 🗂️ Layout

```
run_all.py          one command: simulations -> tables -> figures
wwdm/
  model.py          parameters, deterministic model, Numba stochastic simulator
  analysis.py       fixed points, separatrix, bifurcations, geometry, regression/GLM
  experiments.py    baseline, sweeps, regime classification, H1/H2 tests, landscape
  plotting.py       all figures
  export.py         CSV/JSON tables and LaTeX macros for the report
tests/test_model.py unit tests
figures/            fig1..fig8 (PNG + PDF)
results/            CSV tables, summary.json, results.pkl (cache)
report/             Proposal.pdf, report.tex, report.pdf
```

## 📊 What is computed

- 🗺️ **Phase plane** (`fig1`): nullclines, fixed points, separatrix, basins, noisy trajectories
- ⏱️ **Baseline** (`fig2`): psychometric/chronometric curves, error vs. correct RT, RT distributions
- 🔀 **Bifurcations** (`fig3`): fixed points vs. cross-inhibition J12, (J12, coherence) map
- 🌡️ **Regime map** (`fig4`): accuracy, RT, premature responses, competition, regimes over (J12, σ)
- 📐 **H1** (`fig5`): λᵤ, d, σ as predictors of RT and error rate
- ⛰️ **Landscape** (`fig6`): empirical −log P, barrier vs. noise
- 🔍 **Sensitivity** (`fig7`): threshold fraction, time step, boundary robustness
- ➕ **Supplement** (`fig8`): (J11, σ) sweep, where the *indecisive* regime appears

## 💡 Main findings (see the report for details)

* ✅ Baseline reproduces the classic signatures: monotone psychometric/chronometric curves, errors slower than correct trials.
* 🟡 **H1 partly supported**: λᵤ and σ explain 78 % of the variance of log RT; the distance d to the separatrix is nearly constant and uninformative; error rate depends on σ only.
* ❌ **H2 not supported in its strong form**: behavioral boundaries do not sit on bifurcation values. The noise tolerance before premature responses is predicted (R² = 0.98) by a deterministic barrier proxy; the indecisive regime requires weaker recurrent excitation J11 and follows the read-out criterion rather than the existence bifurcation.
* 🌊 The empirical double-well landscape collapses at σ ≈ 0.06 nA, where accuracy at 12.8 % coherence breaks down.

## ⚠️ Notes and caveats

* Model parameters are those given in the project proposal and were **not** re-verified against the original paper's table.
* Regime labels are operational and depend on preset cut-offs and on the decision threshold (see the report, Sec. 4.7).
* The reference list in the report was written from memory/the proposal; please verify each entry before submission.

## 👤 Author

Ali Taghizadeh — Department of Computer Science, Amirkabir University of Technology (Tehran Polytechnic)
