# Decision-Circuit-Dynamics

**Dynamical Origins of Speed–Accuracy Trade-offs and Dysfunctional Decision Regimes in a Reduced Cortical Decision Circuit**

A computational bachelor's project studying the two-variable decision circuit of [Wong & Wang (2006)](https://doi.org/10.1523/JNEUROSCI.3733-05.2006), focusing on phase-plane geometry, bifurcations, and noise-resolved behavioral regimes.

**Repository:** [alitkbbl/Decision-Circuit-Dynamics](https://github.com/alitkbbl/Decision-Circuit-Dynamics)

## Documents

The following documents describe the project from its initial research plan to the final results. The report and its LaTeX source are available below.

> **Project Proposal:** [Read the proposal →](report/Proposal.pdf)  
> Research question, hypotheses, methodology, and planned experiments.
>
> **Final Report:** [Read the complete report →](report/report.pdf)  
> Detailed presentation of the model, experimental results, analysis, and limitations.


## Quick Start

```bash
git clone https://github.com/alitkbbl/Decision-Circuit-Dynamics.git
cd Decision-Circuit-Dynamics
pip install -r requirements.txt

python run_all.py
python run_all.py --quick
python run_all.py --plots-only
pytest -q

cd report
pdflatex report.tex
pdflatex report.tex
```

- `python run_all.py` — Run the full pipeline (~4–5 minutes on one CPU core).
- `python run_all.py --quick` — Run a minimal smoke test. **Overwrites figures with noisy results.**
- `python run_all.py --plots-only` — Regenerate figures and tables from `results/results.pkl`.
- `pytest -q` — Run unit tests.
- The report can be rebuilt using the included LaTeX source; numerical values are generated automatically.

All stochastic simulations use fixed seeds, with trial `i` seeded by `seed + i`, ensuring reproducible results.

## Repository Structure

```text
run_all.py          Main simulation and analysis pipeline
wwdm/
  model.py          Parameters, deterministic model, stochastic simulator
  analysis.py       Fixed points, separatrix, bifurcations, geometry, regression/GLM
  experiments.py    Baseline, parameter sweeps, regime classification, H1/H2 tests
  plotting.py       Figure generation
  export.py         CSV/JSON tables and LaTeX macros
tests/
  test_model.py     Unit tests
figures/            Figures 1–8 (PNG and PDF)
results/            CSV tables, summary.json, results.pkl
report/
  Proposal.pdf
  report.tex
  report.pdf
```

## Analyses

| Figure | Analysis |
|---|---|
| `fig1` | Phase plane: nullclines, fixed points, separatrix, basins, and noisy trajectories |
| `fig2` | Baseline: psychometric and chronometric curves, error versus correct reaction times, and RT distributions |
| `fig3` | Bifurcations: fixed points versus cross-inhibition \(J_{12}\), and the \((J_{12}, \text{coherence})\) map |
| `fig4` | Regime map: accuracy, reaction time, premature responses, competition, and behavioral regimes over \((J_{12}, \sigma)\) |
| `fig5` | H1: \(\lambda_u\), \(d\), and \(\sigma\) as predictors of reaction time and error rate |
| `fig6` | Energy landscape: empirical \(-\log P\) and barrier versus noise |
| `fig7` | Sensitivity analysis: threshold fraction, time step, and boundary robustness |
| `fig8` | Supplementary sweep over \((J_{11}, \sigma)\), examining the indecisive regime |

## Main Findings

- The baseline reproduces classic behavioral signatures: monotonic psychometric and chronometric curves, with errors taking longer than correct responses.
- **H1 is partially supported.** The unstable eigenvalue \(\lambda_u\) and noise level \(\sigma\) explain 78% of the variance in log reaction time. The distance \(d\) to the separatrix is nearly constant and uninformative, while error rate depends only on \(\sigma\).
- **H2 is not supported in its strong form.** Behavioral boundaries do not coincide with bifurcation values. A deterministic barrier proxy predicts the noise tolerance before premature responses (\(R^2 = 0.98\)). The indecisive regime requires weaker recurrent excitation \(J_{11}\) and follows the read-out criterion rather than the existence bifurcation.
- The empirical double-well landscape collapses at \(\sigma \approx 0.06\) nA, where accuracy at 12.8% coherence breaks down.

## Author

**Ali Taghizadeh**  
Department of Computer Science  
Amirkabir University of Technology (Tehran Polytechnic)
