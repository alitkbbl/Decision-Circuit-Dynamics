# Decision-Circuit-Dynamics

**Phase-Plane Geometry Versus Structural Bifurcations: Dissociating Speed–Accuracy Control and Behavioral Failure Modes in a Reduced Cortical Decision Circuit**

Code and results for a computational study of the two-population NMDA-dominated decision circuit of
[Wong & Wang (2006)](https://doi.org/10.1523/JNEUROSCI.3733-05.2006). The study asks how deterministic
phase-plane geometry (saddle eigenvalue, separatrix distance) and structural bifurcations relate to
stochastic decision speed, accuracy and behavioral failure modes (impulsive, indecisive, co-activation). 🔬

---

## 📄 Manuscript

- 📘 [**Report (PDF)**](report/report.pdf) 

The report is hand-maintained. The files `report/generated_numbers.tex` and `report/table_*.tex` are
**static, frozen copies** of the numbers and tables that the report includes; no script writes into
`report/` any more. If you change the analysis, update those files (or inline the numbers) by hand.

---

## 🚀 Quick start

```bash
git clone https://github.com/alitkbbl/Decision-Circuit-Dynamics.git
cd Decision-Circuit-Dynamics
pip install -r requirements.txt   
python run_all.py                
python run_all.py --quick        
python run_all.py --plots-only   
pytest -q                        
```
🎲 All randomness uses fixed seeds (trial `i` is seeded with `seed + i`), so every run reproduces the same results.

---

## 🗂️ Layout

```
run_all.py          one command: simulations -> CSV/JSON tables -> figures
wwdm/
  model.py          parameters, deterministic model, Numba stochastic simulator
  analysis.py       fixed points, separatrix, bifurcations, geometry, regression/GLM
  experiments.py    baseline, sweeps, regimes, joint (J11, J12) grid, premature-response and trapping analyses
  plotting.py       all figures
  export.py         CSV tables and summary.json
tests/              unit tests
figures/            fig1..fig10 (PNG + PDF; git-ignored, regenerate with run_all.py)
results/            CSV tables, summary.json, results.pkl (cache; git-ignored)
report/             report.tex, report.pdf and the static number/table files it includes
```

---

## 📊 What is computed

- **Phase plane** (`fig1`): nullclines, fixed points, separatrix, basins, noisy trajectories
- **Baseline** (`fig2`): psychometric/chronometric curves, error vs. correct RT, RT distributions
- **Bifurcations** (`fig3`): fixed points vs. cross-inhibition J12, (J12, coherence) map
- **Regime map** (`fig4`): accuracy, RT, premature responses, losing-population rate, regimes over (J12, σ)
- **H1, one-parameter sweep** (`fig5`): λᵤ, d, σ as predictors of RT and error rate along J12
- **Landscape** (`fig6`): empirical −log P and its noise dependence
- **Sensitivity** (`fig7`): read-out threshold fraction and integration step
- **J11 sweep** (`fig8`): (J11, σ) sweep, where the *indecisive* regime appears
- **H1, joint grid** (`fig9`): 2D (J11, J12) bistable grid, collapse test and grouped cross-validation
- **H2, tripartite dissociation** (`fig10`): premature escape, circuit-to-readout mismatch, co-activation

---

## 💡 Main findings (details and numbers in the report and in `results/summary.json`)

* ✅ The baseline reproduces the classic signatures: monotone psychometric and chronometric curves, errors slower than correct trials.
* ❌ **H1: a single saddle eigenvalue does not predict decision speed across circuits.** Along J12 alone, λᵤ and σ explain much of log RT, but a raw-parameter model does equally well. On the joint (J11, J12) grid, RT collapses onto J11 rather than onto λᵤ, and {J11, J12, σ} generalizes better than {λᵤ, d, σ} in grouped cross-validation. The separatrix distance d is nearly a function of J11.
* ❌ **H2: behavioral regime boundaries do not coincide with bifurcations.**
  * *Impulsivity*: premature responses occur while the resting state stays stable in every simulated circuit; their probability follows an empirical Kramers-like scaling with a geometric barrier proxy.
  * *Indecisiveness*: appears before the saddle-node of the choice attractors, because the attractor rate falls below the fixed read-out threshold (circuit-to-readout mismatch).
  * *Co-activation*: loss of winner-take-all competition at weak J12 leaves accuracy and first-crossing RT smooth; the losing-population rate reveals it.
* The empirical double-well landscape collapses at the noise level where accuracy breaks down.

---

## 👤 Author

Ali Taghizadeh — Department of Mathematics and Computer Science, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran
