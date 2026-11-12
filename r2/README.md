# R2 kernel: code and reproduction package

Reference implementation of the R2 stochastic discrete-event factory kernel and
the numerical conformance suite (experiments E1 to E6) reported in the paper.

## Files

| File | Purpose |
|---|---|
| `r2kernel.py` | Model-agnostic R2 engine (flow, detection, jump, tie rule, marks) plus two models: a single-server FIFO station (optional finite capacity) and a closed CONWIP line |
| `experiments.py` | Runs E1 to E5 with seeded, independent replications and writes `results.json` |
| `analyze.py` | Builds all figures (`../paper/fig_*.pdf`) and table bodies from `results.json` |
| `rerun_e3.py` | Regenerates E3 only |
| `results.json` | Raw results used in the paper |

## Reproduce

Requires Python 3.10+, NumPy, SciPy and Matplotlib (see `requirements.txt` in the repository root).

```bash
cd r2
python3 experiments.py      # about 6 minutes on 2 cores; writes results.json
python3 analyze.py          # reads results.json and writes the figures to ../paper/ (create that folder or edit OUT in analyze.py)
```

`rerun_e3.py` regenerates experiment E3 only, after the warm-up boundary fix described in the paper.

All random streams are seeded (`BASE_SEED = 20261111`), so the results are reproducible exactly.
