# Results inventory included in the consolidated Git bundle

This catalog only describes artifacts **present in the original supplied Git bundle**. Earlier experiments discussed outside the repository may exist, but their raw data are not part of this snapshot.

## Gurobi constrained-MIQP runs

Location: `src/waveform_optimization/solvers/gurobi/results/gurobi_exact/surya/`.

| N | Recorded status | Recorded best energy | Objective independently recalculated? |
| ---: | --- | ---: | --- |
| 5 | OPTIMAL | 2 | Yes |
| 7 | OPTIMAL | 3 | Yes |
| 11 | OPTIMAL | 5 | Yes |
| 15 | OPTIMAL | 15 | Yes |
| 20 | OPTIMAL | 26 | Yes |
| 25 | TIME_LIMIT | 68 | Yes |
| 30 | TIME_LIMIT | 83 | Yes |
| 40 | TIME_LIMIT | 216 | Yes |
| 50 | TIME_LIMIT | 273 | Yes |

These statuses are copied from the **historical CSV**, not freshly certified. The independent check recomputed `E=Σ_k(Σ_i x_i x_{i+k})²` from every saved bit string and found exact agreement with the recorded objective. This is a data-integrity check, **not** a Gurobi rerun or fresh optimality proof. The code uses an auxiliary-XOR **constrained MIQP**, not an unconstrained QUBO.

Associated artifacts: CSV records of runs, checkpoints and incumbent events; solver logs; stored best sequences; and two plots.

## Cross-language MATLAB/Python experiments

Location: `results/cross_language/` with three CSV files, plus scripts in `py_mtlb_efficiency/` and MATLAB code under `matlab/`. Those results are retained as historical records. MATLAB execution and matched-runtime claims were not reverified in the current environment.

## Classical search and landscape work not in this bundle

Later TABU/VNS cross-size summary values, detailed basin-escape and guided perturbation experiments, Legendre-sequence tests, cascade survival experiments, and subsequent ALNS parameter sweeps discussed outside the Git repository do **not** appear as the raw result set in the supplied bundle. Do not imply their figure data are reproducible solely from this repository until the relevant files and scripts have been reviewed and added.

## Suggested minimum before submission

For published comparisons, state machine details, code commit, solver settings, seed(s), time limit, sample count, summary statistic (best/mean/median), objective convention and exact external reference used. Avoid mixing Gurobi certified lower bounds with heuristic objective comparisons without explaining that they answer different questions.
