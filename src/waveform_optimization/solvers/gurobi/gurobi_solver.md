# Gurobi Exact-Optimization Results (Surya)

## Background: what Gurobi is actually proving

Gurobi solves this as a **mixed-integer quadratic program (MIQP)** via
**branch-and-bound**, not by guessing-and-improving like ALNS does. The
concepts below (primal, dual, bound, gap) are what let it _prove_ a
solution is optimal rather than just report a good one.

- **Primal problem** — the actual problem we care about: find binary
  `b_i ∈ {0,1}` (and the derived XOR variables `z_ij`) satisfying every
  constraint, minimizing `Σ C_k²`. Any _feasible_ assignment of all
  variables is a valid primal solution. The best one found so far is the
  **incumbent**, and its objective value is the **upper bound** — it's a
  real, achievable answer, just not necessarily the best possible one yet.

- **Dual / relaxation problem** — at every node of the search tree, Gurobi
  temporarily relaxes the binary constraints (`b_i ∈ {0,1}` → `b_i ∈ [0,1]`)
  and solves the resulting convex quadratic relaxation. Because relaxing a
  constraint can only make a minimization problem's optimal value _smaller
  or equal_, never larger, this relaxed value is a guaranteed **lower
  bound** on the true optimum — this is the "linear/dual" side you asked
  about (technically a QP relaxation here, not a plain LP, but it plays the
  same role an LP relaxation plays in classic branch-and-bound). The best
  (largest) such bound across all remaining unexplored nodes is the
  **best bound** Gurobi reports.

- **Gap** — how far apart these two provable facts still are:

  ```
  gap = (upper_bound − lower_bound) / |upper_bound|
      = (incumbent − best_bound) / |incumbent|
  ```

  A large gap doesn't mean the incumbent is bad — it means Gurobi hasn't
  yet _proven_ it's good. A gap of 800% (N=30) just means the current best
  feasible answer (83) is a long way above the best bound Gurobi has
  managed to certify (-581) — it says nothing about how close 83 actually
  is to the true optimum.

- **Why convergence at gap = 0 means "done":** the upper bound is a real,
  achievable solution; the lower bound is a mathematical guarantee that
  _no_ feasible solution can score better. If those two numbers ever meet,
  you have a feasible solution that is simultaneously proven to be
  unbeatable — that's what `status = OPTIMAL` means in `gurobi_runs.csv`.
  Until they meet, all you have is "best guess so far" (upper bound) plus
  "can't be sure it's not this good" (lower bound), with a real, unproven
  gap between them — which is exactly why N≥25 are reported as
  `TIME_LIMIT`, not `OPTIMAL`, even though they did return usable feasible
  sequences.

Companion to [`docs/team_assignments.md`](../team_assignments.md), section 4
of the team brief. Covers the exact Q=2 binary MIQP: minimize the integrated
sidelobe level (ISL) `Σ C_k²` for N ∈ {5, 7, 11, 15, 20, 25, 30, 40, 50}, one
deterministic cold-start run per N, no ALNS warm start.

## Where everything lives

```
src/waveform_optimization/solvers/gurobi/
├── gurobi_exact_solver.py          # model builder + solve + logging (the reproducible builder)
├── gurobi_N{5,7,11,...,50}.log     # raw Gurobi console output, one per N — mandatory deliverable
└── results/gurobi_exact/surya/
    ├── gurobi_runs.csv             # (A) one row per N: final status, energy, bound, gap, timings
    ├── gurobi_checkpoints.csv      # (B) fixed-time snapshots: 0,10,30,60,120,300,600,900,1200,1800s
    ├── gurobi_incumbents.csv       # (C) every new-incumbent / bound-update event
    ├── plot_convergence.py         # regenerates the two PNGs below from the 3 CSVs above
    │                               #   (rename from `lpot_convergence.py` — likely a typo)
    ├── plots/
    │   ├── convergence_grid.png    # incumbent-vs-time & bound-vs-time, one panel per N
    │   └── final_quality_vs_N.png  # summary: certified (green) vs time-limited (orange), all N
    └── sequences/
        └── N{5,7,11,...,50}_best.txt   # (D) best phase sequence per N, plain 0/1 string
```

Maps directly onto the brief's section-5 hand-off list:

| #   | Deliverable                         | Location                                                                                                                                     |
| --- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Run-level CSV                       | `gurobi_runs.csv`                                                                                                                            |
| 2   | Checkpoint/progress CSV             | `gurobi_checkpoints.csv`                                                                                                                     |
| 3   | Improvement/incumbent event CSV     | `gurobi_incumbents.csv`                                                                                                                      |
| 4   | Best sequences                      | `sequences/N*_best.txt`                                                                                                                      |
| 5   | Exact config + git commit           | `solver_parameters_json` / `git_commit` columns in `gurobi_runs.csv`, plus `gurobi_exact_solver.py` itself as the reproducible model builder |
| 6   | Raw logs/errors                     | `gurobi_N*.log` (one per N)                                                                                                                  |
| 7   | Anomalies note                      | **TODO — not yet written**, see below                                                                                                        |
| —   | Convergence plots (Gurobi-specific) | `plots/convergence_grid.png`, `plots/final_quality_vs_N.png`                                                                                 |

The `.lic` license file is **not** and should never be in this tree — it
lives outside the repo per the brief.

## Results summary

| N   | role       | status      | best feasible (E) | best proven bound | gap     | Barker ratio | time to proof / cutoff |
| --- | ---------- | ----------- | ----------------- | ----------------- | ------- | ------------ | ---------------------- |
| 5   | validation | **OPTIMAL** | 2                 | 2                 | 0%      | 1.00         | 0.015 s                |
| 7   | validation | **OPTIMAL** | 3                 | 3                 | 0%      | 1.00         | 0.009 s                |
| 11  | validation | **OPTIMAL** | 5                 | 5                 | 0%      | 1.00         | 0.111 s                |
| 15  | scale-up   | **OPTIMAL** | 15                | 15                | 0%      | 2.14         | 2.87 s                 |
| 20  | scale-up   | **OPTIMAL** | 26                | 26                | 0%      | 2.74         | 250.1 s (of 300 s)     |
| 25  | bounds     | TIME_LIMIT  | 68                | -260              | 482.4%  | 5.67         | cutoff at 600 s        |
| 30  | bounds     | TIME_LIMIT  | 83                | -581              | 800.0%  | 5.72         | cutoff at 900 s        |
| 40  | bounds     | TIME_LIMIT  | 216               | -3716             | 1820.4% | 11.08        | cutoff at 1200 s       |
| 50  | bounds     | TIME_LIMIT  | 273               | -8663             | 3273.3% | 11.14        | cutoff at 1800 s       |

**Validation:** N=5, 7, 11 all match the known Barker optima (2, 3, 5) and
`objective_reference_validation = match` for every N (independent
`energy_from_bits` recomputation agrees with Gurobi's reported objective) —
confirms the MIQP formulation itself is correct, separately from whether it
converges in time.

**Reading the table:** only `OPTIMAL` rows are _certified_ — the energy is
proven best-possible. `TIME_LIMIT` rows are the best _feasible_ sequence
found in the time budget with no such proof; for these, `gap` measures
distance from a loose relaxation bound, not from a competing solution — this
is expected for N≥25 given the problem's combinatorial difficulty, not a
bug (see `plots/final_quality_vs_N.png` for the visual version of this
distinction).

## Anomalies and completion notes

- **Completion:** all 9 planned N values (5, 7, 11, 15, 20, 25, 30, 40, 50)
  completed with no interrupted or skipped runs — confirmed by 9 rows in
  `gurobi_runs.csv`, 9 matching `gurobi_N*.log` files, and 9 matching
  `sequences/N*_best.txt` files, all cross-checked consistent with each
  other (same best-energy, same best-sequence, same gap in each source).

- **Difficulty scaling / N≥25 not converging is expected, not a
  misconfiguration:** N=15 solved in 2.9 s and N=20 in 250.1 s of its 300 s
  budget — already a ~90x jump in solve time for one step in N, plus
  node counts jumping from 3,345 (N=15) to 33,315 (N=20). This is the
  expected exponential blow-up of this problem class, not a bug in the
  model or solver settings; N=25–50 continuing that trend and hitting
  `TIME_LIMIT` (gaps of 482%–3273%) is the anticipated continuation of the
  same pattern, not a sign anything was configured incorrectly. All four
  time-limited runs still returned valid, reference-checked feasible
  sequences (`objective_reference_validation = match` in every row) — they
  simply weren't _proven_ optimal within budget.

- **Gurobi WLS licensing:** a one-time `gurobipy._exception.GurobiError:
Unauthorized access` was hit before these runs, while Gurobi was still
  correctly picking up the WLS credentials (`Set parameter WLSAccessID` /
  `WLSSecret` / `LicenseID` printed fine) but failing the cloud
  authentication handshake. _[Surya: fill in the specific step that
  actually fixed it — e.g. confirming the license was Active on the
  portal, re-downloading a fresh `gurobi.lic` after a secret
  regeneration, or a clock/network fix — so this note is precise rather
  than just "it started working."]_ All 9 runs in this results set
  authenticated successfully (`license_test_result = ok` in every row),
  confirming it was a one-time environment issue and not a recurring
  problem affecting these results.

- **Cold-start only, per the brief's requirement:** every run here starts
  from Gurobi's own root relaxation with no ALNS-provided warm start,
  satisfying the brief's "Gurobi standalone first: no ALNS warm start in
  the primary comparison" requirement. If a warm-start variant is added
  later, it must be logged as a separate, clearly labeled experiment, not
  folded into these numbers.
