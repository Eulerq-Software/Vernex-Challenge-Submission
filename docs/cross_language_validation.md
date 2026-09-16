# Python–MATLAB Cross-Language Validation

Per challenge doc §6.6. This file is **append-only in spirit** — add a new
`## Machine: <hostname>` section each time results come in from a new machine,
and a new row to the relevant table each time a new solver is benchmarked.
Do not delete or overwrite past entries.

## How the test is run

- Harness: `scripts/xlang_benchmark.sh`, driving `xlang_run_python_solver.py`
  and `xlang_run_solver.m` (invoked via `matlab -batch` or `octave --eval`).
- For each `(solver, N, Q)`, the same solver runs once in Python and once in
  MATLAB/Octave, with **no time limit** (`time_limit=None`/`[]`), so both
  sides do identical work.
- Every run is one row in `<host>.csv`; every `(python, matlab)` pair for the
  same instance is compared and logged as one row in `<host>_comparison.csv`.
- Results files are append-only and per-machine (hostname + hardware
  fingerprint), so runs from different machines never mix.

## How results are verified

1. **Independent recomputation** — each language recomputes its own reported
   objective through the shared core (`islr_from_phase_indices`) and checks
   agreement (`recompute_ok_python` / `recompute_ok_matlab`).
2. **Cross-language agreement** — `|F_python − F_matlab| < ε` (`tolerance`
   column, default `ε = 1e-9`) → `within_tolerance`.
3. **Validity** — both solutions checked against `0 ≤ qᵢ < Q`, `len(q) = N`.
4. **`indices_identical`** — informational only. Ties are common (many phase
   sequences can share the same objective), so this is expected to be
   `false` sometimes even when objectives match; it is not a failure signal.

## Machine: `su-MS-7E70`

|             |                                                                                     |
| ----------- | ----------------------------------------------------------------------------------- |
| CPU         | AMD Ryzen 9 7900X, 12-core / 24 threads                                             |
| RAM         | 61.9 GiB                                                                            |
| OS          | Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic                                         |
| Python      | 3.12.12                                                                             |
| "MATLAB"    | **Octave 8.4.0** (no MATLAB license — run via `octave --eval`, not `matlab -batch`) |
| Fingerprint | `f6dfc5df`                                                                          |
| Files       | `results/cross_language/su-MS-7E70.csv`, `su-MS-7E70_comparison.csv`                |

> Note: since Octave stands in for MATLAB here, "MATLAB" below means "the
> MATLAB-language implementation run on Octave." Octave and MATLAB can differ
> slightly in numerics/timing, so treat this as Octave validation until run
> on real MATLAB.

### Results — `enumeration` solver

N = 4–12, Q ∈ {2, 3}, 18 instance pairs.

- **18/18 within tolerance.** All objectives agree to ~1e-15 (floating-point
  noise only).
- All solutions valid on both sides; objective-evaluation counts match
  exactly (both do full `Q^N` enumeration).
- `indices_identical` is `false` for N∈{6,8,11,12}, Q=3 — expected tie
  behavior, not a bug (objectives still match exactly).
- **Runtime:** Octave is consistently slower than Python, by **~3.4×–5.2×**
  across all instances, growing with problem size (largest: N=12, Q=3 →
  17.5 s Python vs 90.6 s Octave).

### Results — `greedy` solver

N = 4–12, Q ∈ {2, 3}, 18 instance pairs.

- **17/18 within tolerance.**
- ⚠️ **1 mismatch:** `greedy, N=10, Q=3` — Python objective `21.0`, Octave
  objective `18.0` (`abs_diff ≈ 3.0`, exceeds `ε`). Both sides individually
  pass recomputation and validity checks, and both report status
  `completed`, so this is not a bug in the objective/recompute logic — the
  two greedy implementations made different tie-breaking / assignment-order
  choices at N=10, Q=3 and landed in different basins. **Needs follow-up**
  (compare tie-break rules in the Python vs MATLAB greedy solvers per
  §4.2.2/§6.2).
- Everywhere else, objectives match to ~1e-15 despite `indices_identical`
  being `false` in several rows (again, tie behavior).
- **Runtime:** Octave slower by **~2.9×–4.5×**, smaller absolute gap than
  enumeration since greedy does far fewer objective evaluations.

### Summary — `su-MS-7E70`

| Solver      | Pairs | Within tolerance | Notes                                                        |
| ----------- | ----- | ---------------- | ------------------------------------------------------------ |
| enumeration | 18    | 18/18            | Octave ~3.4–5.2× slower                                      |
| greedy      | 18    | 17/18            | 1 mismatch at N=10,Q=3 — open issue; Octave ~2.9–4.5× slower |

_(random_search, coordinate_descent, simulated_annealing not yet registered
on the MATLAB side — see `SOLVERS` in `xlang_benchmark.sh`.)_

---

<!-- Add new "## Machine: <hostname>" sections below as more machines report in. -->
