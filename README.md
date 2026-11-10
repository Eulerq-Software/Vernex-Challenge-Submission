# Phase-Coded Radar Waveform Optimization

**VerneX / Thales radar waveform challenge — consolidated research repository**

This repository studies the design of phase-coded radar waveforms with low aperiodic autocorrelation sidelobes. It contains **implemented classical optimizers**, exact/MIQP validation, reproducible benchmark utilities, MATLAB cross-language validation, and experimental artifacts recovered from the original team branches. The primary tested scope is the **matched-filter, binary-phase problem (`Q=2`)**. Quantum and higher-phase methods are prospective directions, not validated implementations in this repository.

## Mathematical problem

For `N` chips and `Q` equally spaced phases, phase indices `q_i ∈ {0,...,Q-1}` give a constant-modulus code `x_i = exp(-2πj q_i / Q)`. For positive lags `k=1,...,N-1`:

`ρ(k) = Σ_{i=0}^{N-k-1} x_i conj(x_{i+k})`, and `E(q) = Σ_{k=1}^{N-1} |ρ(k)|²`.

For `Q=2`, `x_i ∈ {+1,-1}`, this is the **low-autocorrelation binary sequences (LABS)** objective. The raw spin objective generally has up to four-body interactions. The code in the classical solver stack evaluates this objective **directly**, not through an unconstrained QUBO. Separately, the Gurobi module introduces auxiliary XOR variables and linear constraints to build a **constrained mixed-integer quadratic program (MIQP)**. This is *not* an implemented unconstrained QUBO or quantum algorithm. See [mathematical definition](docs/mathematical_definition.md) and [quantum status](docs/quantum_scope.md).

Merit factor is `MF = N² / (2E)` for `E>0`. Lower energy / higher MF is better. The `E / ((N-1)/2)` scale is the organizers' Barker-code reference convention, **not** a general approximation ratio.

## What is implemented?

| Area | Status | Location |
| --- | --- | --- |
| Reference autocorrelation, phase encoding and validation | Implemented, unit-tested | `src/waveform_optimization/core/` |
| Enumeration and known-solution validation | Implemented, unit-tested | `src/waveform_optimization/solvers/enumeration/` |
| Greedy, random search, coordinate descent | Implemented, unit-tested | `src/waveform_optimization/solvers/weak_baselines/` |
| Genetic algorithm, simulated annealing, Tabu, VNS | Implemented, unit-tested | `src/waveform_optimization/solvers/strong_baselines/` |
| Adaptive large neighbourhood search (ALNS) | Implemented; benchmarking adapter and telemetry available | `src/waveform_optimization/solvers/alns/` |
| Constrained MIQP and exactness/bound experiments (Gurobi) | Implemented; optional commercial dependency | `src/waveform_optimization/solvers/gurobi/` |
| MATLAB implementations and cross-language experiments | Included; MATLAB tests not run in this consolidated environment | `matlab/`, `py_mtlb_efficiency/`, `results/cross_language/` |
| Quantum QAOA / quantum annealing / SimCIM for LABS | **Not implemented in supplied Git bundle** | [Proposed design](docs/quantum_scope.md) |
| `Q>2` extensions | Core objective supports multiple phases; most tested search algorithms are for `Q=2` | [Scope](docs/quantum_scope.md) |

## Quick start

Python 3.10+ is required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pytest -q
```

The optional Gurobi module also requires a separately installed `gurobipy` and a valid Gurobi license. MATLAB scripts need a MATLAB installation. They are *not* prerequisites for the normal Python test suite.

### Validate known binary sequences

```bash
python experiments/scripts/run_validation.py
python experiments/scripts/run_enumeration.py
```

### Run binary-phase baselines

```bash
python experiments/scripts/run_benchmark.py --help
python experiments/scripts/run_strong_baselines_benchmark.py --help
```

For a quick smoke run of the weak baselines:

```bash
python experiments/scripts/run_benchmark.py --algorithms greedy --N 5 --Q 2 --runs 1 --time-limit 0.01 --output-root results/local/smoke
```

**Read `--help` before running a benchmark.** The checked-in `experiments/configs/benchmark.yaml` includes sizes up to `N=1000` and potentially long per-run budgets; do not accidentally launch the full schedule when intending a smoke test. `experiments/configs/development.yaml` is now populated with a short development schedule.

For ALNS, the standard benchmarking adapter is `waveform_optimization.solvers.alns.adapter.ALNSWaveformSolver`. The ALNS convergence experiment runner is `experiments/scripts/run_alns_convergence_experiment.py`; more details are in [consolidation notes](docs/consolidation_notes.md).

## Repository map

```text
src/waveform_optimization/
  core/                 direct ISLR/ISL objective and validation
  instances/            generators and known solutions
  solvers/
    enumeration/        exhaustive reference solver
    weak_baselines/     greedy, random and coordinate descent
    strong_baselines/   GA, simulated annealing, Tabu, VNS
    alns/               ALNS, repair, telemetry, benchmark adapter
    gurobi/             constrained MIQP and historical outputs
    quantum/            future-work marker (no demonstrated solver)
  benchmarking/         benchmark result schema, validation, aggregation
  visualization/        plotting utilities
experiments/
  configs/              benchmark and ALNS settings
  scripts/              benchmark and convergence runners
matlab/                 MATLAB reference solvers and tests
py_mtlb_efficiency/     historical cross-language runner scripts
results/cross_language/ historical cross-language result CSVs
docs/                   technical, reproduction and branch consolidation notes
tests/                  Python tests
```

## Results and provenance

This is a **merge of distinct historical contributor branches**. See [consolidation notes](docs/consolidation_notes.md) and [results catalog](docs/results_catalog.md) for branch-by-branch provenance, merge choices and known limitations. Existing Gurobi logs, benchmark CSVs and historical solver scripts remain in their original relative locations to minimize disruption to reproducibility. No new benchmark measurements are represented as historical team results.

**Important:** The large later LABS landscape / guided-escape datasets described in the team's separate research discussions were **not present in the supplied Git bundle**. They must be added separately before claiming that those raw results are reproducible from this repository.

## Submission guidance

Use the Algorithm Design Dossier as the scientific description and link to a fixed repository commit (or a release tag) for reproducibility. This README describes what the code actually contains; it does not claim demonstrated quantum speedup, quantum hardware runs, or a validated QUBO-based LABS optimizer.

## Contributors and licensing

Work is credited through preserved Git history and original branch commits. **No open-source licence has been selected in the supplied repository.** Verify ownership, challenge sharing rules, permissions and any applicable IP agreements before making this repository public or adding a licence.
