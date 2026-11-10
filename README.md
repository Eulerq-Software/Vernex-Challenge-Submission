# Hybrid ALNS–Quantum Optimization for Phase-Coded Radar Waveforms

**VerneX / Thales radar waveform challenge — consolidated research repository**

**Proposed algorithm:** a hybrid **Adaptive Large Neighbourhood Search (ALNS) + quantum neighbourhood-repair solver** for radar waveform optimization. ALNS searches the complete waveform, while a quantum routine is designed to solve or sample selected smaller subproblems. Classical local refinement and full-objective evaluation determine whether the quantum-generated candidate improves the incumbent.

This repository contains the **implemented classical foundation**—ALNS, other baselines, exact/constrained-MIQP validation, benchmark infrastructure and MATLAB experiments—on which that hybrid algorithm is built. The principal experimentally tested setting is the matched-filter, binary-phase problem (`Q=2`). The quantum repair algorithm is **specified in the dossier, but has not yet been implemented or benchmarked for LABS in this repository**. We propose a pathway to larger phase alphabets (`Q>2`); no quantum advantage is claimed.

## Proposed solution: ALNS with quantum repair

We do **not** propose to solve every large LABS instance solely on a quantum processor. A direct full-length higher-order cost Hamiltonian may be dense and expensive to compile, while a complete QUBO conversion introduces auxiliary variables and potentially difficult penalty and embedding overhead. These are **practical resource limitations**, not a proof that quantum-only algorithms are impossible.

Instead, the proposed hybrid solver assigns complementary roles:

1. **ALNS controls the full problem.** It initializes a candidate binary waveform, selects a subset `S` of `m << N` active bits, and freezes all remaining bits.
2. **The selected neighbourhood is encoded exactly.** We construct the restricted higher-order Ising Hamiltonian, or its valid quadratized QUBO, including interactions with frozen variables.
3. **Quantum repair proposes candidates.** QAOA (or a QUBO-based quantum optimizer) explores assignments to the `m` active bits. A matched-budget classical repair operator is always available as a fallback and comparator.
4. **Classical search completes the move.** We restore the complete waveform, perform local descent, calculate the *original* LABS energy, and update the ALNS incumbent and adaptive operator scores.
5. **Measured evidence decides whether quantum is used.** We evaluate final energy, productive basin escapes per unit time, and end-to-end resource costs—not just quantum circuit quality or low immediate subproblem energy.

A specific experimental direction is **basin-trained QAOA**: use the restricted LABS Hamiltonian in the quantum circuit, but train its parameters by scoring measured candidates *after* classical descent. This avoids presuming a reversible quantum implementation of the descent algorithm and targets the productive uphill triggers observed in the classical landscape studies. The classical experiments motivate this choice; its comparative quantum performance remains to be tested.

**Technical proposal:** [Algorithm Design Dossier (PDF)](docs/Algorithm_Design_Dossier.pdf) · [LaTeX source](docs/Algorithm_Design_Dossier.tex) · [Quantum execution design and status](docs/quantum_scope.md).

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
| ALNS + quantum repair (QAOA / QUBO-based alternatives) | **Proposed algorithm; quantum component not yet implemented for LABS** | [Design and status](docs/quantum_scope.md) |
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
    quantum/            proposed repair interface (no implemented LABS quantum solver)
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

