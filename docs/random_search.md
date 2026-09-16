# Random Search Solver

## Purpose

`RandomSearchSolver` is a stochastic baseline for discrete-phase radar waveform optimization. It repeatedly generates random valid phase-index sequences, evaluates the shared ISLR objective, and retains the best solution found.

## Method

For each iteration, the solver:

1. Generates a phase-index sequence of length `N`.

2. Samples every phase index uniformly from `0` to `Q - 1`.

3. Evaluates the candidate using `islr_from_phase_indices`.

4. Replaces the current best solution only when the candidate has a lower ISLR value.

5. Stops when a time limit, iteration limit, or objective-evaluation limit is reached.

## Parameters

- `N`: waveform length; must be at least 1.

- `Q`: number of phase choices; must be at least 2.

- `seed`: random seed used for reproducible runs.

- `time_limit`: optional maximum runtime in seconds.

- `max_iterations`: optional maximum number of randomly generated candidates.

- `max_evaluations`: optional maximum number of objective-function evaluations.

- `record_history`: when `True`, records best-so-far progress after every iteration.

- `initial_solution`: optional valid initial waveform considered as an incumbent solution.

## History format

When `record_history=True`, each iteration records:

```python

{

    "elapsed_seconds": 0.42,

    "objective": 18.0,

}

## Evaluation counting

The solver first evaluates one valid all-zero waveform as the initial best solution. Each random candidate then requires one additional ISLR evaluation.

Therefore, when the solver runs for `M` random iterations, the expected total is:

\[

1 + M

\]

For example, with `max_iterations=1000`, the solver performs:

\[

1 + 1000 = 1001

\]

objective evaluations.

If `max_evaluations` is set, the solver stops before exceeding that configured evaluation budget.

## Benchmark results

Five seeds `11, 22, 33, 44, 55`) were run for each binary waveform instance with `Q=2` and 1,000 random candidates per seed. Enumeration provides the exact global optimum for these small instances.

| N | Exact ISLR | Best Random ISLR | Mean Random ISLR | Standard deviation | Mean Random runtime (ms) | Scores across five seeds |

|---:|---:|---:|---:|---:|---:|---|

| 5 | 2 | 2 | 2.0 | 0.000 | 29.371 | [2, 2, 2, 2, 2] |

| 7 | 3 | 3 | 3.0 | 0.000 | 27.537 | [3, 3, 3, 3, 3] |

| 11 | 5 | 5 | 6.6 | 3.200 | 38.395 | [5, 5, 13, 5, 5] |

For `N=5` and `N=7`, all five random-search runs found the exact optimum. For `N=11`, four out of five runs found the optimum ISLR of `5`, while one run found `13`.

## Limitation

Random Search does not use information from earlier samples to construct better future samples. Each candidate is independent.

It can find strong solutions with a sufficient evaluation budget, but it does not guarantee the global optimum. Result quality can vary with the random seed, especially as waveform length increases. This makes Random Search a useful stochastic baseline, but not a guaranteed optimal solver.