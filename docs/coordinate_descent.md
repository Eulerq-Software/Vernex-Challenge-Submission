# Coordinate Descent Solver

## Purpose

`CoordinateDescentSolver` is a local-search baseline for discrete-phase radar waveform optimization. It improves one phase index at a time while keeping every other phase index fixed.

## Algorithmic procedure

The solver uses the **best-improvement** version of Coordinate Descent.

1. Start with a valid phase-index sequence.
2. Visit coordinates from left to right.
3. For the current coordinate, test every phase index from `0` to `Q - 1`.
4. Keep all other coordinates fixed.
5. Evaluate each candidate using the shared `islr_from_phase_indices` objective.
6. Apply the phase value with the lowest ISLR only if it improves the current solution.
7. Repeat full left-to-right passes until no coordinate change improves the objective or a configured limit is reached.

One complete pass across all `N` coordinates is counted as one iteration.

## Initialization

- If `initial_solution` is supplied, it is validated and used as the starting waveform.
- Otherwise, the solver generates a random valid waveform using `random.Random(seed)`.
- The random initialization is reproducible when the same seed is used.



## Stopping criteria

The solver stops when one of the following occurs:

- A full coordinate pass makes no improvement. The solver has reached a local optimum.
- The runtime limit is reached.
- `max_iterations` full passes have been completed.
- `max_evaluations` objective-function evaluations have been completed.



## Parameters

- `N`: waveform length; must be at least 2.
- `Q`: number of phase choices; must be at least 2.
- `seed`: random seed for reproducible random initialization.
- `time_limit`: optional maximum runtime in seconds.
- `initial_solution`: optional user-supplied valid starting waveform.
- `max_iterations`: optional maximum number of full coordinate passes.
- `max_evaluations`: optional maximum number of ISLR evaluations.
- `record_history`: when `True`, records best-so-far progress after each completed pass.



## Evaluation counting

The initial waveform requires one objective evaluation. Each tested phase value for a coordinate requires one additional objective evaluation.

A completed full pass evaluates up to `N × Q` candidate solutions.

Therefore, a run with `P` completed full passes has approximately `1 + P × N × Q` objective evaluations, unless a time or evaluation limit stops the solver earlier.

## Computational complexity

For each full pass, the solver evaluates up to `N × Q` candidates. The shared direct ISLR calculation is approximately `O(N²)`. Therefore, the approximate runtime complexity is `O(P × Q × N³)`, where `P` is the number of full coordinate passes.

Memory usage is approximately `O(N)`, excluding temporary candidate lists.

## Benchmark results

Five seeds (`11`, `22`, `33`, `44`, and `55`) were run for each binary waveform instance with `Q=2`. The solver used best improvement and a maximum of 20 full passes. Enumeration provides the exact global optimum for these small instances.


| N   | Exact ISLR | Best Coordinate Descent ISLR | Mean ISLR | Standard deviation | Mean runtime (ms) | Scores across five seeds |
| --- | ---------- | ---------------------------- | --------- | ------------------ | ----------------- | ------------------------ |
| 5   | 2          | 2                            | 3.6       | 1.960              | 0.539             | [6, 2, 6, 2, 2]          |
| 7   | 3          | 3                            | 5.4       | 3.200              | 0.827             | [3, 3, 11, 7, 3]         |
| 11  | 5          | 13                           | 20.2      | 3.919              | 1.243             | [13, 21, 25, 21, 21]     |




## Strengths

- Simple and easy to interpret.
- Usually much faster than exhaustive enumeration.
- Improves a starting waveform systematically.
- Supports user-provided initial solutions.
- Produces reproducible results when a seed is fixed.



## Weaknesses

- It considers only one-coordinate changes.
- It may stop at a local optimum that is worse than the global optimum.
- Its final quality depends on the initialization seed.
- It does not escape local optima unless optional random restarts or perturbations are added later.

