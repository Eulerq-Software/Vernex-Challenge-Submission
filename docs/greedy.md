# Greedy Solver

## Purpose

`GreedySolver` is a simple deterministic baseline for discrete-phase radar waveform optimization. It constructs a phase-index sequence one coordinate at a time and minimizes the shared Integrated Sidelobe Level (ISLR) objective.

## Strategy

The solver assigns coordinates from left to right.

For each position:

1. Try every phase index from `0` to `Q - 1`.

2. Retain already assigned values.

3. Temporarily set all unassigned positions to phase index `0`.

4. Evaluate the complete candidate with the shared `islr_from_phase_indices` function.

5. Select the phase with the lowest ISLR value.

6. Resolve ties by retaining the lowest phase index.

This is called the **zero-fill completion strategy**.

## Parameters

- `N`: waveform length; must be at least 1.

- `Q`: number of phase choices; must be at least 2.

- `time_limit`: optional maximum runtime in seconds.

- `seed`: accepted for interface compatibility. The current solver is deterministic.

- `initial_solution`: optional valid full waveform considered as an additional incumbent solution.

## Output

The solver returns a `SolverResult` containing the best valid phase-index sequence, its objective value, runtime, objective-evaluation count, status, and solver metadata.

For a completed run without an initial solution, the expected number of objective evaluations is:

\[

1 + NQ

\]

The first evaluation is the all-zero initial completion, followed by `Q` candidate evaluations for each of the `N` coordinates.

## Benchmark results

The following comparison used binary waveforms `Q=2`) and the supplied exhaustive enumeration solver. Runtime values are machine-dependent.

| N | Greedy ISLR | Exact ISLR | Gap | Greedy runtime (ms) | Enumeration runtime (ms) |

|---:|---:|---:|---:|---:|---:|

| 5 | 6 | 2 | 4 | 0.936 | 1.227 |

| 7 | 7 | 3 | 4 | 0.688 | 3.591 |

| 11 | 21 | 5 | 16 | 1.072 | 96.738 |

## Limitation

Greedy search is fast and easy to interpret, but it is myopic. A phase that looks good under the zero-fill completion at an early coordinate can prevent a better global waveform later. It should therefore be used as a baseline, not as an expected optimum solver.