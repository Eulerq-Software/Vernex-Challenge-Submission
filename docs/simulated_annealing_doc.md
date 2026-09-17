# Simulated Annealing

## 1. Algorithmic procedure

Simulated Annealing explores the search space by moving to a neighboring
solution at every iteration, occasionally accepting a worse solution in order
to escape local minima.

Given a current solution `q` and a candidate solution `q'`, define

$$
\Delta = F(q') - F(q).
$$

The candidate is accepted according to:

- If `Δ ≤ 0`, the candidate is always accepted.
- If `Δ > 0`, the candidate is accepted with probability

$$
P(\text{accept}) = \exp\left(-\frac{\Delta}{T}\right),
$$

where `T` is the current temperature.

At each temperature level, `iterations_per_temperature` candidate moves are
attempted. After completing a level, the temperature is reduced using a
geometric cooling schedule:

$$
T_{\text{new}} = \alpha \cdot T, \qquad 0 < \alpha < 1.
$$

The outer loop continues until the temperature drops to or below
`final_temperature`, the configured `time_limit` is reached, or
`max_iterations` total candidate moves have been attempted.

### Neighborhood

The neighborhood operation is a single-coordinate change: one coordinate is
selected at random and its phase index is replaced with a different valid
phase value, chosen by shifting the current value by a random nonzero offset
modulo `Q`. This guarantees the candidate always differs from the current
solution.

## 2. Initialization

- If `initial_solution` is supplied, it is validated and used as the starting
  point.
- Otherwise, a random valid phase-index sequence is generated using the
  solver's local random generator.

The best-found solution is initialized to the starting solution before the
search begins.

## 3. Stopping criteria

The search stops when any of the following occurs first:

- The temperature reaches `final_temperature` (status `completed`).
- `max_iterations` total candidate moves have been attempted, if configured
  (status `completed`).
- The configured `time_limit` is exceeded (status `time_limit_reached`).

## 4. Computational complexity

Each iteration performs one objective evaluation, and one objective
evaluation costs `O(N^2)` in the worst case (aperiodic autocorrelation over
all lags). The total number of iterations is bounded by

$$
\left\lceil \log_{\alpha}\left(\frac{T_{\text{final}}}{T_{\text{initial}}}\right) \right\rceil \times \text{iterations\_per\_temperature},
$$

so the overall cost is that quantity multiplied by the per-evaluation cost,
unless capped earlier by `max_iterations` or `time_limit`.

## 5. Randomization behavior

- Uses a local `random.Random(seed)` generator; no global random state is
  touched.
- Coordinate selection, offset selection, and the Metropolis acceptance draw
  all come from this generator.
- Given the same `seed` and the same parameters, the search is fully
  reproducible.

## 6. Strengths and weaknesses

**Strengths**
- Simple to implement and tune.
- Can escape local minima through controlled acceptance of worse moves.
- Naturally trades exploration for exploitation as temperature decreases.

**Weaknesses**
- Performance is sensitive to the choice of `initial_temperature`,
  `cooling_rate`, and `iterations_per_temperature`; poor settings lead to
  premature convergence or excessive runtime.
- Uses no memory of past moves, unlike Tabu Search, so it can revisit the
  same region repeatedly.
- Single-coordinate neighborhood limits how far the search can move in one
  step.

## 7. Algorithm-specific parameters

| Parameter | Meaning | Default |
|---|---|---|
| `initial_temperature` | Starting temperature `T` | `10.0` |
| `final_temperature` | Temperature at which the search stops | `1e-3` |
| `cooling_rate` | Geometric cooling factor `α` | `0.95` |
| `iterations_per_temperature` | Candidate moves attempted per temperature level | `50` |
| `max_iterations` | Optional cap on total candidate moves | `None` |
| `time_limit` | Optional runtime limit in seconds | `None` |
| `seed` | Random seed for reproducibility | `None` |
| `initial_solution` | Optional starting phase-index sequence | `None` |

## 8. Result metadata

In addition to the common `SolverResult` fields, `metadata` reports:

- `initial_temperature`, `final_temperature_configured`,
  `final_temperature_reached`, `cooling_rate`, `iterations_per_temperature`.
- `total_iterations`: total candidate moves attempted.
- `accepted_moves`: number of candidates accepted (including uphill moves).
- `improving_moves`: number of accepted candidates that strictly improved the
  current objective.

## 9. Example usage

```python
from waveform_optimization.solvers.strong_baselines.simulated_annealing import (
    SimulatedAnnealingSolver,
)

solver = SimulatedAnnealingSolver()
result = solver.solve(N=31, Q=4, seed=42, time_limit=5.0)

print(result.best_objective)
print(result.best_phase_indices)
```
