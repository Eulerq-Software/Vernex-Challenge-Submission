# Tabu Search

## 1. Algorithmic procedure

Tabu Search is a local-search method that uses short-term memory to
discourage cycling back to recently visited moves. At each iteration:

1. Generate the neighborhood of candidate moves from the current solution.
2. Evaluate every candidate in the neighborhood.
3. Exclude tabu moves, unless the aspiration criterion is satisfied.
4. Select the best admissible candidate.
5. Move to the selected candidate and update the tabu list.
6. Update the global best solution if the candidate improves on it.

### Neighborhood

A move changes a single coordinate: `(index, new_value)`. The full
neighborhood contains at most `N * (Q - 1)` moves. If `neighborhood_size` is
set, a random subset of that size is sampled each iteration instead of
enumerating the full neighborhood, which bounds per-iteration cost for large
instances.

### Move selection

Best-improvement: every candidate in the (possibly sampled) neighborhood is
evaluated, and the best admissible one is selected, rather than stopping at
the first improving move.

### Tabu list and aspiration

The tabu list stores the *reverse* of each applied move. When coordinate `i`
changes from `old_value` to `new_value`, the entry `(i, old_value)` becomes
tabu for `tabu_tenure` iterations, forbidding the search from immediately
moving that coordinate back to `old_value`.

The aspiration criterion (enabled by default via `use_aspiration`): a tabu
move is accepted anyway if it produces a solution strictly better than the
best solution found so far. If every candidate in the neighborhood happens to
be tabu and none satisfy aspiration, the algorithm falls back to treating all
candidates as admissible for that iteration rather than stalling.

### Diversification

An optional diversification strategy is available via
`no_improvement_limit`: if that many consecutive iterations pass without
improving the global best, the current solution is replaced with a fresh
random solution and the tabu list is cleared. It is disabled (`None`) by
default.

## 2. Initialization

- If `initial_solution` is supplied, it is validated and used as the
  starting solution.
- Otherwise, a random valid phase-index sequence is generated.
- The starting solution is evaluated once before the first iteration.

## 3. Stopping criteria

The search stops when any of the following occurs first:

- `max_iterations` iterations have been completed (status `completed`).
- The configured `time_limit` is exceeded, checked both before an iteration
  starts and immediately after a move is made (status `time_limit_reached`).

## 4. Computational complexity

Each iteration evaluates the full neighborhood (`N * (Q - 1)` candidates) or
`neighborhood_size` candidates if sampling is enabled, and each evaluation
costs `O(N^2)` in the worst case. Total cost is approximately

$$
O\big(\text{max\_iterations} \times \min(N(Q-1), \text{neighborhood\_size}) \times N^2\big).
$$

## 5. Randomization behavior

- Uses a local `random.Random(seed)` generator for the initial random
  solution (if applicable), neighborhood sampling, and diversification
  restarts.
- No global random state is touched.
- Given the same `seed` and parameters, the run is fully reproducible.

## 6. Strengths and weaknesses

**Strengths**
- Memory of recent moves (the tabu list) prevents immediate cycling, which
  simple local search cannot do on its own.
- The aspiration criterion keeps the tabu mechanism from blocking genuinely
  good moves.
- Best-improvement selection makes efficient use of every evaluated
  candidate.

**Weaknesses**
- Full neighborhood evaluation is expensive for large `N` or `Q` unless
  `neighborhood_size` sampling is used, which in turn trades solution
  quality for speed.
- Performance is sensitive to `tabu_tenure`: too short allows cycling, too
  long can forbid useful moves.
- Without diversification enabled, the search can still stagnate in a
  region if every non-tabu move is unattractive.

## 7. Algorithm-specific parameters

| Parameter | Meaning | Default |
|---|---|---|
| `tabu_tenure` | Number of iterations a reverse move stays tabu | `10` |
| `neighborhood_size` | Optional sample size per iteration; `None` uses the full neighborhood | `None` |
| `max_iterations` | Maximum number of iterations | `200` |
| `no_improvement_limit` | Optional diversification trigger (iterations without improvement) | `None` |
| `use_aspiration` | Whether the aspiration criterion can override tabu status | `True` |
| `time_limit` | Optional runtime limit in seconds | `None` |
| `seed` | Random seed for reproducibility | `None` |
| `initial_solution` | Optional starting phase-index sequence | `None` |

## 8. Result metadata

In addition to the common `SolverResult` fields, `metadata` reports
`tabu_tenure`, `neighborhood_size` (resolved to the full neighborhood size
when not overridden), `max_iterations`, `iterations_completed`,
`no_improvement_limit`, and `use_aspiration`.

## 9. Example usage

```python
from waveform_optimization.solvers.strong_baselines.tabu_search import TabuSearchSolver

solver = TabuSearchSolver()
result = solver.solve(N=31, Q=4, seed=42, tabu_tenure=15, max_iterations=300)

print(result.best_objective)
print(result.best_phase_indices)
```
