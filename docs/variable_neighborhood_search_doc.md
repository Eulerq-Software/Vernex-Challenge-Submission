# Variable Neighborhood Search

## 1. Algorithmic procedure

Variable Neighborhood Search (VNS) changes the neighborhood structure used
for perturbation during the search in order to escape local minima. Each
iteration:

1. Selects the current neighborhood level `k` (starting at level 0).
2. Perturbs ("shakes") the incumbent solution using that neighborhood.
3. Applies local search to the perturbed solution.
4. If the locally-searched result improves on the incumbent, accepts it and
   resets to level 0.
5. Otherwise, keeps the incumbent and advances to the next neighborhood
   level, cycling back to level 0 after the last level.

### Neighborhood levels

Three fixed levels are used, in order:

| Level | Neighborhood |
|---|---|
| 0 | One-coordinate change |
| 1 | Two-coordinate change |
| 2 | Contiguous block perturbation (`block_size` coordinates, clamped to `N`) |

At every level, each perturbed coordinate is shifted by a random nonzero
offset modulo `Q`, guaranteeing an actual change at that coordinate.

### Local search

After shaking, a bounded best-improvement coordinate descent is applied: for
each coordinate in turn, every alternative phase value is evaluated and the
best improving value (if any) is adopted. Full sweeps over all coordinates
repeat until a sweep makes no change or `local_search_max_sweeps` sweeps have
been performed. This local search is self-contained within this solver and
does not depend on any other team member's coordinate-descent
implementation.

### Acceptance and neighborhood reset

The locally-searched candidate replaces the incumbent only if it strictly
improves on the incumbent's objective, at which point the neighborhood level
resets to 0. Otherwise the incumbent is unchanged and the level advances. The
best solution ever found is tracked separately from the incumbent, so a
candidate that does not beat the incumbent but beats the global best is still
recorded.

## 2. Initialization

- If `initial_solution` is supplied, it is validated and used as the
  starting incumbent.
- Otherwise, a random valid phase-index sequence is generated.
- The starting incumbent is evaluated once before the first iteration.

## 3. Stopping criteria

The search stops when any of the following occurs first:

- `max_iterations` shake-and-local-search cycles have been completed (status
  `completed`).
- The configured `time_limit` is exceeded, checked before each iteration and
  during local search (status `time_limit_reached`).

## 4. Computational complexity

Each iteration performs one shake evaluation plus a bounded local search. In
the worst case, local search evaluates `N * (Q - 1)` candidates per sweep for
up to `local_search_max_sweeps` sweeps, and each evaluation costs `O(N^2)`.
Total cost per iteration is approximately

$$
O\big(\text{local\_search\_max\_sweeps} \times N \times (Q - 1) \times N^2\big),
$$

multiplied by `max_iterations` unless stopped earlier by `time_limit`.

## 5. Randomization behavior

- Uses a local `random.Random(seed)` generator for the initial random
  solution (if applicable) and for every shake operation.
- No global random state is touched.
- Given the same `seed` and parameters, the run is fully reproducible.

## 6. Strengths and weaknesses

**Strengths**
- Systematically varies perturbation strength, which helps escape local
  minima that a single fixed neighborhood (as in plain local search) cannot.
- Combines diversification (the shake) with intensification (the local
  search) in a single, simple loop.
- Global best is never lost, since it is tracked independently of the
  incumbent's acceptance decisions.

**Weaknesses**
- Local search after every shake is the dominant cost; large
  `local_search_max_sweeps` values make each iteration expensive.
- Only three neighborhood levels are used; problems that need a wider range
  of perturbation strengths may need this extended.
- As with other local-search-based methods, the strict-improvement
  acceptance rule can cause many iterations to make no progress once the
  incumbent is near a strong local optimum.

## 7. Algorithm-specific parameters

| Parameter | Meaning | Default |
|---|---|---|
| `max_iterations` | Maximum number of shake-and-local-search cycles | `100` |
| `block_size` | Coordinates perturbed at neighborhood level 2 (clamped to `N`) | `3` |
| `local_search_max_sweeps` | Cap on full coordinate-descent sweeps per local search call | `10` |
| `time_limit` | Optional runtime limit in seconds | `None` |
| `seed` | Random seed for reproducibility | `None` |
| `initial_solution` | Optional starting phase-index sequence | `None` |

## 8. Result metadata

In addition to the common `SolverResult` fields, `metadata` reports
`max_iterations`, `iterations_completed`, `num_neighborhood_levels` (always
`3`), `block_size` (the clamped value actually used), `local_search_max_sweeps`,
`accepted_shakes` (number of iterations that improved the incumbent), and
`final_neighborhood_level`.

## 9. Example usage

```python
from waveform_optimization.solvers.strong_baselines.variable_neighborhood_search import (
    VariableNeighborhoodSearchSolver,
)

solver = VariableNeighborhoodSearchSolver()
result = solver.solve(N=31, Q=4, seed=42, max_iterations=120)

print(result.best_objective)
print(result.best_phase_indices)
```
