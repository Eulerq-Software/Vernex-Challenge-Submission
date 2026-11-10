# Genetic Algorithm

## 1. Algorithmic procedure

The Genetic Algorithm maintains a population of `population_size` valid
phase-index sequences and evolves it generation by generation through:

1. Ranking the population by objective value.
2. Carrying the best `elite_count` individuals forward unchanged (elitism).
3. Repeatedly selecting two parents via tournament selection, applying
   crossover and mutation, and adding the resulting children to the next
   generation until it reaches `population_size`.
4. Replacing the population with the new generation.

Since the objective is minimized, lower objective values are always treated
as better throughout selection and elitism; no fitness-scaling transform is
applied.

### Selection

Tournament selection: `tournament_size` individuals are drawn at random
(with replacement) from the current population, and the one with the lowest
objective value wins and becomes a parent. Because tournament selection only
compares raw objective values, it does not require a fitness transform such
as `1 / (1 + F(q))`.

### Crossover

Single-point crossover, applied with probability `crossover_rate`. A
crossover point is chosen uniformly in `[1, N-1]`, and the two parents
exchange tails to produce two children. If crossover does not occur, the
children are exact copies of the parents.

### Mutation

Applied independently to each gene of each child: with probability
`mutation_rate`, the phase index at that position is replaced by shifting it
by a random nonzero offset modulo `Q`, guaranteeing an actual change.

### Elitism

The best `elite_count` individuals from the current generation are copied
directly into the next generation without re-evaluation, guaranteeing the
best objective value never regresses across generations.

## 2. Initialization

- If `initial_solution` is supplied, it is validated and inserted as one
  member of the initial population.
- The remaining individuals (and the entire population if no
  `initial_solution` is given) are generated uniformly at random.
- Every individual in the initial population is evaluated once.

## 3. Stopping criteria

The search stops when any of the following occurs first:

- `max_generations` generations have been completed (status `completed`).
- The configured `time_limit` is exceeded, checked both before a generation
  starts and while children are being produced within a generation (status
  `time_limit_reached`).

## 4. Computational complexity

Each generation evaluates `population_size - elite_count` new individuals
(elites are reused without re-evaluation), and each evaluation costs
`O(N^2)` in the worst case. Total cost is approximately

$$
O\big(\text{max\_generations} \times (\text{population\_size} - \text{elite\_count}) \times N^2\big),
$$

unless stopped earlier by `time_limit`.

## 5. Randomization behavior

- Uses a local `random.Random(seed)` generator for population initialization,
  tournament sampling, crossover point selection, and mutation.
- No global random state is touched.
- Given the same `seed` and parameters, the run is fully reproducible.

## 6. Strengths and weaknesses

**Strengths**
- Maintains diversity across a population rather than a single trajectory,
  which helps avoid premature convergence to one local minimum.
- Elitism guarantees monotonic improvement of the best-found solution.
- Naturally parallelizable (not exploited in this implementation).

**Weaknesses**
- More parameters to tune than single-trajectory methods (population size,
  crossover rate, mutation rate, tournament size, elite count).
- Higher per-generation cost, since a whole population must be evaluated.
- Single-point crossover on a fixed-length sequence can disrupt good
  sub-patterns near the crossover point ("positional bias").

## 7. Algorithm-specific parameters

| Parameter | Meaning | Default |
|---|---|---|
| `population_size` | Number of individuals per generation | `50` |
| `max_generations` | Maximum number of generations | `200` |
| `crossover_rate` | Probability that crossover is applied to a parent pair | `0.9` |
| `mutation_rate` | Per-gene mutation probability | `0.05` |
| `tournament_size` | Number of contestants per tournament selection | `3` |
| `elite_count` | Number of top individuals carried over unchanged | `2` |
| `time_limit` | Optional runtime limit in seconds | `None` |
| `seed` | Random seed for reproducibility | `None` |
| `initial_solution` | Optional individual seeded into the initial population | `None` |

## 8. Result metadata

In addition to the common `SolverResult` fields, `metadata` reports
`population_size`, `max_generations`, `generations_completed`,
`crossover_rate`, `mutation_rate`, `tournament_size`, and `elite_count`.

## 9. Example usage

```python
from waveform_optimization.solvers.strong_baselines.genetic_algorithm import (
    GeneticAlgorithmSolver,
)

solver = GeneticAlgorithmSolver()
result = solver.solve(N=31, Q=4, seed=42, population_size=80, max_generations=150)

print(result.best_objective)
print(result.best_phase_indices)
```
