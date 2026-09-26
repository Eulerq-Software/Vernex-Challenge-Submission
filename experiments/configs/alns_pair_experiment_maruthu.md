# Maruthu ALNS Pair Experiment

The runner uses the team brief's controlled binary-phase experiment settings:

- pairs: `random/exact`, `block/exact`, `best_flip/exact`, `correlation_guided/exact`, `random/greedy`
- sizes: 5, 7, 11, 20, 30, 50, 100, 150, 200, 300, 500, 1000
- seeds: 42, 123, 456, 789, 2026
- outer acceptance: simulated annealing
- destroy fraction: 0.10
- exact neighbourhood cap: 10

Run the exact-repair preflight first:

```bash
PYTHONPATH=src python3 experiments/scripts/run_alns_pair_experiment.py \
  --output-root results/alns_pair_experiments_preflight \
  --sizes 100,500 --seeds 42 --pairs random/exact
```

Then run the assigned 300-run slice:

```bash
PYTHONPATH=src python3 experiments/scripts/run_alns_pair_experiment.py \
  --output-root results/alns_pair_experiments --owner maruthu
```
