import importlib.util
import json
from pathlib import Path

import numpy as np

from waveform_optimization.core.binary_objective import BinaryObjectiveState
from waveform_optimization.solvers.alns.solver import ALNSConfig, ALNSTelemetry, ALNSSolver
from waveform_optimization.solvers.alns.repair import randomized_greedy_repair


SCRIPT = Path(__file__).parents[1] / "experiments" / "scripts" / "run_alns_convergence_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_alns_convergence_experiment", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_checkpoint_uses_pre_iteration_snapshot():
    telemetry = ALNSTelemetry(checkpoint_seconds=(5.0,))
    telemetry.initial({
        "iteration": 0, "best_energy": 30, "current_energy": 30,
        "iterations_completed": 0, "accepted_moves": 0,
        "rejected_moves": 0, "improving_moves": 0, "new_best_moves": 0,
    })
    telemetry.pre_iteration_checkpoint({
        "iteration": 3, "best_energy": 20, "current_energy": 25,
        "iterations_completed": 3, "accepted_moves": 2,
        "rejected_moves": 1, "improving_moves": 1, "new_best_moves": 1,
    }, 5.4)

    assert telemetry.checkpoints[0]["checkpoint_seconds"] == 0.0
    assert telemetry.checkpoints[0]["best_energy"] == 30
    assert telemetry.checkpoints[1]["best_energy"] == 20
    assert telemetry.checkpoints[1]["current_energy"] == 25
    assert telemetry.checkpoints[1]["iterations_completed"] == 3
    assert telemetry.checkpoints[1]["actual_observation_time_seconds"] == 5.4


def test_new_best_events_are_strict_and_include_sequence():
    telemetry = ALNSTelemetry()
    result = ALNSSolver(ALNSConfig(
        time_limit=None, max_iterations=2, seed=42,
        destroy_methods=("random",), repair_methods=("randomized_greedy",),
        acceptance_method="simulated_annealing",
    )).solve(20, initial_state=BinaryObjectiveState([0] * 20), telemetry=telemetry)

    assert result.telemetry is telemetry
    assert all(event["new_best_energy"] < event["previous_best_energy"]
               for event in telemetry.new_best_events)
    assert all(len(event["best_sequence"]) == 20 for event in telemetry.new_best_events)


def test_movement_counters_track_unchanged_candidate(monkeypatch):
    import waveform_optimization.solvers.alns.solver as solver_module

    monkeypatch.setattr(
        solver_module, "apply_destroy",
        lambda state, **kwargs: np.array([], dtype=np.int64),
    )
    monkeypatch.setattr(
        solver_module, "apply_repair",
        lambda state, **kwargs: state.copy(),
    )
    telemetry = ALNSTelemetry()
    ALNSSolver(ALNSConfig(
        time_limit=None, max_iterations=1, seed=42,
        destroy_methods=("random",), repair_methods=("randomized_greedy",),
    )).solve(20, initial_state=BinaryObjectiveState([0] * 20), telemetry=telemetry)

    assert telemetry.counters["unchanged_candidate_count"] == 1
    assert telemetry.counters["equal_energy_changed_sequence_count"] == 0


def test_instrumented_repair_does_not_restore_reference_state():
    reference = BinaryObjectiveState([0] * 20)
    destroyed = reference.copy()
    destroyed.apply_flip(3)
    repaired = randomized_greedy_repair(
        destroyed,
        [3],
        np.random.default_rng(42),
        reference_state=reference,
    )

    assert repaired.sequence[3] == 1
    assert repaired.N == 20
    assert repaired.recompute_energy() == repaired.energy


def test_cached_initial_state_is_reused(tmp_path, monkeypatch):
    calls = []
    original = runner.generate_initial_solution

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "generate_initial_solution", counted)
    first = runner.cache_initial_state(tmp_path, 20, 42, 0.01)
    second = runner.cache_initial_state(tmp_path, 20, 42, 0.01)

    assert len(calls) == 1
    assert first["original_sequence"] == second["original_sequence"]
    assert json.loads((tmp_path / "initial_states" / "N_20_seed_42.json").read_text())["N"] == 20
