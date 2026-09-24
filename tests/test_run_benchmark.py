import importlib.util
import csv
import json
from pathlib import Path



SCRIPT = Path(__file__).parents[1] / "experiments" / "scripts" / "run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_benchmark", SCRIPT)
run_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_benchmark)


def test_load_config_applies_yaml_values(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "algorithms:\n  - greedy\nN_values:\n  - 3\nQ: 2\nruns: 1\n",
        encoding="utf-8",
    )

    config = run_benchmark.load_config(config_path)

    assert config["algorithms"] == ["greedy"]
    assert config["N_values"] == [3]
    assert config["normalization"]["barker_reference_scale"] == 1.0


def test_official_config_matches_full_schedule():
    config = run_benchmark.load_config(
        Path(__file__).parents[1] / "experiments" / "configs" / "benchmark.yaml"
    )
    expected_N = [5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 100, 150,
                  200, 300, 400, 500, 600, 700, 800, 900, 1000]
    assert config["N_values"] == expected_N
    assert config["Q"] == 2
    assert config["time_limit_seconds_by_N"][1000] == 300
    assert config["runs_by_N"][100] == 10
    assert config["runs_by_N"][150] == 5


def test_smoke_run_writes_json_and_csv(tmp_path):
    records = run_benchmark.run_benchmark(
        run_benchmark.DEFAULT_CONFIG,
        output_root=tmp_path,
        algorithms=["greedy"], N_values=[5], Q=2, runs=1, time_limit=0.01,
        seed=7,
    )

    assert len(records) == 1
    assert records[0]["status"] == "completed"
    output = tmp_path / "raw" / "greedy" / "N_5" / "0.01s" / "run_1.json"
    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["normalized_objective"] == saved["best_objective"] / 2
    assert "Barker reference scale" == saved["normalization_label"]
    csv_path = tmp_path / "binary_baseline_raw_results.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "greedy"
    assert set(rows[0]) == set(run_benchmark.CSV_FIELDS)
    assert "history" not in rows[0]
    assert "best_phase_indices" not in rows[0]


def test_validation_failure_is_saved_as_failed_record(tmp_path, monkeypatch):
    class InvalidSolver:
        def solve(self, **kwargs):
            return {
                "algorithm": "greedy", "N": 5, "Q": 2,
                "best_phase_indices": [0, 0, 0, 0, 0], "best_objective": 999.0,
                "runtime_seconds": 0.0, "objective_evaluations": 1,
                "status": "completed",
            }

    monkeypatch.setitem(run_benchmark.SOLVERS, "greedy", InvalidSolver)
    records = run_benchmark.run_benchmark(
        run_benchmark.DEFAULT_CONFIG, output_root=tmp_path,
        algorithms=["greedy"], N_values=[5], runs=1, time_limit=2,
    )

    assert records[0]["status"] == "validation_failed"
    assert "does not match" in records[0]["validation_error"]
    assert (tmp_path / "raw" / "greedy" / "N_5" / "2s" / "run_1.json").exists()


def test_randomized_runs_get_independent_seeds_without_config_seed(tmp_path):
    config = dict(run_benchmark.DEFAULT_CONFIG)
    config["seed"] = None
    records = run_benchmark.run_benchmark(
        config, output_root=tmp_path, algorithms=["random_search"],
        N_values=[5], Q=2, runs=3, time_limit=0.001,
    )

    seeds = [record["seed"] for record in records]
    assert all(isinstance(value, int) for value in seeds)
    assert len(set(seeds)) == 3


def test_normalization_uses_barker_reference_scale(tmp_path, monkeypatch):
    class KnownObjectiveSolver:
        def solve(self, **kwargs):
            return {
                "algorithm": "greedy", "N": 5, "Q": 2,
                "best_phase_indices": [0, 0, 1, 0, 0], "best_objective": 6.0,
                "runtime_seconds": 0.0, "objective_evaluations": 1,
                "status": "completed",
            }

    monkeypatch.setitem(run_benchmark.SOLVERS, "greedy", KnownObjectiveSolver)
    records = run_benchmark.run_benchmark(
        run_benchmark.DEFAULT_CONFIG, output_root=tmp_path,
        algorithms=["greedy"], N_values=[5], runs=1,
    )

    assert records[0]["normalized_objective"] == 3.0


def test_csv_rerun_replaces_same_budget_identity(tmp_path):
    kwargs = dict(run_benchmark.DEFAULT_CONFIG, algorithms=["greedy"])
    run_benchmark.run_benchmark(kwargs, output_root=tmp_path, algorithms=["greedy"],
                                N_values=[5], runs=1, time_limit=0.01, seed=7)
    run_benchmark.run_benchmark(kwargs, output_root=tmp_path, algorithms=["greedy"],
                                N_values=[5], runs=1, time_limit=0.01, seed=7)

    with (tmp_path / "binary_baseline_raw_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["time_limit_seconds"] == "0.01"