# Consolidation record and reproducibility notes

## Input and strategy

Source: team-provided `Vernex-complete.bundle` dated 10 November 2026. The bundle was successfully verified as complete and contained branches `main`, `team-member-1-baseline-solvers`, `strong-baseline-solvers`, `surya-matlab-validation`, `alns`, and `fix/alns-convergence-telemetry` (some as remote-tracking refs).

Created `submission/consolidated` **without rewriting the original commit history**, starting from `fix/alns-convergence-telemetry`, merging the MATLAB/strong-baseline branch, and merging the ALNS/Gurobi branch. The parent graph retains the contributions from the other branches. All merged source versions are recoverable from the original commits.

## Conflict resolution

- `.gitignore`: combined sensible ignores for Python, environments and generated local files.
- `src/waveform_optimization/solvers/alns/solver.py` and `repair.py`: retained **telemetry-branch** implementation; the separate ALNS adapter and Gurobi work from the `alns` branch were added.
- `src/waveform_optimization/solvers/weak_baselines/greedy.py`, its tests and `docs/greedy.md`: retained the more developed `team-member-1-baseline-solvers` / telemetry lineage.
- `src/waveform_optimization/solvers/alns/run_alns_pair_experiment.py`: the original ALNS branch's experiment script was retained alongside the experiment runner from the telemetry branch. They are historical variants, not automatically verified to produce identical results.
- Gurobi logs/results are left at original paths to preserve their connection with existing historical documentation.

**Important ALNS caveat:** The telemetry branch materializes `destroy` as bit flips before repair **when telemetry is enabled**. Therefore telemetry on/off can affect the search trajectory, not just logging. Don't treat convergence results from instrumented and uninstrumented runs as a controlled comparison without resolving this difference in a subsequent review.

## Validation performed on consolidation

- Parsed and corrected whitespace in `experiments/configs/benchmark.yaml` (literal tabs had made the file invalid YAML for PyYAML).
- Updated basic Python dependency list.
- Executed the Python test suite using `PYTHONPATH=src python -m pytest -q`: **128 passed**.
- Performed a brief ALNS adapter smoke check on `N=5`, `11`, and `20` (`Q=2`), and confirmed the reported energies match `islr_from_phase_indices`. These are smoke checks, **not performance benchmarks**.

## Unverified / external requirements

- MATLAB tests and full MATLAB/Python cross-language benchmark scripts were **not run** during consolidation; they require a MATLAB environment.
- Gurobi exact solver was **not executed** during consolidation; `gurobipy` and an appropriate licence are not available in this environment. Historical Gurobi CSV, logs, plots and best sequences are preserved as provided.
- The later guided-escape, local-minimum, Legendre and landscape experiment artifacts discussed outside this Git repository are **not included in the supplied bundle**. Those experiments should be added through a separate reviewed contribution.
- Quantum solvers or measured quantum improvement for LABS are not present in this input.
- No repository licence or claim of permission to make private code public has been inferred.

## Next actions for the team

1. Review the consolidation diff and historical branch merges; pay special attention to the telemetry/no-telemetry ALNS semantic difference.
2. Provide any later experiment notebooks, raw CSVs, figures and scripts that are intended to support the dossier and are not in this Git bundle.
3. Reproduce representative results in a fresh environment, recording machine, run budget, seeds and version tag.
4. Confirm external dependency, data and licence/IP checks before publishing.
5. Tag a reviewed immutable snapshot for inclusion in the dossier's GitHub reference.
