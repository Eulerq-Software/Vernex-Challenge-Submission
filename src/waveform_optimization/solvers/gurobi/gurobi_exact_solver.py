#!/usr/bin/env python3
"""
Surya's deliverable: exact Q=2 binary-phase MIQP via Gurobi.

Objective (matches waveform_optimization's binary energy, Q=2 case):

    x_i = 1 - 2*b_i,          b_i in {0,1},  i = 0..N-1
    z_ij = b_i XOR b_j        (linearized with 4 inequalities)
    C_k  = (N-k) - 2 * sum_i z_{i, i+k},     k = 1..N-1
    minimize  sum_{k=1}^{N-1} C_k^2

C_k is exactly the integer aperiodic autocorrelation at lag k of the
+-1 sequence x (C_k = sum_i x_i * x_{i+k}); the objective is the
integrated sidelobe level (ISL), i.e. the same "energy" the ALNS
solvers minimize. Because C_k is affine in the binary z variables,
sum(C_k^2) is a convex quadratic in binary variables -> a standard
convex MIQP that Gurobi solves natively (no PSD tricks needed).

This script produces the three CSVs the brief asks for:
  - <root>/gurobi_runs.csv        (section 4, run-level CSV)
  - <root>/gurobi_checkpoints.csv (section 4, progress/checkpoint log)
  - <root>/gurobi_incumbents.csv  (every new incumbent + bound update)
plus one saved best sequence per N under <root>/sequences/.

Usage
-----
    python gurobi_exact_solver.py --sizes 5,7,11,15,20,25,30,40,50 \
        --output-root results/gurobi_exact --owner surya

    # include the two optional scale-up sizes (70, 100):
    python gurobi_exact_solver.py --sizes 5,7,11,15,20,25,30,40,50,70,100 ...

Requires a licensed `gurobipy` on the machine you run this on; the
.lic file must stay outside the repo/artifacts per the brief.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import socket
import subprocess
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Fixed experiment design (from the team brief, section 4 table)
# ============================================================

# N -> (time_limit_seconds, role)
SIZE_TABLE: dict[int, tuple[float, str]] = {
    5:   (30.0,   "validation"),
    7:   (30.0,   "validation"),
    11:  (60.0,   "validation"),
    15:  (120.0,  "scale-up"),
    20:  (300.0,  "scale-up"),
    25:  (600.0,  "bounds"),
    30:  (900.0,  "bounds"),
    40:  (1200.0, "bounds"),
    50:  (1800.0, "bounds"),
    70:  (1200.0, "optional"),
    100: (1800.0, "optional"),
}

CORE_SIZES = (5, 7, 11, 15, 20, 25, 30, 40, 50)   # 5,040 s
OPTIONAL_SIZES = (70, 100)                        # +3,000 s

# Known optimal energies for validation (Barker-code lengths).
KNOWN_OPTIMA = {5: 2, 7: 3, 11: 5}

CHECKPOINTS = (0, 10, 30, 60, 120, 300, 600, 900, 1200, 1800)

RUN_FIELDNAMES = (
    "N", "Q", "git_commit", "machine", "cpu", "ram_gb", "gurobi_version",
    "license_test_result", "solver_parameters_json", "model_build_seconds",
    "optimize_seconds", "total_wall_seconds", "model_variables",
    "binary_variables", "constraints", "status", "sol_count",
    "best_feasible_energy", "best_proven_lower_bound", "relative_mip_gap",
    "best_sequence", "first_incumbent_time", "final_best_incumbent_time",
    "time_to_certified_optimality", "node_count", "barker_reference_ratio",
    "objective_reference_validation", "error_or_termination_reason",
    "symmetry_break_b0", "time_limit_seconds", "role",
)

CHECKPOINT_FIELDNAMES = (
    "N", "checkpoint_seconds", "incumbent_objective", "best_bound",
    "has_incumbent", "elapsed_solver_seconds",
)

INCUMBENT_FIELDNAMES = (
    "N", "event_type", "elapsed_seconds", "objective", "bound", "gap",
)


# ============================================================
# Reference (independent) energy computation, for cross-checks
# ============================================================

def energy_from_bits(bits: list[int]) -> int:
    """
    Direct ISL computation from a 0/1 phase-index sequence, used to
    cross-validate the Gurobi objective. Mirrors the project's
    islr_from_phase_indices(q=2) definition: x_i = 1 - 2*b_i,
    C_k = sum_i x_i * x_{i+k}, energy = sum_k C_k^2.
    """
    N = len(bits)
    x = [1 - 2 * b for b in bits]
    energy = 0
    for k in range(1, N):
        c_k = sum(x[i] * x[i + k] for i in range(N - k))
        energy += c_k * c_k
    return energy


def brute_force_optimum(N: int) -> tuple[int, list[int]]:
    """Exhaustive search, only meant for small N (<= ~15) sanity checks."""
    best_e, best_bits = None, None
    for mask in range(2 ** N):
        bits = [(mask >> i) & 1 for i in range(N)]
        if bits[0] != 0:  # exploit sign symmetry to halve the search
            continue
        e = energy_from_bits(bits)
        if best_e is None or e < best_e:
            best_e, best_bits = e, bits
    return best_e, best_bits


# ============================================================
# Model construction
# ============================================================

def build_model(N: int, *, break_symmetry: bool = True) -> tuple[gp.Model, dict]:
    """
    Build the exact convex MIQP for one sequence length N, Q=2.

    Returns (model, info) where info carries variable handles needed
    to extract the best sequence after solve.
    """
    m = gp.Model(f"binary_isl_N{N}")

    b = m.addVars(N, vtype=GRB.BINARY, name="b")

    if break_symmetry:
        # Global sign symmetry: x -> -x leaves every C_k unchanged, so
        # fixing b_0 = 0 removes the mirror duplicate without changing
        # the optimal objective value.
        b[0].lb = 0
        b[0].ub = 0

    # z[i,k] = b_i XOR b_{i+k} for every pair (i, i+k) that appears in
    # some lag-k autocorrelation term.
    z = {}
    for k in range(1, N):
        for i in range(N - k):
            z[i, k] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{i+k}")

    m.update()

    # Four standard XOR-linearization inequalities per pair.
    for (i, k), zij in z.items():
        j = i + k
        m.addConstr(zij >= b[i] - b[j])
        m.addConstr(zij >= b[j] - b[i])
        m.addConstr(zij <= b[i] + b[j])
        m.addConstr(zij <= 2 - b[i] - b[j])

    # C_k = (N-k) - 2 * sum_i z[i,k]; objective = sum_k C_k^2.
    obj = gp.QuadExpr()
    c_exprs = {}
    for k in range(1, N):
        c_k = (N - k) - 2 * gp.quicksum(z[i, k] for i in range(N - k))
        c_exprs[k] = c_k
        obj += c_k * c_k

    m.setObjective(obj, GRB.MINIMIZE)

    info = {
        "b": b,
        "z": z,
        "c_exprs": c_exprs,
        "n_vars": N + len(z),
        "n_binary": N + len(z),
        "n_constraints": 4 * len(z),
    }
    return m, info


# ============================================================
# Checkpoint / incumbent callback
# ============================================================

class CheckpointCallback:
    """
    Polls elapsed solver time; whenever a checkpoint from CHECKPOINTS
    is crossed, records the *last known* incumbent/bound at or before
    that checkpoint (never a future value). Also records every new
    incumbent and every materially improved bound as an event row.
    """

    def __init__(self, N: int, checkpoints=CHECKPOINTS):
        self.N = N
        self.pending = list(checkpoints)
        self.checkpoint_rows: list[dict] = []
        self.incumbent_rows: list[dict] = []
        self._last_obj = None
        self._last_bound = None
        self.first_incumbent_time = None
        self.final_best_incumbent_time = None

    def __call__(self, model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            t = model.cbGet(GRB.Callback.RUNTIME)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if self._last_obj is None or obj < self._last_obj - 1e-9:
                if self.first_incumbent_time is None:
                    self.first_incumbent_time = t
                self.final_best_incumbent_time = t
                self.incumbent_rows.append({
                    "N": self.N, "event_type": "new_incumbent",
                    "elapsed_seconds": t, "objective": obj, "bound": bound,
                    "gap": (obj - bound) / max(abs(obj), 1e-9),
                })
            self._last_obj = obj
            self._last_bound = bound
            self._advance_checkpoints(t)

        elif where == GRB.Callback.MIP:
            t = model.cbGet(GRB.Callback.RUNTIME)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            has_sol = model.cbGet(GRB.Callback.MIP_SOLCNT) > 0
            obj = model.cbGet(GRB.Callback.MIP_OBJBST) if has_sol else None
            if self._last_bound is None or (
                bound is not None and abs(bound - self._last_bound) > 1e-6
            ):
                self.incumbent_rows.append({
                    "N": self.N, "event_type": "bound_update",
                    "elapsed_seconds": t, "objective": obj, "bound": bound,
                    "gap": ((obj - bound) / max(abs(obj), 1e-9)
                            if obj is not None else ""),
                })
            self._last_bound = bound
            if obj is not None:
                self._last_obj = obj
            self._advance_checkpoints(t)

    def _advance_checkpoints(self, t: float) -> None:
        while self.pending and t >= self.pending[0]:
            cp = self.pending.pop(0)
            self.checkpoint_rows.append({
                "N": self.N,
                "checkpoint_seconds": cp,
                "incumbent_objective": self._last_obj if self._last_obj is not None else "",
                "best_bound": self._last_bound if self._last_bound is not None else "",
                "has_incumbent": self._last_obj is not None,
                "elapsed_solver_seconds": t,
            })

    def flush_final(self, t_final: float) -> None:
        """Emit a final row for any checkpoints the run never reached, plus
        the mandatory 'final row at actual completion'."""
        while self.pending:
            cp = self.pending.pop(0)
            self.checkpoint_rows.append({
                "N": self.N, "checkpoint_seconds": cp,
                "incumbent_objective": self._last_obj if self._last_obj is not None else "",
                "best_bound": self._last_bound if self._last_bound is not None else "",
                "has_incumbent": self._last_obj is not None,
                "elapsed_solver_seconds": t_final,
            })
        self.checkpoint_rows.append({
            "N": self.N, "checkpoint_seconds": "final",
            "incumbent_objective": self._last_obj if self._last_obj is not None else "",
            "best_bound": self._last_bound if self._last_bound is not None else "",
            "has_incumbent": self._last_obj is not None,
            "elapsed_solver_seconds": t_final,
        })


# ============================================================
# Environment / provenance
# ============================================================

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def machine_info() -> dict:
    return {
        "machine": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": "",  # fill in manually if psutil isn't available
    }


# ============================================================
# Run one size
# ============================================================

def run_one(
    N: int,
    *,
    time_limit: float,
    role: str,
    threads: int | None,
    seed: int,
    mip_gap: float,
    break_symmetry: bool,
) -> tuple[dict, list[dict], list[dict]]:

    build_start = time.perf_counter()
    model, info = build_model(N, break_symmetry=break_symmetry)
    build_seconds = time.perf_counter() - build_start

    params = {
        "TimeLimit": time_limit,
        "MIPGap": mip_gap,
        "Seed": seed,
    }
    if threads is not None:
        params["Threads"] = threads

    for k, v in params.items():
        model.setParam(k, v)
    model.setParam("LogFile", f"gurobi_N{N}.log")

    cb = CheckpointCallback(N)

    optimize_start = time.perf_counter()
    error = ""
    try:
        model.optimize(cb)
    except gp.GurobiError as exc:
        error = f"GurobiError: {exc}"
    optimize_seconds = time.perf_counter() - optimize_start
    cb.flush_final(model.Runtime if error == "" else optimize_seconds)

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    status = status_map.get(model.Status, f"STATUS_{model.Status}") if error == "" else "ERROR"

    sol_count = model.SolCount if error == "" else 0
    best_feasible = model.ObjVal if (error == "" and sol_count > 0) else None
    best_bound = model.ObjBound if error == "" else None
    rel_gap = model.MIPGap if (error == "" and sol_count > 0) else None

    best_sequence = ""
    ref_validation = ""
    if error == "" and sol_count > 0:
        bits = [int(round(info["b"][i].X)) for i in range(N)]
        best_sequence = "".join(str(bit) for bit in bits)
        ref_energy = energy_from_bits(bits)
        ref_validation = "match" if math.isclose(
            ref_energy, best_feasible, abs_tol=1e-6
        ) else f"MISMATCH reference={ref_energy} gurobi={best_feasible}"

    barker_ratio = (
        best_feasible / ((N - 1) / 2) if best_feasible is not None else ""
    )

    minfo = machine_info()
    row = {
        "N": N, "Q": 2, "git_commit": git_commit(),
        "machine": minfo["machine"], "cpu": minfo["cpu"], "ram_gb": minfo["ram_gb"],
        "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        "license_test_result": "ok" if error == "" else "error",
        "solver_parameters_json": json.dumps(params),
        "model_build_seconds": build_seconds,
        "optimize_seconds": optimize_seconds,
        "total_wall_seconds": build_seconds + optimize_seconds,
        "model_variables": info["n_vars"],
        "binary_variables": info["n_binary"],
        "constraints": info["n_constraints"],
        "status": status,
        "sol_count": sol_count,
        "best_feasible_energy": best_feasible if best_feasible is not None else "NA",
        "best_proven_lower_bound": best_bound if best_bound is not None else "NA",
        "relative_mip_gap": rel_gap if rel_gap is not None else "NA",
        "best_sequence": best_sequence,
        "first_incumbent_time": cb.first_incumbent_time if cb.first_incumbent_time is not None else "NA",
        "final_best_incumbent_time": cb.final_best_incumbent_time if cb.final_best_incumbent_time is not None else "NA",
        "time_to_certified_optimality": (
            model.Runtime if (error == "" and status == "OPTIMAL") else "NA"
        ),
        "node_count": model.NodeCount if error == "" else "NA",
        "barker_reference_ratio": barker_ratio,
        "objective_reference_validation": ref_validation,
        "error_or_termination_reason": error,
        "symmetry_break_b0": break_symmetry,
        "time_limit_seconds": time_limit,
        "role": role,
    }

    if N in KNOWN_OPTIMA and best_feasible is not None:
        expected = KNOWN_OPTIMA[N]
        if status == "OPTIMAL" and not math.isclose(best_feasible, expected, abs_tol=1e-6):
            row["error_or_termination_reason"] += (
                f" | VALIDATION FAILED: expected {expected}, got {best_feasible}"
            )

    return row, cb.checkpoint_rows, cb.incumbent_rows


# ============================================================
# CSV helpers
# ============================================================

def append_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/gurobi_exact"))
    parser.add_argument("--owner", default="surya")
    parser.add_argument(
        "--sizes", default=",".join(map(str, CORE_SIZES)),
        help="Comma-separated N values; must be keys of SIZE_TABLE.",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--no-symmetry-break", action="store_true")
    parser.add_argument(
        "--brute-force-check", action="store_true",
        help="Also run an exhaustive search for N<=15 to cross-check Gurobi.",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    unknown = [n for n in sizes if n not in SIZE_TABLE]
    if unknown:
        raise ValueError(f"No time-limit entry for N={unknown}; add to SIZE_TABLE.")

    total_budget = sum(SIZE_TABLE[n][0] for n in sizes)
    print(f"Planned sizes: {sizes}")
    print(f"Total solver-time budget: {total_budget:.0f} s "
          f"({total_budget/60:.1f} min)")

    owner_dir = args.output_root / args.owner
    run_csv = owner_dir / "gurobi_runs.csv"
    checkpoint_csv = owner_dir / "gurobi_checkpoints.csv"
    incumbent_csv = owner_dir / "gurobi_incumbents.csv"
    seq_dir = owner_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    for N in sizes:
        time_limit, role = SIZE_TABLE[N]
        print(f"\n=== N={N}  time_limit={time_limit}s  role={role} ===")

        if args.brute_force_check and N <= 15:
            bf_energy, bf_bits = brute_force_optimum(N)
            print(f"  brute-force optimum: energy={bf_energy} "
                  f"bits={''.join(map(str, bf_bits))}")

        row, checkpoints, incumbents = run_one(
            N,
            time_limit=time_limit,
            role=role,
            threads=args.threads,
            seed=args.seed,
            mip_gap=args.mip_gap,
            break_symmetry=not args.no_symmetry_break,
        )

        append_rows(run_csv, RUN_FIELDNAMES, [row])
        append_rows(checkpoint_csv, CHECKPOINT_FIELDNAMES, checkpoints)
        append_rows(incumbent_csv, INCUMBENT_FIELDNAMES, incumbents)

        if row["best_sequence"]:
            (seq_dir / f"N{N}_best.txt").write_text(row["best_sequence"] + "\n")

        print(f"  status={row['status']} best_energy={row['best_feasible_energy']} "
              f"bound={row['best_proven_lower_bound']} gap={row['relative_mip_gap']} "
              f"validation={row['objective_reference_validation']}")

    print(f"\nWrote:\n  {run_csv}\n  {checkpoint_csv}\n  {incumbent_csv}\n  {seq_dir}/N*_best.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
