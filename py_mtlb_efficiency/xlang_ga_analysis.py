#!/usr/bin/env python3
"""
Python vs MATLAB genetic_algorithm comparison report, with parameter-sweep
support.

Read-only: this reads a <host>.csv results file produced by
xlang_benchmark.sh and prints/writes a summary. It never touches the
append-only raw files.

A plain run (no --variants-dir sweep) writes rows under solver key
"genetic_algorithm". A swept run (xlang_benchmark.sh --variants-dir ...)
writes rows under "genetic_algorithm__<variant>" for every *.json variant
file found. This script auto-discovers whichever of those solver keys are
present in the CSV and reports each variant separately, plus an overall
tally across all variants.

For each (variant, N), it looks at the LAST recorded row for python and
matlab with the given seed, and reports:
  - which language found the lower (better) objective
  - which language ran faster
  - the gap to the exact optimum, when an `enumeration` row exists for the
    same (N, Q, seed) in the same file (enumeration is only feasible for
    small N; larger N is reported as "no exact optimum known")

Usage:
    python3 xlang_ga_analysis.py --csv results/cross_language/su-MS-7E70.csv
    python3 xlang_ga_analysis.py --csv ... --q 2 --seed 42 --out summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

RESULT_COLUMNS = [
    "timestamp",
    "solver",
    "N",
    "Q",
    "seed",
    "language",
    "status",
    "best_objective",
    "recomputed_objective",
    "objective_evaluations",
    "runtime_seconds",
    "wall_seconds",
    "valid",
    "best_phase_indices",
    "error",
]


def read_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("timestamp,"):  # header
                continue
            row = next(csv.reader([line]))
            if len(row) < len(RESULT_COLUMNS):
                continue
            rows.append(dict(zip(RESULT_COLUMNS, row)))
    return rows


def discover_variants(rows: list[dict], prefix: str) -> list[str]:
    """
    Return variant labels found in the CSV for this solver prefix, in a
    stable order: bare "<prefix>" first (as "", meaning "no variant") if
    present, then "<prefix>__X" variants alphabetically by X.
    """
    labels: set[str] = set()
    has_bare = False
    for r in rows:
        s = r["solver"]
        if s == prefix:
            has_bare = True
        elif s.startswith(prefix + "__"):
            labels.add(s[len(prefix) + 2 :])
    ordered = ([""] if has_bare else []) + sorted(labels)
    return ordered


def last_matching(rows: list[dict], solver: str, N: str, Q: str, seed: str, language: str):
    found = None
    for r in rows:
        if (
            r["solver"] == solver
            and r["language"] == language
            and r["N"] == str(N)
            and r["Q"] == str(Q)
            and r["seed"] == seed
        ):
            found = r
    return found


def fmt(v, width, prec=6):
    return f"{v:>{width}.{prec}f}" if v is not None else f"{'--':>{width}}"


def run_one_variant(
    rows: list[dict],
    solver_key: str,
    variant_label: str,
    args: argparse.Namespace,
    seed_field: str,
) -> list[dict]:
    summary_rows = []
    py_wins = ml_wins = ties = 0
    py_faster = ml_faster = 0

    title = f"variant: {variant_label or '(default / no sweep)'}  [solver key: {solver_key}]"
    print(f"\n=== {title} ===")
    header = (
        f"{'N':>3} | {'F_python':>14} | {'F_matlab':>14} | {'winner':>7} | "
        f"{'t_python(s)':>12} | {'t_matlab(s)':>12} | {'faster':>7} | {'gap_py':>10} | {'gap_ml':>10}"
    )
    print(header)
    print("-" * len(header))

    for N in range(args.n_min, args.n_max + 1):
        py = last_matching(rows, solver_key, N, args.q, seed_field, "python")
        ml = last_matching(rows, solver_key, N, args.q, seed_field, "matlab")
        enum = last_matching(rows, "enumeration", N, args.q, seed_field, "python")  # ground truth

        if py is None and ml is None:
            continue

        f_py = float(py["best_objective"]) if py else None
        f_ml = float(ml["best_objective"]) if ml else None
        t_py = float(py["runtime_seconds"]) if py else None
        t_ml = float(ml["runtime_seconds"]) if ml else None
        f_opt = float(enum["best_objective"]) if enum else None

        if f_py is None or f_ml is None:
            winner = "n/a"
        elif abs(f_py - f_ml) <= args.tol:
            winner = "tie"
            ties += 1
        elif f_py < f_ml:
            winner = "python"
            py_wins += 1
        else:
            winner = "matlab"
            ml_wins += 1

        if t_py is None or t_ml is None:
            faster = "n/a"
        elif t_py < t_ml:
            faster = "python"
            py_faster += 1
        else:
            faster = "matlab"
            ml_faster += 1

        gap_py = f_py - f_opt if (f_py is not None and f_opt is not None) else None
        gap_ml = f_ml - f_opt if (f_ml is not None and f_opt is not None) else None

        print(
            f"{N:>3} | {fmt(f_py, 14)} | {fmt(f_ml, 14)} | {winner:>7} | "
            f"{fmt(t_py, 12, 4)} | {fmt(t_ml, 12, 4)} | {faster:>7} | "
            f"{fmt(gap_py, 10)} | {fmt(gap_ml, 10)}"
        )

        summary_rows.append(
            {
                "params_variant": variant_label or "default",
                "solver_key": solver_key,
                "N": N,
                "Q": args.q,
                "seed": seed_field,
                "objective_python": f_py,
                "objective_matlab": f_ml,
                "winner_objective": winner,
                "runtime_python": t_py,
                "runtime_matlab": t_ml,
                "faster": faster,
                "enumeration_optimum": f_opt,
                "gap_python": gap_py,
                "gap_matlab": gap_ml,
            }
        )

    print("-" * len(header))
    print(
        f"objective wins -> python: {py_wins}, matlab: {ml_wins}, tie: {ties}  |  "
        f"speed wins -> python: {py_faster}, matlab: {ml_faster}"
    )
    return summary_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to <host>.csv (the raw results file)")
    ap.add_argument("--solver-prefix", default="genetic_algorithm", help="base solver key (default genetic_algorithm)")
    ap.add_argument("--q", default="2", help="Q value to filter on (default 2)")
    ap.add_argument("--seed", default="none", help="seed used for the GA runs (default: none -> empty seed field)")
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1e-6, help="objective difference below this counts as a tie")
    ap.add_argument("--out", default=None, help="optional path to also write the combined summary as CSV")
    args = ap.parse_args()

    seed_field = "" if args.seed.lower() in {"none", "", "null"} else args.seed
    rows = read_rows(args.csv)

    variants = discover_variants(rows, args.solver_prefix)
    if not variants:
        print(f"no rows found for solver '{args.solver_prefix}' (or its variants) in {args.csv}", file=sys.stderr)
        return 1

    all_summary_rows: list[dict] = []
    for variant_label in variants:
        solver_key = args.solver_prefix if not variant_label else f"{args.solver_prefix}__{variant_label}"
        all_summary_rows.extend(run_one_variant(rows, solver_key, variant_label, args, seed_field))

    if len(variants) > 1:
        print("\n=== overall (all variants combined) ===")
        py_wins = sum(1 for r in all_summary_rows if r["winner_objective"] == "python")
        ml_wins = sum(1 for r in all_summary_rows if r["winner_objective"] == "matlab")
        ties = sum(1 for r in all_summary_rows if r["winner_objective"] == "tie")
        py_faster = sum(1 for r in all_summary_rows if r["faster"] == "python")
        ml_faster = sum(1 for r in all_summary_rows if r["faster"] == "matlab")
        print(
            f"objective wins -> python: {py_wins}, matlab: {ml_wins}, tie: {ties}  |  "
            f"speed wins -> python: {py_faster}, matlab: {ml_faster}"
        )
        print(
            "(gap_py / gap_ml = distance above the exhaustive-enumeration optimum, "
            "'--' where no enumeration ground truth exists for that N)"
        )

    if args.out and all_summary_rows:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)
        print(f"\nwritten: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())