#!/usr/bin/env python3
"""
Turn solver JSON records into CSV rows, and compare the Python and MATLAB
rows for one instance.

Two subcommands, both used by scripts/xlang_benchmark.sh:

    row      <record.json>                  -> one CSV line for the results file
    compare  --csv <results.csv> ...        -> one CSV line for the comparison file

`compare` reads the results CSV rather than the JSON records, so a comparison
can still be produced when one language ran in an earlier session and only the
other one ran now.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
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

COMPARISON_COLUMNS = [
    "timestamp",
    "solver",
    "N",
    "Q",
    "seed",
    "objective_python",
    "objective_matlab",
    "abs_diff",
    "within_tolerance",
    "tolerance",
    "recompute_ok_python",
    "recompute_ok_matlab",
    "valid_python",
    "valid_matlab",
    "indices_identical",
    "evaluations_python",
    "evaluations_matlab",
    "runtime_python",
    "runtime_matlab",
    "runtime_ratio_matlab_over_python",
    "status_python",
    "status_matlab",
]


def _csv_line(fields: list) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(fields)
    return buffer.getvalue()


def _now() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        # MATLAB encodes an absent seed as [], Python encodes it as null.
        return "" if not value else " ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def cmd_row(args: argparse.Namespace) -> int:
    with open(args.json, encoding="utf-8") as handle:
        record = json.load(handle)

    indices = record.get("best_phase_indices") or []
    if not isinstance(indices, list):
        # jsonencode turns a length-1 vector into a bare number.
        indices = [indices]
    fields = [
        _now(),
        args.solver,
        _fmt(record.get("N")),
        _fmt(record.get("Q")),
        _fmt(record.get("seed")),
        record.get("language", args.language),
        _fmt(record.get("status")),
        _fmt(record.get("best_objective")),
        _fmt(record.get("recomputed_objective")),
        _fmt(record.get("objective_evaluations")),
        _fmt(record.get("runtime_seconds")),
        _fmt(record.get("wall_seconds")),
        _fmt(record.get("valid")),
        " ".join(str(int(v)) for v in indices),
        _fmt(record.get("error")),
    ]
    sys.stdout.write(_csv_line(fields) + "\n")
    return 0


def _last_row(csv_path: str, solver: str, N: str, Q: str, seed: str, language: str):
    found = None
    try:
        with open(csv_path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                row = next(csv.reader([line]))
                if len(row) < len(RESULT_COLUMNS):
                    continue
                record = dict(zip(RESULT_COLUMNS, row))
                if record["solver"] != solver or record["language"] != language:
                    continue
                if record["N"] != str(N) or record["Q"] != str(Q):
                    continue
                if record["seed"] != str(seed):
                    continue
                found = record
    except FileNotFoundError:
        return None
    return found


def _as_float(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def cmd_compare(args: argparse.Namespace) -> int:
    seed_field = "" if args.seed.lower() in {"none", "", "null"} else args.seed

    py = _last_row(args.csv, args.solver, args.N, args.Q, seed_field, "python")
    ml = _last_row(args.csv, args.solver, args.N, args.Q, seed_field, "matlab")

    if py is None or ml is None:
        missing = "python" if py is None else "matlab"
        print(f"no {missing} row available for comparison", file=sys.stderr)
        return 2

    f_py = _as_float(py["best_objective"])
    f_ml = _as_float(ml["best_objective"])
    r_py = _as_float(py["recomputed_objective"])
    r_ml = _as_float(ml["recomputed_objective"])
    t_py = _as_float(py["runtime_seconds"])
    t_ml = _as_float(ml["runtime_seconds"])
    tol = float(args.tol)

    abs_diff = None if (f_py is None or f_ml is None) else abs(f_py - f_ml)
    within = None if abs_diff is None else abs_diff < tol
    recompute_ok_py = None if (f_py is None or r_py is None) else abs(f_py - r_py) < tol
    recompute_ok_ml = None if (f_ml is None or r_ml is None) else abs(f_ml - r_ml) < tol
    ratio = None
    if t_py and t_ml is not None and t_py > 0:
        ratio = t_ml / t_py

    fields = [
        _now(),
        args.solver,
        args.N,
        args.Q,
        seed_field,
        _fmt(f_py),
        _fmt(f_ml),
        _fmt(abs_diff),
        _fmt(within),
        _fmt(tol),
        _fmt(recompute_ok_py),
        _fmt(recompute_ok_ml),
        py["valid"],
        ml["valid"],
        _fmt(py["best_phase_indices"] == ml["best_phase_indices"]),
        py["objective_evaluations"],
        ml["objective_evaluations"],
        _fmt(t_py),
        _fmt(t_ml),
        _fmt(ratio),
        py["status"],
        ml["status"],
    ]
    sys.stdout.write(_csv_line(fields) + "\n")

    verdict = "OK " if within else "MISMATCH"
    print(
        f"[{verdict}] F_py={f_py} F_ml={f_ml} |dF|={abs_diff} "
        f"evals {py['objective_evaluations']}/{ml['objective_evaluations']} "
        f"time {t_py:.4f}s / {t_ml:.4f}s"
        if (t_py is not None and t_ml is not None)
        else f"[{verdict}] F_py={f_py} F_ml={f_ml} |dF|={abs_diff}",
        file=sys.stderr,
    )
    return 0 if within else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p_row = sub.add_parser("row")
    p_row.add_argument("json")
    p_row.add_argument("--solver", required=True)
    p_row.add_argument("--language", default="")
    p_row.set_defaults(func=cmd_row)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--csv", required=True)
    p_cmp.add_argument("--solver", required=True)
    p_cmp.add_argument("--N", required=True)
    p_cmp.add_argument("--Q", required=True)
    p_cmp.add_argument("--seed", default="none")
    p_cmp.add_argument("--tol", default="1e-9")
    p_cmp.set_defaults(func=cmd_compare)

    p_hdr = sub.add_parser("columns")
    p_hdr.add_argument("which", choices=["results", "comparison"])
    p_hdr.set_defaults(
        func=lambda a: (
            sys.stdout.write(
                ",".join(
                    RESULT_COLUMNS if a.which == "results" else COMPARISON_COLUMNS
                )
                + "\n"
            ),
            0,
        )[1]
    )

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
