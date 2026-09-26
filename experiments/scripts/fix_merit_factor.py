#!/usr/bin/env python3
"""Safely correct merit_factor in the ALNS convergence run results CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

DEFAULT_PATH = Path("results/alns_convergence_v2/maruthu/run_results.csv")
MERIT_FIELD = "merit_factor"


def _merit_factor(row: dict[str, str]) -> float | None:
    best_energy = row.get("best_energy", "").strip()
    if not best_energy:
        return None
    energy = float(best_energy)
    if energy == 0:
        return None
    return float(int(row["N"]) ** 2 / (2 * energy))


def fix_merit_factor(path: Path) -> tuple[int, list[float], int]:
    """Rewrite only merit_factor after validating the complete replacement."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or MERIT_FIELD not in fieldnames:
            raise ValueError(f"CSV must contain {MERIT_FIELD!r}.")
        original_rows = list(reader)

    corrected_values: list[float] = []
    blank_or_zero_rows = 0
    updated_rows: list[dict[str, str]] = []
    for row in original_rows:
        corrected = _merit_factor(row)
        updated = dict(row)
        if corrected is None:
            blank_or_zero_rows += 1
        else:
            updated[MERIT_FIELD] = repr(corrected)
            corrected_values.append(corrected)
        updated_rows.append(updated)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(updated_rows)
            handle.flush()
            os.fsync(handle.fileno())

        with Path(temp_name).open("r", newline="", encoding="utf-8") as handle:
            rewritten_reader = csv.DictReader(handle)
            if rewritten_reader.fieldnames != fieldnames:
                raise ValueError("Temporary CSV header differs from the source header.")
            rewritten_rows = list(rewritten_reader)

        if len(rewritten_rows) != len(original_rows):
            raise ValueError("Temporary CSV row count differs from the source row count.")
        for original, rewritten in zip(original_rows, rewritten_rows):
            for field in fieldnames:
                if field != MERIT_FIELD and original[field] != rewritten[field]:
                    raise ValueError(f"Unexpected change to CSV field {field!r}.")

        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)

    return len(updated_rows) - blank_or_zero_rows, corrected_values[:5], blank_or_zero_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    updated_rows, first_five, blank_or_zero_rows = fix_merit_factor(args.path)
    print(f"rows updated: {updated_rows}")
    print(f"first five corrected values: {first_five}")
    print(f"count of blank/zero-energy rows: {blank_or_zero_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
