#!/usr/bin/env python3
"""
Plot incumbent-vs-time and best-bound-vs-time convergence curves for
Surya's Gurobi exact runs, and clearly distinguish:

  - CERTIFIED OPTIMAL  (status == OPTIMAL, gap == 0): incumbent and
    bound curves visually meet; a marker + annotation records the
    proof time.
  - FEASIBLE-ONLY / UNPROVEN (status == TIME_LIMIT): curves are left
    open at the time limit, shaded to show the remaining gap, and
    labeled "best known, unproven" rather than "optimal".

Reads the three CSVs produced by gurobi_exact_solver.py:
    gurobi_runs.csv         (one row per N: final status/energy/bound)
    gurobi_incumbents.csv   (event log: every new_incumbent / bound_update)
    gurobi_checkpoints.csv  (fixed-time snapshots; not required here but
                              cross-checked against the event log)

Produces:
    plots/convergence_grid.png   - one small-multiple panel per N
    plots/final_quality_vs_N.png - summary: energy & bound vs N,
                                    certified vs time-limited
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOUND_SENTINEL = -1e100  # "no bound computed yet" placeholder from Gurobi


def load_data(root: Path):
    runs = pd.read_csv(root / "gurobi_runs.csv")
    incumbents = pd.read_csv(root / "gurobi_incumbents.csv")
    return runs, incumbents


def step_series(times: np.ndarray, values: np.ndarray, t_end: float):
    """Turn (time, value) event pairs into step-plot coordinates that
    hold the last known value out to t_end (no forward leakage)."""
    if len(times) == 0:
        return np.array([0.0, t_end]), np.array([np.nan, np.nan])
    t = np.concatenate([times, [t_end]])
    v = np.concatenate([values, [values[-1]]])
    return t, v


def plot_one_panel(ax, N: int, run_row: pd.Series, events: pd.DataFrame):
    t_limit = float(run_row["time_limit_seconds"])
    status = run_row["status"]
    is_optimal = status == "OPTIMAL"

    incumbent_events = events[events.event_type == "new_incumbent"].sort_values("elapsed_seconds")
    bound_events = events.sort_values("elapsed_seconds")  # bound reported on both event types
    bound_events = bound_events[bound_events.bound > BOUND_SENTINEL]  # drop the "-inf" placeholder

    # Certified runs stop early (proof time); time-limited runs are
    # drawn out to the full time limit to show they were cut off.
    t_end = (
        float(run_row["time_to_certified_optimality"])
        if is_optimal
        else t_limit
    )

    it, iv = step_series(
        incumbent_events.elapsed_seconds.to_numpy(),
        incumbent_events.objective.to_numpy(),
        t_end,
    )
    bt, bv = step_series(
        bound_events.elapsed_seconds.to_numpy(),
        bound_events.bound.to_numpy(),
        t_end,
    )

    ax.step(it, iv, where="post", color="#1f77b4", lw=1.8, label="Incumbent (feasible, upper bound)")
    ax.step(bt, bv, where="post", color="#d62728", lw=1.8, label="Best bound (proven lower bound)")

    # Shade the still-open gap for time-limited runs; leave it closed
    # (no fill) once incumbent == bound for certified-optimal runs.
    if not is_optimal:
        ax.fill_between(it, iv, np.interp(it, bt, bv), color="#d62728", alpha=0.08)

    x_floor = max(it[0], bt[0], 1e-4)
    ax.set_xscale("log")
    ax.set_xlim(x_floor, t_limit * 1.05)

    # For time-limited runs the bound starts from a very loose root
    # relaxation (often deeply negative); auto y-limits would squash
    # the late-stage narrowing that's actually of interest. Zoom in
    # on the region around the final incumbent/bound instead, and let
    # the early dip clip off the bottom of the panel.
    if not is_optimal:
        final_bound = float(run_row["best_proven_lower_bound"])
        final_energy = float(run_row["best_feasible_energy"])
        span = max(final_energy - final_bound, 1.0)
        ax.set_ylim(final_bound - 0.25 * span, final_energy + 0.6 * span)

    if is_optimal:
        ax.axvline(t_end, color="green", ls="--", lw=1)
        ax.scatter([t_end], [run_row["best_feasible_energy"]], color="green", zorder=5, s=30)
        label = f"CERTIFIED OPTIMAL\nt={t_end:.3g}s, E={run_row['best_feasible_energy']:.0f}"
        va = "bottom"
    else:
        label = (
            f"TIME LIMIT ({t_limit:.0f}s)\n"
            f"best feasible={run_row['best_feasible_energy']:.0f} (unproven)\n"
            f"best bound={run_row['best_proven_lower_bound']:.0f}\n"
            f"gap={run_row['relative_mip_gap']*100:.1f}%"
        )
        va = "top"

    ax.text(
        0.98, 0.95 if va == "top" else 0.05, label,
        transform=ax.transAxes, ha="right", va=va, fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
    )

    ax.set_title(f"N={N}  ({run_row['role']})", fontsize=10)
    ax.set_xlabel("elapsed time (s, log scale)", fontsize=8)
    ax.set_ylabel("objective (ISL energy)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which="both")


def plot_grid(runs: pd.DataFrame, incumbents: pd.DataFrame, out_path: Path):
    sizes = sorted(runs.N.unique())
    ncols = 3
    nrows = -(-len(sizes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, N in zip(axes, sizes):
        run_row = runs[runs.N == N].iloc[0]
        events = incumbents[incumbents.N == N]
        plot_one_panel(ax, N, run_row, events)

    for ax in axes[len(sizes):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Incumbent-vs-time and best-bound-vs-time (Gurobi exact, Q=2)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_summary(runs: pd.DataFrame, out_path: Path):
    runs = runs.sort_values("N")
    fig, ax = plt.subplots(figsize=(8, 5))

    optimal = runs[runs.status == "OPTIMAL"]
    timelimit = runs[runs.status != "OPTIMAL"]

    ax.plot(runs.N, runs.best_feasible_energy, color="#1f77b4", lw=1.5, zorder=1,
            label="Best feasible energy (incumbent)")
    ax.plot(runs.N, runs.best_proven_lower_bound.clip(lower=0), color="#d62728", lw=1.5,
            ls="--", zorder=1, label="Best proven lower bound")

    ax.scatter(optimal.N, optimal.best_feasible_energy, color="green", marker="o", s=90,
               zorder=3, label="Certified optimal")
    ax.scatter(timelimit.N, timelimit.best_feasible_energy, color="orange", marker="^", s=90,
               zorder=3, label="Time-limited (unproven)")

    for _, row in timelimit.iterrows():
        ax.annotate(f"gap {row.relative_mip_gap*100:.0f}%",
                    (row.N, row.best_feasible_energy),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="orange")

    ax.set_xlabel("N (sequence length)")
    ax.set_ylabel("ISL energy")
    ax.set_yscale("log")
    ax.set_title("Final feasible quality vs. proven bound, by N")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                         help="Folder containing the three CSVs.")
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs, incumbents = load_data(args.root)

    plot_grid(runs, incumbents, args.out / "convergence_grid.png")
    plot_summary(runs, args.out / "final_quality_vs_N.png")

    print(f"Wrote {args.out / 'convergence_grid.png'}")
    print(f"Wrote {args.out / 'final_quality_vs_N.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())