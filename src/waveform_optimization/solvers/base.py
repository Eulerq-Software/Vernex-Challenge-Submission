
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult


class Solver(ABC):
    """
    Common interface for all waveform-optimization solvers.
    """

    @abstractmethod
    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        """
        Solve the waveform-optimization problem.

        Parameters
        ----------
        N:
            Number of waveform chips.
        Q:
            Number of phase choices.
        time_limit:
            Maximum allowed runtime in seconds.
        seed:
            Random seed, if applicable.
        initial_solution:
            Optional starting phase-index sequence.

        Returns
        -------
        A SolverResult containing the optimization outcome.
        """
        raise NotImplementedError