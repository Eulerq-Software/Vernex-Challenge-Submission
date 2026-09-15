from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SolverResult:
    """Standard result returned by a waveform optimization solver."""

    algorithm: str
    N: int
    Q: int

    best_phase_indices: list[int]
    best_objective: float

    runtime_seconds: float
    objective_evaluations: int
    status: str

    seed: int | None = None
    time_limit_seconds: float | None = None

    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a serializable dictionary."""
        return asdict(self)