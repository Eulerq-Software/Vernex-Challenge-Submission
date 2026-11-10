# Quantum solvers (planned)

No quantum hardware, QAOA, quantum annealing or LABS SimCIM solver is implemented in the supplied Git bundle. The original binary-phase LABS classical solvers operate on the direct spin objective. The Gurobi module uses a constrained MIQP with auxiliary XOR bits and linear constraints; this is not an unconstrained QUBO implementation.

For a detailed design and caveats, see [`docs/quantum_scope.md`](../../../../docs/quantum_scope.md).
