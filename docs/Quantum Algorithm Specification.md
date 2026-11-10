# Proposed quantum-assisted ALNS algorithm for LABS

## Proposed algorithm and motivation

**Main proposal:** hybrid Adaptive Large Neighbourhood Search (ALNS) with quantum-assisted repair. The classical ALNS controller optimizes a full-length binary radar waveform; a quantum algorithm works on selected smaller sets of phase variables. Full objective evaluation, local descent, acceptance and adaptive operator scoring remain classical.

A quantum-only optimizer for large LABS instances faces practical challenges. The original binary-spin energy contains up to four-body interactions with potentially dense cost structure. An exact unconstrained QUBO transformation can substantially increase logical variable count and coupling/penalty requirements. For present gate-model or annealing hardware, such overhead may make whole-instance execution unattractive. We do **not** claim that a full quantum treatment is mathematically impossible; instead, we propose that restricted quantum subproblems are a more credible near-term route to useful hybrid performance.

## Mathematical target

For spins `s_i ∈ {−1,+1}`, set `C_k = Σ_{i=0}^{N-k-1} s_i s_{i+k}` and `E(s) = Σ_{k=1}^{N-1} C_k²`. For an ALNS-selected active set `S` with `|S| = m`, all other spins remain fixed and the exact quantum subproblem is

`E_S(u | s) = E(s_{outside S}, u_S)`, for `u ∈ {−1,+1}^m`.

When reduced to active variables, this remains a diagonal Ising polynomial of degree at most four. Two quantum execution paths are proposed:

- **Gate-model QAOA:** implement the restricted higher-order diagonal Hamiltonian directly, using `m` logical qubits before any additional hardware/compiler requirements.
- **QUBO/annealing:** introduce product auxiliaries and sufficiently strong quadratic consistency penalties; the resulting quadratic binary objective has the same global optimum as the restricted LABS subproblem. Auxiliary count, coefficient ranges and embedding cost must be measured for each selected neighbourhood.

The implemented classical solver stack evaluates the original LABS energy directly. The Gurobi module separately uses auxiliary XOR variables and explicit linear constraints in a **constrained MIQP**; this is **not** the unconstrained QUBO proposed for quantum annealing.

## Proposed ALNS + quantum repair workflow

1. Start from a valid sequence obtained using classical initialization and local descent.
2. Use ALNS destroy/selection logic to identify a promising active set `S` with `m << N`.
3. Construct and verify the exact restricted objective `E_S`.
4. Invoke quantum repair (e.g., QAOA) when resource gating permits; otherwise use classical repair. For fair evaluation, also run matched classical-repair baselines on the same subproblem.
5. Restore each sampled assignment to a complete waveform, apply classical descent, and evaluate exact full-length LABS energy.
6. Accept or reject using an ALNS rule, update the best incumbent and adapt repair selection using *post-descent gain per end-to-end time*.

### Productive uphill triggers and basin-trained QAOA

A central finding of the classical landscape experiments is that a locally improving move may be unavailable even when a multi-step *uphill perturbation followed by descent* reaches a better basin. Standard QAOA minimizing the immediate energy of `E_S` may overlook such triggers.

We therefore propose comparing ordinary **energy-trained QAOA** with **basin-trained QAOA**. In the latter case, circuit samples are scored by the classically computed quantity `E(Relax(restore(s, S, u)))` when updating variational parameters. The quantum circuit still uses an implementable restricted cost Hamiltonian; the post-relaxation objective is evaluated *after measurement* and does not presume a quantum oracle for the classical descent routine. The approach may increase productive basin transitions, but this requires direct tests.

## Required experiment to demonstrate value

The decisive comparison is **the same ALNS framework with quantum repair versus classical-only repair**, using identical starting waveforms, active sets, full-objective verification, descent operators and wall-clock budgets. Report improvement probability, final energy or merit factor, successes per second, simulator/QPU costs, circuit depth, logical-qubit count, QUBO auxiliary variables, and scaling with `N` and `m`. Separate proof-of-concept simulator measurements, QPU measurements, projected benefits and validated end-to-end advantage.

## Evidence and current status

| Component | Status |
| --- | --- |
| Binary matched-filter LABS objective and baseline classical solvers | Implemented and tested |
| Classical ALNS framework, solver adapter and telemetry | Implemented; further algorithm review recommended |
| Landscape studies and guided perturbation findings | Experimentally studied; later raw datasets need addition to this repository |
| Restricted higher-order Ising and exact QUBO construction | Analytical design in dossier; not a measured quantum benchmark |
| QAOA / quantum-assisted ALNS repair for LABS | Proposed; not implemented in the supplied Git bundle |
| Quantum advantage of the hybrid workflow | Hypothesis; not established |
| Polyphase extension `Q>2` | Proposed; not benchmarked |

## Extension to more than two phases

For `Q>2`, take `x_i = exp(−2πj q_i/Q)` for `q_i ∈ {0,...,Q−1}` and minimize the squared aperiodic sidelobes. A one-hot or compact binary phase encoding leads to discrete nonlinear objectives that can be represented by higher-order binary Hamiltonians or quadratized when useful. ALNS can select phase variables for repair and use generalized mutation/descent operators. We propose starting with `Q=4`, verifying the classical objective and resource scaling before evaluating quantum repair.
