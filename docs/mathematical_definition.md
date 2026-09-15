
# Waveform Optimization for SAR

## 1. Problem definition

We design a discrete-phase radar waveform containing `N` chips.

Each chip has one of `Q` possible phase values. The optimization variable is represented by integer phase indices:

$$
q_i \in \{0,1,\ldots,Q-1\},
\qquad i=0,\ldots,N-1.
$$

The phase index vector is

$$
q=(q_0,q_1,\ldots,q_{N-1}).
$$

## 2. Phase encoding

The complex waveform corresponding to `q` is

$$
x_i=\exp\left(-j\frac{2\pi q_i}{Q}\right).
$$

Every chip has unit magnitude:

$$|x_i|=1.
$$

For binary phase coding, `Q = 2`:

- `q_i = 0` maps to $+1$
- `q_i = 1` maps to $-1$

## 3. Aperiodic autocorrelation

For a positive lag $k$, where

$$
k=1,\ldots,N-1,
$$

the aperiodic autocorrelation is

$$
\rho(k)
=
\sum_{i=0}^{N-k-1}
x_i\overline{x_{i+k}}.
$$

The zero-lag autocorrelation is

$$
\rho(0)=N.
$$

Only positive nonzero lags are included in the optimization objective.

## 4. Objective function

The integrated sidelobe level objective is

$$
F(q)
=
\sum_{k=1}^{N-1}|\rho(k)|^2.
$$

The optimization problem is

$$
\min_{q_0,\ldots,q_{N-1}}
F(q)
$$

subject to

$$
q_i\in\{0,\ldots,Q-1\}.
$$

Lower objective values are better.

## 5. Search-space size

There are `Q` possible values for each of the `N` phase indices.

Therefore, the total number of possible waveforms is

$$
Q^N.
$$

A complete enumeration becomes impractical as `N` increases.

## 6. Reference binary cases

For `Q = 2`, the challenge uses Barker-code reference cases:

| N | Reference objective |
|---:|---:|
| 5 | 2 |
| 7 | 3 |
| 11 | 5 |

These values are used to validate the implementation and compare solver quality.

## 7. Implementation conventions

- Python indices are zero-based.
- Phase indices are integers in `[0, Q - 1]`.
- The objective excludes lag zero.
- The waveform must have unit magnitude.
- Solvers must return phase indices, not only complex waveform values.
- Every reported objective must be recomputed from the returned phase sequence.
- All solvers must use the shared objective implementation.