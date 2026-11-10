import numpy as np
import json

def phase_indices_to_waveform(q, Q):
    q = np.asarray(q, dtype=int)
    return np.exp(-1j * 2 * np.pi * q / Q)

def autocorrelation_at_lag(x, k):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    return np.sum(x[:N-k] * np.conj(x[k:]))

def all_autocorrelations(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    return np.array([autocorrelation_at_lag(x, k) for k in range(1, N)], dtype=complex)

def islr_from_waveform(x):
    rhos = all_autocorrelations(x)
    return float(np.sum(np.abs(rhos)**2))

def islr_from_phase_indices(q, Q):
    x = phase_indices_to_waveform(q, Q)
    return islr_from_waveform(x)

vectors = []

# Known Barker-related binary cases from the PDF (Q=2)
barker_cases = [
    (5,  [0, 0, 0, 1, 0]),   # Barker-5: + + + - +  -> objective 2 (one valid framing)
    (7,  [0, 0, 0, 1, 1, 0, 1]),  # from PDF: + + + - - + -  -> objective 3
    (11, [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]),  # Barker-11
]

for N, q in barker_cases:
    Q = 2
    x = phase_indices_to_waveform(q, Q)
    F = islr_from_phase_indices(q, Q)
    vectors.append({
        "label": f"barker_like_N{N}_Q{Q}",
        "N": N, "Q": Q,
        "phase_indices": q,
        "waveform_real": x.real.tolist(),
        "waveform_imag": x.imag.tolist(),
        "objective": F,
    })


rng = np.random.default_rng(seed=42)
random_cases = [
    (4, 3, rng.integers(0, 3, size=4).tolist()),
    (6, 4, rng.integers(0, 4, size=6).tolist()),
    (5, 2, rng.integers(0, 2, size=5).tolist()),
]

for N, Q, q in random_cases:
    x = phase_indices_to_waveform(q, Q)
    F = islr_from_phase_indices(q, Q)
    vectors.append({
        "label": f"random_N{N}_Q{Q}",
        "N": N, "Q": Q,
        "phase_indices": q,
        "waveform_real": x.real.tolist(),
        "waveform_imag": x.imag.tolist(),
        "objective": F,
    })

with open("cross_language_vectors.json", "w") as f:
    json.dump(vectors, f, indent=2)

for v in vectors:
    print(v["label"], "N=",v["N"],"Q=",v["Q"],"F=",round(v["objective"],6))
