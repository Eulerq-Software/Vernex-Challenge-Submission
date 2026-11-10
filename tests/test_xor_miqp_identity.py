"""Independent algebra check for the constrained MIQP's XOR representation.

This checks the formula used by the Gurobi branch without importing gurobipy.
It does not test the Gurobi solver or an unconstrained QUBO.
"""

from itertools import product

import pytest


@pytest.mark.parametrize("N", [2, 3, 5, 7])
def test_xor_autocorrelation_matches_direct_binary_spins(N: int) -> None:
    for bits in product((0, 1), repeat=N):
        spins = [1 - 2 * bit for bit in bits]
        direct = 0
        xor_form = 0
        for k in range(1, N):
            c_direct = sum(spins[i] * spins[i + k] for i in range(N - k))
            c_xor = (N - k) - 2 * sum(bits[i] != bits[i + k] for i in range(N - k))
            assert c_direct == c_xor
            direct += c_direct * c_direct
            xor_form += c_xor * c_xor
        assert direct == xor_form
