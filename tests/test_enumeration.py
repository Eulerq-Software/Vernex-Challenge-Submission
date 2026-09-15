from waveform_optimization.solvers.enumeration.solver import EnumerationSolver


def test_binary_enumeration_n5():
    solver = EnumerationSolver()
    result = solver.solve(N=5, Q=2)

    assert result.best_objective == 2.0


def test_binary_enumeration_n7():
    solver = EnumerationSolver()
    result = solver.solve(N=7, Q=2)

    assert result.best_objective == 3.0


def test_binary_enumeration_n5():
    solver = EnumerationSolver()
    result = solver.solve(N=5, Q=2)

    assert result.best_objective == 2.0


def test_binary_enumeration_n7():
    solver = EnumerationSolver()
    result = solver.solve(N=7, Q=2)

    assert result.best_objective == 3.0