function simple_test()
    addpath('../core')
    addpath('../solvers')
    r = genetic_algorithm_solver(10, 3);
    disp(r);
end