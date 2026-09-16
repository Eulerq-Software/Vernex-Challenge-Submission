function simple_test()
    addpath('../core')
    addpath('../solvers')
    r = enumeration_solver(10, 3, 5);
    disp(r);
end