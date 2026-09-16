function run_all_tests()
%RUN_ALL_TESTS Run every test file in this folder and report a summary.
%
%   Run with:  run_all_tests
%   from inside matlab/tests (or after adding it to the path).
%
%   This adds ../core and ../solvers to the path for you, so you don't
%   need to do it manually every time.

    this_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(this_dir, '..', 'core'));
    addpath(fullfile(this_dir, '..', 'solvers'));

    suites = {
        'test_core_functions', ...
        'test_enumeration_solver', ...
        'test_cross_language_vectors' ...
    };

    any_failed = false;

    for i = 1:numel(suites)
        name = suites{i};
        printf('\n=== Running %s ===\n', name);
        try
            feval(name);
        catch err
            printf('*** %s reported failures: %s ***\n', name, err.message);
            any_failed = true;
        end
    end

    printf('\n==========================================\n');
    if any_failed
        printf('RESULT: one or more test suites had failures.\n');
    else
        printf('RESULT: all test suites passed.\n');
    end
end
