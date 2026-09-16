function test_core_functions()
%TEST_CORE_FUNCTIONS Script-style tests for the MATLAB common core.
%
%   Run with:  test_core_functions
%   (from the matlab/tests folder, with matlab/core on the path)
%
%   This is a plain script-based test file (works identically in MATLAB
%   and GNU Octave) rather than a matlab.unittest class, so it is easy
%   to read and run without any toolboxes. Each check either passes
%   silently or throws, and pass/fail is printed per test.

    tests = {
        @test_binary_phase_encoding, ...
        @test_waveform_has_unit_magnitude, ...
        @test_autocorrelation_at_lag_matches_definition, ...
        @test_all_autocorrelations_length, ...
        @test_islr_from_waveform_matches_manual_sum, ...
        @test_islr_from_phase_indices_zero_for_single_chip_is_error, ...
        @test_barker5_reference_value, ...
        @test_barker7_reference_value, ...
        @test_barker11_reference_value, ...
        @test_validate_phase_indices_rejects_out_of_range, ...
        @test_validate_phase_indices_rejects_wrong_length, ...
        @test_validate_waveform_rejects_bad_magnitude ...
    };

    n_pass = 0;
    n_fail = 0;

    for i = 1:numel(tests)
        fn = tests{i};
        name = func2str(fn);
        try
            fn();
            printf('[PASS] %s\n', name);
            n_pass = n_pass + 1;
        catch err
            printf('[FAIL] %s : %s\n', name, err.message);
            n_fail = n_fail + 1;
        end
    end

    printf('\n%d passed, %d failed (of %d)\n', n_pass, n_fail, numel(tests));

    if n_fail > 0
        error('test_core_functions:Failures', '%d test(s) failed.', n_fail);
    end
end

% ---- individual tests -------------------------------------------------

function test_binary_phase_encoding()
    x = phase_indices_to_waveform([0, 1], 2);
    expected = [1, -1];
    assert(max(abs(x - expected)) < 1e-10, 'waveform mismatch');
end

function test_waveform_has_unit_magnitude()
    x = phase_indices_to_waveform([0, 1, 2, 3], 4);
    assert(max(abs(abs(x) - 1)) < 1e-10, 'not unit magnitude');
end

function test_autocorrelation_at_lag_matches_definition()
    x = [1, 1i, -1, -1i];
    k = 1;
    rho_direct = sum(x(1:end-k) .* conj(x(k+1:end)));
    rho_fn = autocorrelation_at_lag(x, k);
    assert(abs(rho_direct - rho_fn) < 1e-10, 'lag-1 mismatch');
end

function test_all_autocorrelations_length()
    x = phase_indices_to_waveform([0 1 2 3 0], 4);
    rhos = all_autocorrelations(x);
    assert(numel(rhos) == numel(x) - 1, 'wrong number of lags');
end

function test_islr_from_waveform_matches_manual_sum()
    x = phase_indices_to_waveform([0 1 0 1], 2);
    rhos = all_autocorrelations(x);
    manual = sum(abs(rhos) .^ 2);
    F = islr_from_waveform(x);
    assert(abs(manual - F) < 1e-10, 'objective mismatch');
end

function test_islr_from_phase_indices_zero_for_single_chip_is_error()
    threw = false;
    try
        all_autocorrelations([1]);
    catch
        threw = true;
    end
    assert(threw, 'expected an error for a length-1 waveform');
end

function test_barker5_reference_value()
    % Reference value from the project spec: N=5 binary case -> F=2
    F = islr_from_phase_indices([0 0 0 1 0], 2);
    assert(abs(F - 2.0) < 1e-9, sprintf('got %.6f, expected 2', F));
end

function test_barker7_reference_value()
    % From the spec: [0,0,0,1,1,0,1] (i.e. + + + - - + -) -> F=3
    F = islr_from_phase_indices([0 0 0 1 1 0 1], 2);
    assert(abs(F - 3.0) < 1e-9, sprintf('got %.6f, expected 3', F));
end

function test_barker11_reference_value()
    F = islr_from_phase_indices([0 0 0 1 1 1 0 1 1 0 1], 2);
    assert(abs(F - 5.0) < 1e-9, sprintf('got %.6f, expected 5', F));
end

function test_validate_phase_indices_rejects_out_of_range()
    threw = false;
    try
        validate_phase_indices([0 1 2], [], 2);  % 2 is out of range for Q=2
    catch
        threw = true;
    end
    assert(threw, 'expected out-of-range phase index to be rejected');
end

function test_validate_phase_indices_rejects_wrong_length()
    threw = false;
    try
        validate_phase_indices([0 1], 3, 2);
    catch
        threw = true;
    end
    assert(threw, 'expected wrong-length sequence to be rejected');
end

function test_validate_waveform_rejects_bad_magnitude()
    threw = false;
    try
        validate_waveform([1, 0.5]);
    catch
        threw = true;
    end
    assert(threw, 'expected non-unit-magnitude waveform to be rejected');
end
