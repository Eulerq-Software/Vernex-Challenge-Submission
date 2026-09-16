function test_cross_language_vectors()
%TEST_CROSS_LANGUAGE_VECTORS Check MATLAB core against fixed reference
%vectors that were computed once (from the Python core) and stored on
%disk, per the project spec section 6.3: "test vectors should be stored
%explicitly rather than relying on Python and MATLAB to generate
%identical random sequences from the same seed."
%
%   Run with:  test_cross_language_vectors
%   Requires matlab/core on the path, and matlab/data/cross_language_vectors.json
%   to exist (generate it with data/generate_vectors.py, or use the
%   copy already checked in).

    this_dir = fileparts(mfilename('fullpath'));
    data_path = fullfile(this_dir, '..', 'cross_lan_test', 'cross_language_vectors.json');

    if ~isfile(data_path)
        error('test_cross_language_vectors:MissingData', ...
            'Reference vector file not found: %s', data_path);
    end

    raw = fileread(data_path);
    vectors = jsondecode(raw);  % returns a struct array (or cell, depending on version)

    if isstruct(vectors)
        n = numel(vectors);
        get = @(v, f) v.(f);
    else
        n = numel(vectors);
        get = @(v, f) v.(f);
    end

    n_pass = 0; n_fail = 0;
    tol = 1e-6;

    for i = 1:n
        if iscell(vectors)
            v = vectors{i};
        else
            v = vectors(i);
        end

        label = v.label;
        N = v.N;
        Q = v.Q;
        q = v.phase_indices(:).';
        expected_F = v.objective;
        expected_x = v.waveform_real(:).' + 1i * v.waveform_imag(:).';

        try
            x = phase_indices_to_waveform(q, Q);
            assert(numel(x) == N, 'waveform length mismatch');
            assert(max(abs(x - expected_x)) < tol, ...
                sprintf('waveform mismatch (max abs diff %.3e)', ...
                max(abs(x - expected_x))));

            F = islr_from_phase_indices(q, Q);
            assert(abs(F - expected_F) < tol, ...
                sprintf('objective mismatch: got %.6f expected %.6f', F, expected_F));

            printf('[PASS] %s (N=%d, Q=%d, F=%.4f)\n', label, N, Q, F);
            n_pass = n_pass + 1;
        catch err
            printf('[FAIL] %s : %s\n', label, err.message);
            n_fail = n_fail + 1;
        end
    end

    printf('\n%d passed, %d failed (of %d vectors)\n', n_pass, n_fail, n);

    if n_fail > 0
        error('test_cross_language_vectors:Failures', ...
            '%d cross-language vector(s) failed.', n_fail);
    end
end
