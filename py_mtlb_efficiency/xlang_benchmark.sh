#!/usr/bin/env bash
#
# xlang_benchmark.sh - Python vs MATLAB solver comparison (challenge doc 6.6)
#
# For each (solver, N, Q) it runs the Python solver, then the MATLAB solver,
# appends one row per language to a per-machine results file, and appends one
# comparison row (|F_py - F_ml| < eps, evaluations, runtimes) to a per-machine
# comparison file.
#
# Rules this script obeys:
#   * Never rewrite. Result files are only ever appended to.
#   * Never redo work. A (solver, N, Q, seed, language) combination that is
#     already in the results file is skipped unless --force is given.
#   * One file per machine. The file is named after the host, and a different
#     hardware/OS fingerprint under the same hostname gets its own file.
#   * No time limits passed to the solvers. time_limit is always None/[] so
#     both languages do exactly the same amount of work. --timeout below is
#     a harness-level killswitch, not the solver's own time_limit parameter.
#   * Same hyperparameters on both sides. Any solver with a params file in
#     the registry (4th field) has that JSON forwarded to BOTH the Python
#     and MATLAB run, so population size, mutation rate, etc. never drift
#     apart between languages - otherwise a "who's better" comparison would
#     be meaningless.
#   * Parameter sweeps don't touch the CSV schema. --variants-dir runs a
#     parameterized solver once per *.json file in that directory, writing
#     each variant's rows under solver key "<solver>__<variant-filename>"
#     (e.g. "genetic_algorithm__high_mutation"). This reuses the existing
#     append-only / dedup / compare logic untouched - a variant is just
#     another solver key as far as the CSV is concerned.
#
# Adding a solver later = add one line to the SOLVERS registry below.
# Keep N/Q small yourself for enumeration - Q^N grows fast and nothing here
# stops you from asking for something that takes hours.
#
# Usage:
#   scripts/xlang_benchmark.sh                          # everything, defaults
#   scripts/xlang_benchmark.sh --solvers enumeration
#   scripts/xlang_benchmark.sh --solvers genetic_algorithm --seed 42
#   scripts/xlang_benchmark.sh --solvers genetic_algorithm --seed 42 \
#       --variants-dir experiments/configs/ga_variants   # sweep every *.json there
#   scripts/xlang_benchmark.sh --n 4,6,8,10 --q 2,3
#   scripts/xlang_benchmark.sh --seed 12345
#   scripts/xlang_benchmark.sh --timeout 60             # kill any single run past 60s
#   scripts/xlang_benchmark.sh --list
#   scripts/xlang_benchmark.sh --dry-run
#   scripts/xlang_benchmark.sh --force                  # re-run and append again

set -uo pipefail

# --------------------------------------------------------------------------
# Solver registry
# --------------------------------------------------------------------------
# Format (pipe separated, 4 fields):
#
#   key | python module:Class | matlab function | params file (optional)
#
#   key            short name written into the CSV files
#   python spec    import path and class, "" if not implemented yet
#   matlab fn      function name on the MATLAB path, "" if not implemented yet
#   params file    filename (not full path) under experiments/configs/,
#                  forwarded to BOTH runners as --params / params_file so
#                  the two languages use identical hyperparameters. Leave
#                  blank for solvers with no tunable knobs, or whose 5th
#                  MATLAB argument means something other than "params"
#                  (e.g. greedy_solver's initial_solution).
#
# GA is randomized: pass --seed explicitly for a reproducible, comparable
# run (the same seed is handed to both languages; each language's own RNG
# still means the two runs are NOT guaranteed to explore identically, but
# they start from the same initial population size/config and same seed).

SOLVERS=(
  # "enumeration|waveform_optimization.solvers.enumeration.solver:EnumerationSolver|enumeration_solver|"
  # "random_search|waveform_optimization.solvers.weak_baselines.random_search:RandomSearchSolver|random_search_solver|"
  # "greedy|waveform_optimization.solvers.weak_baselines.greedy:GreedySolver|greedy_solver|"
  # "coordinate_descent|waveform_optimization.solvers.weak_baselines.coordinate_descent:CoordinateDescentSolver|coordinate_descent_solver|"
  "genetic_algorithm|waveform_optimization.solvers.strong_baselines.genetic_algorithm:GeneticAlgorithmSolver|genetic_algorithm_solver|ga_xlang_params.json"
  # "simulated_annealing|waveform_optimization.solvers.strong_baselines.simulated_annealing:SimulatedAnnealingSolver|simulated_annealing_solver|"
)

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
N_LIST="4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
Q_LIST="2"
SEED="none"
TOL="1e-9"
TIMEOUT_SECONDS=0         # 0 = no timeout; kills a single solver run past this
FORCE=0
DRY_RUN=0
LIST_ONLY=0
SELECTED_SOLVERS=""
MATLAB_CMD=""
VARIANTS_DIR=""

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results/cross_language"
CONFIGS_DIR="${REPO_ROOT}/experiments/configs"

PY_RUNNER="${SCRIPT_DIR}/xlang_run_python_solver.py"
REPORTER="${SCRIPT_DIR}/xlang_report.py"

usage() {
  sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n|--n-list)      N_LIST="${2//,/ }"; shift 2 ;;
    --q|--q-list)      Q_LIST="${2//,/ }"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --tol)             TOL="$2"; shift 2 ;;
    --solvers)         SELECTED_SOLVERS="${2//,/ }"; shift 2 ;;
    --timeout)         TIMEOUT_SECONDS="$2"; shift 2 ;;
    --matlab-cmd)      MATLAB_CMD="$2"; shift 2 ;;
    --results-dir)     RESULTS_DIR="$2"; shift 2 ;;
    --variants-dir)    VARIANTS_DIR="$2"; shift 2 ;;
    --force)           FORCE=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --list)            LIST_ONLY=1; shift ;;
    -h|--help)         usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------------------------
# Machine identity and specs
# --------------------------------------------------------------------------
collect_specs() {
  HOST_RAW="$(hostname -s 2>/dev/null || hostname)"
  HOST="$(printf '%s' "$HOST_RAW" | tr -c '[:alnum:]._-' '-' | sed 's/-*$//')"

  OS_KERNEL="$(uname -sr)"
  ARCH="$(uname -m)"

  if [[ -r /proc/cpuinfo ]]; then
    CPU_MODEL="$(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo)"
    [[ -z "$CPU_MODEL" ]] && CPU_MODEL="$(awk -F': ' '/Model/{print $2; exit}' /proc/cpuinfo)"
  elif command -v sysctl >/dev/null 2>&1; then
    CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  fi
  [[ -z "${CPU_MODEL:-}" ]] && CPU_MODEL="unknown-cpu"

  CPU_CORES="$(command -v nproc >/dev/null 2>&1 && nproc || echo unknown)"

  if [[ -r /proc/meminfo ]]; then
    MEM_TOTAL="$(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  else
    MEM_TOTAL="unknown"
  fi

  if [[ -r /etc/os-release ]]; then
    OS_NAME="$(. /etc/os-release && echo "${PRETTY_NAME:-$NAME}")"
  else
    OS_NAME="$(uname -s)"
  fi

  PY_VERSION="$(python3 -c 'import platform;print(platform.python_version())' 2>/dev/null || echo unknown)"

  FINGERPRINT="$(printf '%s|%s|%s|%s|%s' \
      "$HOST" "$CPU_MODEL" "$CPU_CORES" "$ARCH" "$OS_KERNEL" \
      | sha256sum | cut -c1-8)"
}

# --------------------------------------------------------------------------
# MATLAB / Octave detection
# --------------------------------------------------------------------------
detect_matlab() {
  MATLAB_KIND=""
  if [[ -n "$MATLAB_CMD" ]]; then
    case "$(basename "$MATLAB_CMD")" in
      octave*) MATLAB_KIND="octave" ;;
      *)       MATLAB_KIND="matlab" ;;
    esac
  elif command -v matlab >/dev/null 2>&1; then
    MATLAB_CMD="matlab"; MATLAB_KIND="matlab"
  elif command -v octave-cli >/dev/null 2>&1; then
    MATLAB_CMD="octave-cli"; MATLAB_KIND="octave"
  elif command -v octave >/dev/null 2>&1; then
    MATLAB_CMD="octave"; MATLAB_KIND="octave"
  fi

  if [[ -z "$MATLAB_KIND" ]]; then
    MATLAB_VERSION="not-found"
    return 1
  fi

  if [[ "$MATLAB_KIND" == "matlab" ]]; then
    MATLAB_VERSION="$("$MATLAB_CMD" -batch "fprintf('%s', version)" 2>/dev/null | tail -n1)"
  else
    MATLAB_VERSION="octave $("$MATLAB_CMD" --version 2>/dev/null | head -n1 | awk '{print $NF}')"
  fi
  [[ -z "$MATLAB_VERSION" ]] && MATLAB_VERSION="unknown"
  return 0
}

run_matlab_solver() {
  # $1 matlab function, $2 N, $3 Q, $4 seed, $5 output json, $6 params path (optional)
  local cmd params_path="${6:-}"
  cmd="addpath(genpath('${REPO_ROOT}/matlab')); addpath('${SCRIPT_DIR}'); xlang_run_solver('$1','$2','$3','$4','$5','${params_path}');"
  if [[ "$MATLAB_KIND" == "matlab" ]]; then
    with_timeout "$MATLAB_CMD" -batch "$cmd"
  else
    with_timeout "$MATLAB_CMD" --no-gui --quiet --eval "$cmd"
  fi
}

with_timeout() {
  if [[ "$TIMEOUT_SECONDS" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT_SECONDS" "$@"
  else
    "$@"
  fi
}

# --------------------------------------------------------------------------
# Result files: pick, create with a spec header, never rewrite
# --------------------------------------------------------------------------
init_files() {
  mkdir -p "$RESULTS_DIR"

  RESULTS_FILE="${RESULTS_DIR}/${HOST}.csv"
  if [[ -f "$RESULTS_FILE" ]]; then
    local existing
    existing="$(grep -m1 '^# fingerprint:' "$RESULTS_FILE" | awk '{print $3}')"
    if [[ -n "$existing" && "$existing" != "$FINGERPRINT" ]]; then
      # Same hostname, different machine. Give it its own file so timings
      # from different hardware never end up mixed together.
      RESULTS_FILE="${RESULTS_DIR}/${HOST}__${FINGERPRINT}.csv"
    fi
  fi
  COMPARISON_FILE="${RESULTS_FILE%.csv}_comparison.csv"

  if [[ ! -f "$RESULTS_FILE" ]]; then
    {
      echo "# Vernex waveform optimization - Python vs MATLAB cross-language results"
      echo "# created: $(date -Is)"
      echo "# fingerprint: ${FINGERPRINT}"
      echo "# host: ${HOST_RAW}"
      echo "# cpu: ${CPU_MODEL}"
      echo "# cpu_cores: ${CPU_CORES}"
      echo "# memory: ${MEM_TOTAL}"
      echo "# arch: ${ARCH}"
      echo "# os: ${OS_NAME}"
      echo "# kernel: ${OS_KERNEL}"
      echo "# python: ${PY_VERSION}"
      echo "# matlab: ${MATLAB_VERSION}"
      echo "# time_limit: none (never used by this harness)"
      echo "# note: append-only file; each row is one solver run on this machine"
      python3 "$REPORTER" columns results
    } >> "$RESULTS_FILE"
    echo "created ${RESULTS_FILE}"
  fi

  if [[ ! -f "$COMPARISON_FILE" ]]; then
    {
      echo "# Python vs MATLAB comparison rows for ${HOST_RAW} (fingerprint ${FINGERPRINT})"
      echo "# created: $(date -Is)"
      echo "# tolerance column records the epsilon used for |F_python - F_matlab|"
      python3 "$REPORTER" columns comparison
    } >> "$COMPARISON_FILE"
    echo "created ${COMPARISON_FILE}"
  fi
}

already_done() {
  # $1 file, $2 solver, $3 N, $4 Q, $5 seed-field, $6 language ("" for comparison)
  local file="$1" pattern
  [[ -f "$file" ]] || return 1
  if [[ -n "$6" ]]; then
    pattern="^[^,]*,$2,$3,$4,$5,$6,"
  else
    pattern="^[^,]*,$2,$3,$4,$5,"
  fi
  grep -qE "$pattern" "$file"
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
collect_specs
detect_matlab
HAVE_MATLAB=$?

if [[ $LIST_ONLY -eq 1 ]]; then
  echo "registered solvers:"
  for entry in "${SOLVERS[@]}"; do
    IFS='|' read -r key pyspec mlfn params_file <<< "$entry"
    printf '  %-22s python=%-70s matlab=%-24s params=%s\n' \
      "$key" "${pyspec:-<none>}" "${mlfn:-<none>}" "${params_file:-<none>}"
  done
  exit 0
fi

init_files

SEED_FIELD="$SEED"
[[ "${SEED,,}" == "none" ]] && SEED_FIELD=""

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

echo "repo        : ${REPO_ROOT}"
echo "results     : ${RESULTS_FILE}"
echo "comparison  : ${COMPARISON_FILE}"
echo "machine     : ${CPU_MODEL} (${CPU_CORES} cores, ${MEM_TOTAL})"
echo "python      : ${PY_VERSION}"
echo "matlab      : ${MATLAB_CMD:-<none>} ${MATLAB_VERSION}"
echo "N           : ${N_LIST}"
echo "Q           : ${Q_LIST}"
echo "seed        : ${SEED}"
echo "timeout     : ${TIMEOUT_SECONDS}s (0 = none)"
echo

for entry in "${SOLVERS[@]}"; do
  IFS='|' read -r key pyspec mlfn params_file <<< "$entry"
  [[ -z "$key" ]] && continue
  if [[ -n "$SELECTED_SOLVERS" ]] && ! grep -qw -- "$key" <<< "$SELECTED_SOLVERS"; then
    continue
  fi

  # ------------------------------------------------------------------
  # Build the list of (variant_label, params_path) pairs to run for
  # this solver. Normally there's exactly one: whatever the registry's
  # own params_file resolves to (possibly none). If --variants-dir was
  # given AND this solver takes a params file at all, every *.json in
  # that directory becomes its own variant instead, each recorded
  # under solver key "<key>__<variant-filename>" so it lives alongside
  # everything else in the same append-only CSV with no schema change.
  # ------------------------------------------------------------------
  variant_pairs=()
  if [[ -n "$VARIANTS_DIR" && -n "$params_file" ]]; then
    shopt -s nullglob
    for vf in "$VARIANTS_DIR"/*.json; do
      variant_pairs+=("$(basename "$vf" .json)|$vf")
    done
    shopt -u nullglob
    if [[ ${#variant_pairs[@]} -eq 0 ]]; then
      echo "warning: --variants-dir given but no *.json files found in ${VARIANTS_DIR}; falling back to '${key}' default params" >&2
    fi
  fi
  if [[ ${#variant_pairs[@]} -eq 0 ]]; then
    default_params_path=""
    if [[ -n "$params_file" ]]; then
      default_params_path="${CONFIGS_DIR}/${params_file}"
      if [[ ! -f "$default_params_path" ]]; then
        echo "warning: params file not found for '${key}': ${default_params_path} (running with each language's own defaults)" >&2
        default_params_path=""
      fi
    fi
    variant_pairs=("|${default_params_path}")
  fi

  for variant_pair in "${variant_pairs[@]}"; do
    variant_label="${variant_pair%%|*}"
    params_path="${variant_pair#*|}"
    effective_key="$key"
    [[ -n "$variant_label" ]] && effective_key="${key}__${variant_label}"

  for N in $N_LIST; do
    for Q in $Q_LIST; do

      echo "[${effective_key}] N=${N} Q=${Q} seed=${SEED}${params_path:+ params=$(basename "$params_path")}"

      # ---------------- Python ----------------
      if [[ -z "$pyspec" ]]; then
        echo "  python : no implementation registered"
      elif already_done "$RESULTS_FILE" "$effective_key" "$N" "$Q" "$SEED_FIELD" "python" && [[ $FORCE -eq 0 ]]; then
        echo "  python : already recorded, skipping"
      elif [[ $DRY_RUN -eq 1 ]]; then
        echo "  python : would run ${pyspec}${params_path:+ (params: $params_path)}"
      else
        py_json="${TMPDIR_RUN}/${effective_key}_${N}_${Q}_py.json"
        py_args=(--solver "$pyspec" --N "$N" --Q "$Q" --seed "$SEED" --out "$py_json")
        [[ -n "$params_path" ]] && py_args+=(--params "$params_path")
        with_timeout python3 "$PY_RUNNER" "${py_args[@]}"
        if [[ -s "$py_json" ]]; then
          python3 "$REPORTER" row "$py_json" --solver "$effective_key" --language python \
            >> "$RESULTS_FILE"
          echo "  python : $(python3 -c "
import json;d=json.load(open('$py_json'))
print(d['status'], 'F=%s' % d.get('best_objective'), 'evals=%s' % d.get('objective_evaluations'), 't=%.4fs' % d.get('runtime_seconds', 0.0), d.get('error',''))")"
        else
          echo "  python : FAILED / TIMED OUT (no record written)"
        fi
      fi

      # ---------------- MATLAB ----------------
      if [[ -z "$mlfn" ]]; then
        echo "  matlab : no implementation registered"
      elif [[ $HAVE_MATLAB -ne 0 ]]; then
        echo "  matlab : interpreter not found, skipping"
      elif already_done "$RESULTS_FILE" "$effective_key" "$N" "$Q" "$SEED_FIELD" "matlab" && [[ $FORCE -eq 0 ]]; then
        echo "  matlab : already recorded, skipping"
      elif [[ $DRY_RUN -eq 1 ]]; then
        echo "  matlab : would run ${mlfn}${params_path:+ (params: $params_path)}"
      else
        ml_json="${TMPDIR_RUN}/${effective_key}_${N}_${Q}_ml.json"
        run_matlab_solver "$mlfn" "$N" "$Q" "$SEED" "$ml_json" "$params_path" >/dev/null 2>"${ml_json}.log"
        if [[ -s "$ml_json" ]]; then
          python3 "$REPORTER" row "$ml_json" --solver "$effective_key" --language matlab \
            >> "$RESULTS_FILE"
          echo "  matlab : $(python3 -c "
import json;d=json.load(open('$ml_json'))
print(d['status'], 'F=%s' % d.get('best_objective'), 'evals=%s' % d.get('objective_evaluations'), 't=%.4fs' % d.get('runtime_seconds', 0.0), d.get('error',''))")"
        else
          echo "  matlab : FAILED / TIMED OUT (no record written); see ${ml_json}.log"
          cp "${ml_json}.log" "${RESULTS_DIR}/${effective_key}_N${N}_Q${Q}_matlab_error.log" 2>/dev/null
        fi
      fi

      # ---------------- Comparison ----------------
      if [[ $DRY_RUN -eq 1 ]]; then
        continue
      fi
      if already_done "$COMPARISON_FILE" "$effective_key" "$N" "$Q" "$SEED_FIELD" "" && [[ $FORCE -eq 0 ]]; then
        echo "  compare: already recorded, skipping"
      else
        cmp_row="$(python3 "$REPORTER" compare --csv "$RESULTS_FILE" --solver "$effective_key" \
          --N "$N" --Q "$Q" --seed "$SEED" --tol "$TOL" 2>"${TMPDIR_RUN}/cmp.err")"
        cmp_status=$?
        if [[ -n "$cmp_row" ]]; then
          printf '%s\n' "$cmp_row" >> "$COMPARISON_FILE"
        fi
        sed 's/^/  compare: /' "${TMPDIR_RUN}/cmp.err"
        if [[ $cmp_status -eq 1 ]]; then
          echo "  compare: OBJECTIVE MISMATCH - investigate before trusting these rows" >&2
        fi
      fi

      echo
    done
  done
  done
done

echo "done. rows appended to:"
echo "  ${RESULTS_FILE}"
echo "  ${COMPARISON_FILE}"