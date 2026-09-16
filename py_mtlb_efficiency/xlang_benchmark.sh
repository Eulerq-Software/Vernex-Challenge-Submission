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
#
# Adding a solver later = add one line to the SOLVERS registry below.
# Keep N/Q small yourself for enumeration - Q^N grows fast and nothing here
# stops you from asking for something that takes hours.
#
# Usage:
#   scripts/xlang_benchmark.sh                          # everything, defaults
#   scripts/xlang_benchmark.sh --solvers enumeration
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
# Format (pipe separated, 3 fields):
#
#   key | python module:Class | matlab function
#
#   key            short name written into the CSV files
#   python spec    import path and class, "" if not implemented yet
#   matlab fn      function name on the MATLAB path, "" if not implemented yet
#
# Only enumeration exists on both sides today. The commented lines are the
# expected names for the other solvers; uncomment them as they land, and fix
# the class / function names to whatever was actually committed.

SOLVERS=(
  "enumeration|waveform_optimization.solvers.enumeration.solver:EnumerationSolver|enumeration_solver"
  # "random_search|waveform_optimization.solvers.weak_baselines.random_search:RandomSearchSolver|random_search_solver"
  "greedy|waveform_optimization.solvers.weak_baselines.greedy:GreedySolver|greedy_solver"
  # "coordinate_descent|waveform_optimization.solvers.weak_baselines.coordinate_descent:CoordinateDescentSolver|coordinate_descent_solver"
  # "simulated_annealing|waveform_optimization.solvers.strong_baselines.simulated_annealing:SimulatedAnnealingSolver|simulated_annealing_solver"
)

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
N_LIST="4 5 6 7 8 9 10 11 12"
Q_LIST="2 3"
SEED="none"
TOL="1e-9"
TIMEOUT_SECONDS=0         # 0 = no timeout; kills a single solver run past this
FORCE=0
DRY_RUN=0
LIST_ONLY=0
SELECTED_SOLVERS=""
MATLAB_CMD=""

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results/cross_language"

PY_RUNNER="${SCRIPT_DIR}/xlang_run_python_solver.py"
REPORTER="${SCRIPT_DIR}/xlang_report.py"

usage() {
  sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
  # $1 matlab function, $2 N, $3 Q, $4 seed, $5 output json
  local cmd
  cmd="addpath(genpath('${REPO_ROOT}/matlab')); addpath('${SCRIPT_DIR}'); xlang_run_solver('$1','$2','$3','$4','$5');"
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
    IFS='|' read -r key pyspec mlfn <<< "$entry"
    printf '  %-22s python=%-70s matlab=%s\n' \
      "$key" "${pyspec:-<none>}" "${mlfn:-<none>}"
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
  IFS='|' read -r key pyspec mlfn <<< "$entry"
  [[ -z "$key" ]] && continue
  if [[ -n "$SELECTED_SOLVERS" ]] && ! grep -qw -- "$key" <<< "$SELECTED_SOLVERS"; then
    continue
  fi

  for N in $N_LIST; do
    for Q in $Q_LIST; do

      echo "[${key}] N=${N} Q=${Q} seed=${SEED}"

      # ---------------- Python ----------------
      if [[ -z "$pyspec" ]]; then
        echo "  python : no implementation registered"
      elif already_done "$RESULTS_FILE" "$key" "$N" "$Q" "$SEED_FIELD" "python" && [[ $FORCE -eq 0 ]]; then
        echo "  python : already recorded, skipping"
      elif [[ $DRY_RUN -eq 1 ]]; then
        echo "  python : would run ${pyspec}"
      else
        py_json="${TMPDIR_RUN}/${key}_${N}_${Q}_py.json"
        with_timeout python3 "$PY_RUNNER" --solver "$pyspec" --N "$N" --Q "$Q" \
          --seed "$SEED" --out "$py_json"
        if [[ -s "$py_json" ]]; then
          python3 "$REPORTER" row "$py_json" --solver "$key" --language python \
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
      elif already_done "$RESULTS_FILE" "$key" "$N" "$Q" "$SEED_FIELD" "matlab" && [[ $FORCE -eq 0 ]]; then
        echo "  matlab : already recorded, skipping"
      elif [[ $DRY_RUN -eq 1 ]]; then
        echo "  matlab : would run ${mlfn}"
      else
        ml_json="${TMPDIR_RUN}/${key}_${N}_${Q}_ml.json"
        run_matlab_solver "$mlfn" "$N" "$Q" "$SEED" "$ml_json" >/dev/null 2>"${ml_json}.log"
        if [[ -s "$ml_json" ]]; then
          python3 "$REPORTER" row "$ml_json" --solver "$key" --language matlab \
            >> "$RESULTS_FILE"
          echo "  matlab : $(python3 -c "
import json;d=json.load(open('$ml_json'))
print(d['status'], 'F=%s' % d.get('best_objective'), 'evals=%s' % d.get('objective_evaluations'), 't=%.4fs' % d.get('runtime_seconds', 0.0), d.get('error',''))")"
        else
          echo "  matlab : FAILED / TIMED OUT (no record written); see ${ml_json}.log"
          cp "${ml_json}.log" "${RESULTS_DIR}/${key}_N${N}_Q${Q}_matlab_error.log" 2>/dev/null
        fi
      fi

      # ---------------- Comparison ----------------
      if [[ $DRY_RUN -eq 1 ]]; then
        continue
      fi
      if already_done "$COMPARISON_FILE" "$key" "$N" "$Q" "$SEED_FIELD" "" && [[ $FORCE -eq 0 ]]; then
        echo "  compare: already recorded, skipping"
      else
        cmp_row="$(python3 "$REPORTER" compare --csv "$RESULTS_FILE" --solver "$key" \
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

echo "done. rows appended to:"
echo "  ${RESULTS_FILE}"
echo "  ${COMPARISON_FILE}"