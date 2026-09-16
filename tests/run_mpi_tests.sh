#!/usr/bin/env bash
# Run SuperNeuroABM's multi-rank tests across several rank counts.
#
# This repo has no pytest configuration at all, so the file list lives here rather
# than behind a marker. Plain `pytest tests/` runs everything single-rank; the
# multi-rank cases in these files self-skip there, so nothing below is covered by
# the default test run.
#
# One GPU is enough: --oversubscribe lets N ranks share it. Rank count changes the
# partition, so each file is run at each rank count separately.
#
# Usage:
#   scripts/run_mpi_tests.sh                # ranks 1 2 4
#   scripts/run_mpi_tests.sh 2              # only 2 ranks
#   PYTHON=/path/to/python scripts/run_mpi_tests.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

PYTHON="${PYTHON:-python}"
TIMEOUT="${TIMEOUT:-900}"
RANKS=("${@:-1 2 4}")
read -r -a RANKS <<< "${RANKS[*]}"

# test_mpi_comparison and test_lif_mixed_synapses_stdp_mpi are written for 4 ranks;
# test_spike_injection and test_load_from_adjacency for 2. Each self-skips the
# cases that do not apply to the rank count it is given.
FILES=(
    tests/test_spike_injection.py
    tests/test_load_from_adjacency.py
    tests/test_mpi_comparison.py
    tests/test_lif_mixed_synapses_stdp_mpi.py
)

printf '%-42s' "test"
for n in "${RANKS[@]}"; do printf '%10s' "n=$n"; done
echo

failed=0
for f in "${FILES[@]}"; do
    printf '%-42s' "$(basename "$f")"
    for n in "${RANKS[@]}"; do
        if [ "$n" -eq 1 ]; then
            out=$(timeout "$TIMEOUT" "$PYTHON" -m pytest "$f" -q 2>&1)
        else
            out=$(timeout "$TIMEOUT" mpirun --oversubscribe -n "$n" \
                  "$PYTHON" -m pytest "$f" -q 2>&1)
        fi
        rc=$?
        clean=$(printf '%s' "$out" | sed -e 's/\x1b\[[0-9;]*m//g')
        if [ $rc -eq 124 ]; then
            printf '%10s' "TIMEOUT"; failed=1
        elif [ $rc -ne 0 ]; then
            printf '%10s' "FAIL"; failed=1
            printf '\n%s\n' "$clean" | grep -E '^(FAILED|ERROR)' | sort -u | sed 's/^/      /'
            printf '%-42s' ""
        else
            p=$(printf '%s' "$clean" | grep -oE '[0-9]+ passed' | head -1 | cut -d' ' -f1)
            printf '%10s' "${p:-0}ok"
        fi
    done
    echo
done

if [ $failed -ne 0 ]; then
    echo
    echo "FAILURES above. Note a rank-0-only readback deadlocks rather than failing:"
    echo "get_agent_property_value is collective, so every rank must call it."
    exit 1
fi
echo
echo "All MPI tests passed at ranks: ${RANKS[*]}"
