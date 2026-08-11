#!/bin/bash
#SBATCH --job-name=mcmc_budget
#SBATCH --output=mcmc_budget_%j.out
#SBATCH --error=mcmc_budget_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0

set -uo pipefail

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
source "$HOME/my_venvs/aco_env/bin/activate"

cd "$SLURM_SUBMIT_DIR"

# One Python process per allocated core.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MCMC_WRAPPER="${MCMC_WRAPPER:-mcmc_wrapper.py}"

# Reuse the same ACO seeds at every proposal budget. This isolates the effect
# of MCMC chain length from differences in the initial seed ensemble.
STRICT_SEED_DIR="${STRICT_SEED_DIR:-results_strict_N40_M78_D12_CC0.35_mh_iter50000/seeds}"
LOOSE_SEED_DIR="${LOOSE_SEED_DIR:-results_loose_N40_M195_D4_CC0.4_mh_iter50000/seeds}"

OUTPUT_ROOT="${OUTPUT_ROOT:-mcmc_budget_experiment}"
SAMPLES="${SAMPLES:-50}"
CORES="${SLURM_CPUS_PER_TASK:-128}"

# Shorter proposal budgets. The existing 50,000-proposal run can be added by
# the plotting script without rerunning it.
BUDGETS=(500 2000 5000 10000)

for required_file in \
    "$MCMC_WRAPPER" \
    "clust_diam_MCMC.py" \
    "metrics_utils.py"
do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done

check_seed_files() {
    local seed_dir="$1"
    local expected=$((SAMPLES + 1))
    local missing=0
    local i

    for ((i=1; i<=expected; i++)); do
        if [[ ! -s "$seed_dir/seed_$i.json" ]]; then
            echo "Missing seed: $seed_dir/seed_$i.json" >&2
            missing=$((missing + 1))
        fi
    done

    if (( missing > 0 )); then
        echo "Aborting: $missing seed files are missing from $seed_dir." >&2
        return 1
    fi

    echo "Verified $expected seeds in $seed_dir."
}

throttle_jobs() {
    while [[ $(jobs -rp | wc -l) -ge $CORES ]]; do
        wait -n || true
    done
}

run_treatment() {
    local seed_dir="$1"
    local setting_name="$2"
    local budget="$3"
    local strategy="$4"
    local output_dir="$OUTPUT_ROOT/$setting_name/iter_$budget/$strategy"
    local i seed_idx input_seed

    mkdir -p "$output_dir"

    for ((i=1; i<=SAMPLES; i++)); do
        if [[ -s "$output_dir/sample_$i.json" ]]; then
            continue
        fi

        if [[ "$strategy" == "mcmc_fixed" ]]; then
            input_seed="$seed_dir/seed_1.json"
        else
            seed_idx=$((i + 1))
            input_seed="$seed_dir/seed_$seed_idx.json"
        fi

        throttle_jobs
        python3 "$MCMC_WRAPPER" \
            --input_seed "$input_seed" \
            --output_file "$output_dir/sample_$i.json" \
            --n_iter "$budget" &
    done

    wait || true
}

check_outputs() {
    local output_dir="$1"
    local missing=0
    local i

    for ((i=1; i<=SAMPLES; i++)); do
        if [[ ! -s "$output_dir/sample_$i.json" ]]; then
            echo "Missing output: $output_dir/sample_$i.json" >&2
            missing=$((missing + 1))
        fi
    done

    if (( missing > 0 )); then
        echo "Aborting: $missing outputs are missing from $output_dir." >&2
        return 1
    fi

    echo "Verified $SAMPLES outputs in $output_dir."
}

run_setting() {
    local seed_dir="$1"
    local setting_name="$2"
    local budget

    echo "================================================"
    echo "SETTING: $setting_name"
    echo "SEEDS:   $seed_dir"
    echo "================================================"

    check_seed_files "$seed_dir" || exit 1

    for budget in "${BUDGETS[@]}"; do
        echo "------------------------------------------------"
        echo "$setting_name: N_ITER=$budget"
        echo "------------------------------------------------"

        run_treatment "$seed_dir" "$setting_name" "$budget" "mcmc_fixed"
        check_outputs \
            "$OUTPUT_ROOT/$setting_name/iter_$budget/mcmc_fixed" || exit 1

        run_treatment "$seed_dir" "$setting_name" "$budget" "mcmc_multiseed"
        check_outputs \
            "$OUTPUT_ROOT/$setting_name/iter_$budget/mcmc_multiseed" || exit 1
    done
}

run_setting "$STRICT_SEED_DIR" "strict"
run_setting "$LOOSE_SEED_DIR" "loose"

echo "DONE."
echo "Results written to: $OUTPUT_ROOT"
