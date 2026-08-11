#!/bin/bash
#SBATCH --job-name=random_mcmc_plot
#SBATCH --output=random_mcmc_plot_%j.out
#SBATCH --error=random_mcmc_plot_%j.err
#SBATCH --time=02:00:00
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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MCMC_WRAPPER="${MCMC_WRAPPER:-mcmc_wrapper.py}"
PLOTTER="${PLOTTER:-plot_mcmc_3way_results.py}"
N_ITER="${N_ITER:-5000}"
CORES="${SLURM_CPUS_PER_TASK:-128}"

STRICT_ROOT="results_3way_strict_N50_M245_D19_CC0.9_mh_iter5000"
LOOSE_ROOT="results_3way_loose_N50_M490_D4_CC0.4_mh_iter5000"

for required_file in "$MCMC_WRAPPER" "$PLOTTER" "clust_diam_MCMC.py" "metrics_utils.py"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done

throttle_jobs() {
    while [[ $(jobs -rp | wc -l) -ge $CORES ]]; do
        wait -n || true
    done
}

run_available_random_seeds() {
    local root="$1"
    local seed_dir="$root/seeds_random_layered"
    local output_dir="$root/mcmc_random_layered"
    local seed_file base seed_id output_file
    local launched=0
    local available=0
    local completed=0

    if [[ ! -d "$seed_dir" ]]; then
        echo "Missing random-layered seed directory: $seed_dir" >&2
        return 1
    fi

    mkdir -p "$output_dir"

    mapfile -t seed_files < <(
        find "$seed_dir" -maxdepth 1 -type f -name 'seed_*.json' | sort -V
    )

    available=${#seed_files[@]}

    if (( available == 0 )); then
        echo "No random-layered seed files found in $seed_dir" >&2
        return 1
    fi

    echo "------------------------------------------------"
    echo "Running random-layered MCMC for: $root"
    echo "Available valid random-layered seeds: $available"
    echo "------------------------------------------------"

    for seed_file in "${seed_files[@]}"; do
        base=$(basename "$seed_file")
        seed_id=${base#seed_}
        seed_id=${seed_id%.json}
        output_file="$output_dir/sample_$seed_id.json"

        if [[ -s "$output_file" ]]; then
            continue
        fi

        throttle_jobs
        python3 "$MCMC_WRAPPER" \
            --input_seed "$seed_file" \
            --output_file "$output_file" \
            --n_iter "$N_ITER" &

        launched=$((launched + 1))
    done

    if (( launched > 0 )); then
        wait || true
    fi

    completed=$(
        find "$output_dir" -maxdepth 1 -type f -name 'sample_*.json' | wc -l
    )

    echo "Available seeds: $available"
    echo "Completed random-layered MCMC outputs: $completed"

    if (( completed < available )); then
        echo "Some random-layered MCMC outputs are missing in $output_dir" >&2
        return 1
    fi
}

run_available_random_seeds "$STRICT_ROOT" || exit 1
run_available_random_seeds "$LOOSE_ROOT" || exit 1

echo "Creating three-way figure with all available chains..."
python3 "$PLOTTER" \
    --strict_dir "$STRICT_ROOT" \
    --loose_dir "$LOOSE_ROOT" \
    --out_dir mcmc_plots_3way

echo "DONE."
echo "Plot: mcmc_plots_3way/mcmc_3way_final_ensemble_diversity.pdf"
echo "Summary: mcmc_plots_3way/mcmc_3way_summary.txt"
