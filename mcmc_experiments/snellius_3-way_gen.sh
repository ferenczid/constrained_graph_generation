#!/bin/bash
#SBATCH --job-name=comparison_3way
#SBATCH --output=comp3_%j.out
#SBATCH --error=comp3_%j.err
#SBATCH --time=08:00:00
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

# One Python process per allocated CPU core.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ACO_SCRIPT="${ACO_SCRIPT:-aco_seed_gen.py}"
RANDOM_LAYERED_SCRIPT="${RANDOM_LAYERED_SCRIPT:-random_layered_seed_gen.py}"
MCMC_WRAPPER="${MCMC_WRAPPER:-mcmc_wrapper.py}"

SAMPLES="${SAMPLES:-100}"
ACO_SEED_COUNT=$((SAMPLES + 1))
RANDOM_SEED_COUNT=$SAMPLES

N_ITER="${N_ITER:-5000}"
ANTS="${ANTS:-40}"
CONSTRUCTION_ITER="${CONSTRUCTION_ITER:-50}"
CC_TOL="${CC_TOL:-0.025}"

CORES="${SLURM_CPUS_PER_TASK:-128}"
MAX_ACO_SEED_ROUNDS="${MAX_ACO_SEED_ROUNDS:-10}"
MAX_RANDOM_SEED_ROUNDS="${MAX_RANDOM_SEED_ROUNDS:-10}"

for required_file in \
    "$ACO_SCRIPT" \
    "$RANDOM_LAYERED_SCRIPT" \
    "$MCMC_WRAPPER" \
    "clust_diam_MCMC.py" \
    "metrics_utils.py"
do
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

count_missing_seeds() {
    local seed_dir="$1"
    local expected="$2"
    local missing=0
    local i

    for ((i=1; i<=expected; i++)); do
        if [[ ! -s "$seed_dir/seed_$i.json" ]]; then
            missing=$((missing + 1))
        fi
    done

    echo "$missing"
}

generate_missing_seeds() {
    local generator_label="$1"
    local generator_script="$2"
    local seed_dir="$3"
    local expected="$4"
    local n="$5"
    local m="$6"
    local diam="$7"
    local cc="$8"
    local max_rounds="$9"

    local round i missing launched attempt_dir candidate_seed

    mkdir -p "$seed_dir" "$seed_dir/jobs"

    for ((round=1; round<=max_rounds; round++)); do
        missing=$(count_missing_seeds "$seed_dir" "$expected")
        if (( missing == 0 )); then
            echo "Verified all $expected $generator_label seed files in $seed_dir."
            return 0
        fi

        echo "------------------------------------------------"
        echo "$generator_label seed generation round $round"
        echo "Missing before round: $missing / $expected"
        echo "------------------------------------------------"

        launched=0

        for ((i=1; i<=expected; i++)); do
            if [[ -s "$seed_dir/seed_$i.json" ]]; then
                continue
            fi

            attempt_dir="$seed_dir/jobs/job_$i/attempt_$round"
            mkdir -p "$attempt_dir"

            throttle_jobs
            (
                python3 "$generator_script" \
                    --N "$n" \
                    --M "$m" \
                    --diam "$diam" \
                    --diam_min "$diam" \
                    --diam_max "$diam" \
                    --cc "$cc" \
                    --cc_tol "$CC_TOL" \
                    --ants "$ANTS" \
                    --max_iter "$CONSTRUCTION_ITER" \
                    --out_dir "$attempt_dir" \
                    --job_id "$i" \
                    >"$attempt_dir/generation.out" \
                    2>"$attempt_dir/generation.err"
            ) &

            launched=$((launched + 1))
        done

        if (( launched > 0 )); then
            wait || true
        fi

        # Collect successful seeds into one flat directory for the MCMC stage.
        for ((i=1; i<=expected; i++)); do
            if [[ -s "$seed_dir/seed_$i.json" ]]; then
                continue
            fi

            attempt_dir="$seed_dir/jobs/job_$i/attempt_$round"
            candidate_seed="$attempt_dir/seed_$i.json"

            if [[ -s "$candidate_seed" ]]; then
                cp "$candidate_seed" "$seed_dir/seed_$i.json"
            fi
        done

        missing=$(count_missing_seeds "$seed_dir" "$expected")
        echo "$generator_label seeds still missing after round $round: $missing"

        if (( missing == 0 )); then
            echo "Verified all $expected $generator_label seed files in $seed_dir."
            return 0
        fi
    done

    missing=$(count_missing_seeds "$seed_dir" "$expected")
    echo "Unable to produce $missing of $expected $generator_label seeds" \
         "after $max_rounds equal-budget attempts per missing seed." >&2
    return 1
}

run_fixed_seed_chains() {
    local seed_file="$1"
    local output_dir="$2"
    local i

    mkdir -p "$output_dir"

    for ((i=1; i<=SAMPLES; i++)); do
        if [[ -s "$output_dir/sample_$i.json" ]]; then
            continue
        fi

        throttle_jobs
        python3 "$MCMC_WRAPPER" \
            --input_seed "$seed_file" \
            --output_file "$output_dir/sample_$i.json" \
            --n_iter "$N_ITER" &
    done

    wait || true
}

run_multiseed_chains() {
    local seed_dir="$1"
    local output_dir="$2"
    local seed_offset="$3"
    local i seed_idx

    mkdir -p "$output_dir"

    for ((i=1; i<=SAMPLES; i++)); do
        if [[ -s "$output_dir/sample_$i.json" ]]; then
            continue
        fi

        seed_idx=$((i + seed_offset))

        if [[ ! -s "$seed_dir/seed_$seed_idx.json" ]]; then
            echo "Missing input seed: $seed_dir/seed_$seed_idx.json" >&2
            return 1
        fi

        throttle_jobs
        python3 "$MCMC_WRAPPER" \
            --input_seed "$seed_dir/seed_$seed_idx.json" \
            --output_file "$output_dir/sample_$i.json" \
            --n_iter "$N_ITER" &
    done

    wait || true
}

check_outputs() {
    local output_dir="$1"
    local expected="$2"
    local missing=0
    local i

    for ((i=1; i<=expected; i++)); do
        if [[ ! -s "$output_dir/sample_$i.json" ]]; then
            echo "Missing output: $output_dir/sample_$i.json" >&2
            missing=$((missing + 1))
        fi
    done

    if (( missing > 0 )); then
        echo "Aborting treatment: $missing MCMC outputs are missing from $output_dir." >&2
        return 1
    fi

    echo "Verified $expected MCMC outputs in $output_dir."
}

write_experiment_metadata() {
    local out_dir="$1"
    local label="$2"
    local n="$3"
    local m="$4"
    local diam="$5"
    local cc="$6"
    local random_ready="$7"

    cat > "$out_dir/experiment_metadata.txt" <<EOF
label=$label
N=$n
M=$m
diam_min=$diam
diam_max=$diam
target_cc=$cc
cc_tol=$CC_TOL
samples_per_treatment=$SAMPLES
mcmc_proposals_per_chain=$N_ITER
construction_ants=$ANTS
construction_iterations=$CONSTRUCTION_ITER
candidate_budget_per_seed_attempt=$((ANTS * CONSTRUCTION_ITER))
max_aco_seed_rounds=$MAX_ACO_SEED_ROUNDS
max_random_seed_rounds=$MAX_RANDOM_SEED_ROUNDS
random_layered_treatment_ready=$random_ready

Treatments:
1. mcmc_fixed: all chains start from ACO seed_1.json
2. mcmc_multiseed: chains start from ACO seeds 2 through $((SAMPLES + 1))
3. mcmc_random_layered: chains start from random-layered seeds 1 through $SAMPLES
EOF
}

run_experiment() {
    local label="$1"
    local n="$2"
    local m="$3"
    local diam="$4"
    local cc="$5"
    local out_dir="$6"

    local aco_seed_dir="$out_dir/seeds_aco"
    local random_seed_dir="$out_dir/seeds_random_layered"
    local fixed_dir="$out_dir/mcmc_fixed"
    local aco_multiseed_dir="$out_dir/mcmc_multiseed"
    local random_multiseed_dir="$out_dir/mcmc_random_layered"
    local random_ready="no"

    mkdir -p "$out_dir"

    echo "================================================"
    echo "RUNNING $label THREE-WAY EXPERIMENT"
    echo "N=$n M=$m D=$diam CC=$cc"
    echo "Construction budget: $ANTS x $CONSTRUCTION_ITER candidates per attempt"
    echo "MCMC budget: $N_ITER proposals per chain"
    echo "================================================"

    # ACO needs one fixed seed plus one independent seed per multiseed chain.
    generate_missing_seeds \
        "ACO" \
        "$ACO_SCRIPT" \
        "$aco_seed_dir" \
        "$ACO_SEED_COUNT" \
        "$n" "$m" "$diam" "$cc" \
        "$MAX_ACO_SEED_ROUNDS" || {
            echo "ACO did not produce the required seeds; cannot run this experiment." >&2
            return 1
        }

    # Random layered construction needs one independent seed per chain.
    # Failure is scientifically informative, so the ACO treatments still run.
    if generate_missing_seeds \
        "RANDOM_LAYERED" \
        "$RANDOM_LAYERED_SCRIPT" \
        "$random_seed_dir" \
        "$RANDOM_SEED_COUNT" \
        "$n" "$m" "$diam" "$cc" \
        "$MAX_RANDOM_SEED_ROUNDS"
    then
        random_ready="yes"
    else
        echo "Random-layered construction did not produce all required seeds."
        echo "The random-layered MCMC treatment will be skipped for $label."
    fi

    write_experiment_metadata \
        "$out_dir" "$label" "$n" "$m" "$diam" "$cc" "$random_ready"

    echo "Running fixed-ACO-seed MCMC chains..."
    run_fixed_seed_chains "$aco_seed_dir/seed_1.json" "$fixed_dir"
    check_outputs "$fixed_dir" "$SAMPLES" || return 1

    echo "Running multiple-ACO-seed MCMC chains..."
    run_multiseed_chains "$aco_seed_dir" "$aco_multiseed_dir" 1
    check_outputs "$aco_multiseed_dir" "$SAMPLES" || return 1

    if [[ "$random_ready" == "yes" ]]; then
        echo "Running multiple-random-layered-seed MCMC chains..."
        run_multiseed_chains "$random_seed_dir" "$random_multiseed_dir" 0
        check_outputs "$random_multiseed_dir" "$SAMPLES" || return 1
    fi

    echo "Completed $label experiment."
    echo "Results directory: $out_dir"
}

N1=50
M1=245
DIAM1=19
CC1=0.9
OUT1="results_3way_strict_N${N1}_M${M1}_D${DIAM1}_CC${CC1}_mh_iter${N_ITER}"

N2=50
M2=490
DIAM2=4
CC2=0.4
OUT2="results_3way_loose_N${N2}_M${M2}_D${DIAM2}_CC${CC2}_mh_iter${N_ITER}"

run_experiment "STRICT" "$N1" "$M1" "$DIAM1" "$CC1" "$OUT1" || exit 1
run_experiment "LOOSE" "$N2" "$M2" "$DIAM2" "$CC2" "$OUT2" || exit 1

echo "DONE."
echo "Strict results: $OUT1"
echo "Loose results:  $OUT2"
