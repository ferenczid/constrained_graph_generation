#!/bin/bash
#SBATCH --job-name=aco_scaling
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -u

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
source $HOME/my_venvs/aco_env/bin/activate

# Important when launching many independent Python jobs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Run from the directory where sbatch was submitted.
cd "$SLURM_SUBMIT_DIR"

# -------------------------------------------------------------------
# Usage:
#   mkdir -p logs
#   sbatch snellius_scaling_ore.sh 50
#   sbatch snellius_scaling_ore.sh 100
#   sbatch snellius_scaling_ore.sh 200
#
# Smoke test:
#   TRIALS=2 CC_STEP=0.25 MAX_JOBS=20 sbatch snellius_scaling_ore.sh 50
# -------------------------------------------------------------------

N=${1:-50}

# Default benchmark settings.
DENSITIES=(${DENSITIES:-0.10 0.20 0.40 0.60 0.80})
TRIALS=${TRIALS:-30}
ANTS=${ANTS:-40}
MAX_ITER=${MAX_ITER:-50}

# Original/default ACO parameters.
B=${B:-2.0}
H=${H:-0.5}
RHO=${RHO:-0.90}

# Exact validation settings.
CC_TOL=${CC_TOL:-0.025}
CC_STEP=${CC_STEP:-0.05}

# Parallel jobs per node. Keep below 192.
MAX_JOBS=${MAX_JOBS:-180}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="scaling_ore_N${N}_${TIMESTAMP}"

mkdir -p "$OUT_DIR/metrics"
mkdir -p "$OUT_DIR/graphs"
mkdir -p "$OUT_DIR/logs"

FINAL_CSV="$OUT_DIR/final_results.csv"
RUN_LOG="$OUT_DIR/run_log.txt"

echo "N,M,Density,Diam,Diam_Min,Diam_Max,Target_Trans,CC_Tol,Trials,Wins,Success,Gap,Runtime_Total,Runtime_Per_Trial,Exact_Checked,Exact_Valid,Exact_Valid_Fraction,Mean_Exact_Diam,Mean_Exact_CC,B,H,Rho,Ants,Max_Iter" > "$FINAL_CSV"

{
    echo "Starting ACO scaling experiment"
    echo "Host: $(hostname)"
    echo "Submit dir: $SLURM_SUBMIT_DIR"
    echo "N: $N"
    echo "Densities: ${DENSITIES[*]}"
    echo "Trials: $TRIALS"
    echo "Ants: $ANTS"
    echo "Max iterations: $MAX_ITER"
    echo "B: $B"
    echo "H: $H"
    echo "Rho: $RHO"
    echo "CC tolerance: $CC_TOL"
    echo "CC step: $CC_STEP"
    echo "Max jobs: $MAX_JOBS"
    echo "Output directory: $OUT_DIR"
    python --version
} | tee "$RUN_LOG"

for P in "${DENSITIES[@]}"; do

    M=$(python3 - <<EOF
n = $N
p = float("$P")
print(int(round(p * n * (n - 1) / 2)))
EOF
)

    ORE_DMAX=$(python3 - <<EOF
import math
n = $N
m = $M

def mmax(n, d):
    # Ore extremal edge bound for graphs of order n and diameter at least d.
    return math.comb(n - d + 2, 2) + d - 3

feasible = [d for d in range(2, n) if m <= mmax(n, d)]
print(max(feasible) if feasible else 1)
EOF
)

    # Adaptive diameter resolution.
    if [ "$ORE_DMAX" -le 30 ]; then
        D_STEP=1
    elif [ "$ORE_DMAX" -le 80 ]; then
        D_STEP=2
    else
        D_STEP=5
    fi

    {
        echo "------------------------------------------------------------"
        echo "Density P=$P"
        echo "M=$M"
        echo "Ore feasible maximum diameter: $ORE_DMAX"
        echo "Using D_STEP=$D_STEP"
    } | tee -a "$RUN_LOG"

    if [ "$ORE_DMAX" -lt 2 ]; then
        echo "Skipping P=$P because no feasible D>=2 exists." | tee -a "$RUN_LOG"
        continue
    fi

    for (( d=2; d<=ORE_DMAX; d+=D_STEP )); do

        for cc in $(seq 0.00 "$CC_STEP" 1.00); do

            while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                wait -n
            done

            python3 aco_single_run.py \
                --N "$N" \
                --M "$M" \
                --trials "$TRIALS" \
                --diam "$d" \
                --diam_min "$d" \
                --diam_max "$((d + 1))" \
                --cc "$cc" \
                --cc_tol "$CC_TOL" \
                --ants "$ANTS" \
                --max_iter "$MAX_ITER" \
                --rho "$RHO" \
                --B "$B" \
                --H "$H" \
                --out_dir "$OUT_DIR" &

        done

        echo "Submitted jobs for N=$N, P=$P, M=$M, D=$d" | tee -a "$RUN_LOG"
    done
done

wait

echo "All simulations complete." | tee -a "$RUN_LOG"

if compgen -G "$OUT_DIR/metrics/*.csv" > /dev/null; then
    cat "$OUT_DIR/metrics"/*.csv >> "$FINAL_CSV"
else
    echo "Warning: no metric files generated." | tee -a "$RUN_LOG"
fi

echo "Merged results written to $FINAL_CSV" | tee -a "$RUN_LOG"
echo "Done." | tee -a "$RUN_LOG"
