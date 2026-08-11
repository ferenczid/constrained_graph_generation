#!/bin/bash
#SBATCH --job-name=aco_BH_sens
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -u

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
source $HOME/my_venvs/aco_env/bin/activate

# We parallelize by launching many independent Python processes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# Usage:
#   mkdir -p logs
#   TRIALS=2 MAX_JOBS=20 sbatch snellius_BH_sensitivity.sh
#   TRIALS=30 MAX_JOBS=180 sbatch snellius_BH_sensitivity.sh

TRIALS=${TRIALS:-30}
ANTS=${ANTS:-40}
MAX_ITER=${MAX_ITER:-50}
RHO=${RHO:-0.90}
CC_TOL=${CC_TOL:-0.025}
MAX_JOBS=${MAX_JOBS:-180}

B_VALUES=(1.0 1.5 2.0 3.0 4.0)
H_VALUES=(0.10 0.25 0.50 0.75 1.00)

SETTINGS=(
  # Low density: broad diagonal feasible band
  "100 0.10 12 0.20 p010_easy_lowD"
  "100 0.10 24 0.65 p010_boundary_midD"
  "100 0.10 36 0.80 p010_hard_highD"

  # Sparse/moderate density
  "100 0.20 6 0.35 p020_easy_lowD"
  "100 0.20 12 0.70 p020_boundary_midD"
  "100 0.20 22 0.90 p020_hard_highD"

  # Medium density
  "100 0.40 2 0.60 p040_easy_lowD"
  "100 0.40 8 0.85 p040_boundary_midD"
  "100 0.40 12 0.90 p040_hard_highD"

  # Dense
  "100 0.60 2 0.70 p060_easy_lowD"
  "100 0.60 6 0.85 p060_boundary_midD"
  "100 0.60 9 0.95 p060_hard_highD"

  # Very dense
  "100 0.80 2 0.85 p080_easy_lowD"
  "100 0.80 4 0.90 p080_boundary_midD"
  "100 0.80 6 0.95 p080_hard_highD"
)










TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="BH_sensitivity_${TIMESTAMP}"

mkdir -p "$OUT_DIR/metrics"
mkdir -p "$OUT_DIR/graphs"

FINAL_CSV="$OUT_DIR/final_results.csv"
RUN_LOG="$OUT_DIR/run_log.txt"

HEADER="N,M,Density,Diam,Diam_Min,Diam_Max,Target_Trans,CC_Tol,Trials,Wins,Success,Gap,Runtime_Total,Runtime_Per_Trial,Exact_Checked,Exact_Valid,Exact_Valid_Fraction,Mean_Exact_Diam,Mean_Exact_CC,B,H,Rho,Ants,Max_Iter,Setting_Label"
echo "$HEADER" > "$FINAL_CSV"

{
    echo "Starting B,H sensitivity experiment"
    echo "Host: $(hostname)"
    echo "Submit dir: $SLURM_SUBMIT_DIR"
    echo "Trials: $TRIALS"
    echo "Ants: $ANTS"
    echo "Max iterations: $MAX_ITER"
    echo "Rho: $RHO"
    echo "CC tolerance: $CC_TOL"
    echo "Max jobs: $MAX_JOBS"
    echo "B values: ${B_VALUES[*]}"
    echo "H values: ${H_VALUES[*]}"
    echo "Output directory: $OUT_DIR"
    python --version
} | tee "$RUN_LOG"

for setting in "${SETTINGS[@]}"; do
    read -r N P D CC LABEL <<< "$setting"

    M=$(python3 - <<EOF
n = int("$N")
p = float("$P")
print(int(round(p * n * (n - 1) / 2)))
EOF
)

    echo "------------------------------------------------------------" | tee -a "$RUN_LOG"
    echo "Setting: $LABEL | N=$N P=$P M=$M D=$D C=$CC" | tee -a "$RUN_LOG"

    for B in "${B_VALUES[@]}"; do
        for H in "${H_VALUES[@]}"; do

            while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                wait -n
            done

            python3 aco_bh_sensitivity.py \
                --N "$N" \
                --M "$M" \
                --trials "$TRIALS" \
                --diam "$D" \
                --diam_min "$D" \
                --diam_max "$((D + 1))" \
                --cc "$CC" \
                --cc_tol "$CC_TOL" \
                --ants "$ANTS" \
                --max_iter "$MAX_ITER" \
                --rho "$RHO" \
                --B "$B" \
                --H "$H" \
                --setting_label "$LABEL" \
                --out_dir "$OUT_DIR" \
                --no_save_graphs &

        done
    done
done

wait

echo "All B,H sensitivity runs complete." | tee -a "$RUN_LOG"

if compgen -G "$OUT_DIR/metrics/*.csv" > /dev/null; then
    cat "$OUT_DIR/metrics"/*.csv >> "$FINAL_CSV"
else
    echo "Warning: no metric files generated." | tee -a "$RUN_LOG"
fi

echo "Merged results written to $FINAL_CSV" | tee -a "$RUN_LOG"
echo "Done." | tee -a "$RUN_LOG"
