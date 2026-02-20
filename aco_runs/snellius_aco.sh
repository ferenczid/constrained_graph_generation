#!/bin/bash
#SBATCH --job-name=aco_graph_gen     # Job name
#SBATCH --output=logs/%x_%j.out      # Standard output log (make sure 'logs' folder exists)
#SBATCH --error=logs/%x_%j.err       # Standard error log
#SBATCH --time=02:15:00              # Time limit (HH:MM:SS) - Adjust as needed
#SBATCH --partition=genoa            # Partition (Genoa nodes have 192 cores on SURF)
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks=1                   # Run 1 main script
#SBATCH --cpus-per-task=192          # Request all 192 cores on that node



module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0  
source $HOME/my_venvs/aco_env/bin/activate


python --version


N=40
M=585
TRIALS=100
MAX_JOBS=190
OUT_DIR="results_N${N}_M${M}_$(date +%Y%m%d_%H%M%S)"


mkdir -p logs
mkdir -p "$OUT_DIR/metrics"
mkdir -p "$OUT_DIR/graphs"

echo "Starting Experiment on $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Output Directory: $OUT_DIR"

DIAM_MAX=$((N / 2))
# 1. Loop Diameter
for (( d=2; d<=DIAM_MAX; d++ )); do
    
    # 2. Loop Clustering Coefficient (0.00 to 1.00)
    for cc in $(seq 0.00 0.01 1.00); do
        
        
        while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
            wait -n
        done


        python3 aco_single_run.py \
            --N $N \
            --M $M \
            --trials $TRIALS \
            --diam $d \
            --cc $cc \
            --out_dir "$OUT_DIR" &
            
    done
    echo "Submitted batch for Diameter $d..."
done


wait

echo "All simulations complete."

# Merge Results
FINAL_CSV="$OUT_DIR/final_results.csv"
echo "Diam,Target_Trans,Success,Gap,Jaccard,Spectral" > "$FINAL_CSV"
cat "$OUT_DIR/metrics"/*.csv >> "$FINAL_CSV"

echo "Data merged to $FINAL_CSV"
