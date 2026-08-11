#!/bin/bash
#SBATCH --job-name=comparison_study
#SBATCH --output=logs/comp_%j.out
#SBATCH --error=logs/comp_%j.err
#SBATCH --time=04:00:00  # Give it enough time for both runs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0  
source $HOME/my_venvs/aco_env/bin/activate

mkdir -p logs


N1=40
M1=78
DIAM1=12
CC1=0.35
SAMPLES=100
CORES=128

OUT1="results_strict_N${N1}_M${M1}_D${DIAM1}_CC${CC1}"
mkdir -p "$OUT1/seeds" "$OUT1/mcmc_fixed" "$OUT1/mcmc_varied"

echo "------------------------------------------------"
echo "RUNNING STRICT EXPERIMENT (Isomers Expected)"
echo "N=$N1 M=$M1 D=$DIAM1 CC=$CC1"
echo "------------------------------------------------"

# 1. Generate Seeds (101 total)
for (( i=1; i<=101; i++ )); do
    while [ $(jobs -r | wc -l) -ge $CORES ]; do wait -n; done
    python3 aco_gen.py --N $N1 --M $M1 --diam $DIAM1 --cc $CC1 --out_dir "$OUT1/seeds" --job_id $i &
done
wait

# 2. Fixed MCMC (Local)
FIXED_SEED="$OUT1/seeds/seed_1.json"
for (( i=1; i<=SAMPLES; i++ )); do
    while [ $(jobs -r | wc -l) -ge $CORES ]; do wait -n; done
    python3 mcmc_wrapper.py --input_seed "$FIXED_SEED" --output_file "$OUT1/mcmc_fixed/sample_$i.json" --swaps 50000 &
done
wait

# 3. Varied MCMC (Global)
for (( i=1; i<=SAMPLES; i++ )); do
    while [ $(jobs -r | wc -l) -ge $CORES ]; do wait -n; done
    # Use seeds 2..101
    SEED_IDX=$((i + 1))
    python3 mcmc_wrapper.py --input_seed "$OUT1/seeds/seed_$SEED_IDX.json" --output_file "$OUT1/mcmc_varied/sample_$i.json" --swaps 50000 &
done
wait



N2=40
M2=195
DIAM2=4
CC2=0.4
SAMPLES=100

OUT2="results_loose_N${N2}_M${M2}_D${DIAM2}_CC${CC2}"
mkdir -p "$OUT2/seeds" "$OUT2/mcmc_fixed" "$OUT2/mcmc_varied"

echo "------------------------------------------------"
echo "RUNNING LOOSE EXPERIMENT (Mixing Expected)"
echo "N=$N2 M=$M2 D=$DIAM2 CC=$CC2"
echo "------------------------------------------------"

# 1. Generate Seeds
for (( i=1; i<=101; i++ )); do
    while [ $(jobs -r | wc -l) -ge $CORES ]; do wait -n; done
    python3 aco_gen.py --N $N2 --M $M2 --diam $DIAM2 --cc $CC2 --out_dir "$OUT2/seeds" --job_id $i &
done
wait

# 2. Fixed MCMC
FIXED_SEED="$OUT2/seeds/seed_1.json"
for (( i=1; i<=SAMPLES; i++ )); do
    while [ $(jobs -r | wc -l) -ge $CORES ]; do wait -n; done
    python3 mcmc_wrapper.py --input_seed "$FIXED_SEED" --output_file "$OUT2/mcmc_fixed/sample_$i.json" --swaps 20000 &
done
wait

# 3. Varied MCMC
for (( i=1; i<=SAMPLES; i++ )); do
    while [ $(jobs -r | wc -l) -ge $CORES ]; do wait -n; done
    SEED_IDX=$((i + 1))
    python3 mcmc_wrapper.py --input_seed "$OUT2/seeds/seed_$SEED_IDX.json" --output_file "$OUT2/mcmc_varied/sample_$i.json" --swaps 20000 &
done
wait

echo "DONE. Results in $OUT1 and $OUT2"
