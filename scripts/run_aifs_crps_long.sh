#!/bin/bash

# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --job-name=aifs_run          # name of job
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
#SBATCH -C a100
#SBATCH -A udt@a100
# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
##SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs V100 node)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=01:30:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=run_aifs%j.out    # name of output file
#SBATCH --error=run_aifs%j.out     # name of error file (here, in common with the output file)

# ==========================================
#              PREPARE ENV
# ==========================================

# Cleans out the modules loaded in interactive and inherited by default 
module purge
module load pytorch-gpu/py3/2.8.0

# Echo of launched commands
set -x

# ==========================================
#          EXPERIMENT PARAMETERS
# ==========================================

EXP_NAME="test_crps_100members"
DATE_STR="2025062000"

N_MEMBERS=3
VARIABLES="2t 10u 10v z_500" #variables to save
SAVETIMES="0 6 12 18" #at which times to save variables
LT=2160

RUN_IN_PATH="/lustre/fsn1/projects/rech/udt/udm13lc/inputs_aifs_ensemble_boosting"
RUN_OUT_PATH="/lustre/fsn1/projects/rech/udt/udm13lc/outputs_aifs_ensemble_boosting"
CHECKPOINT="/lustre/fswork/projects/rech/udt/udm13lc/aifs-ens-crps-1.0.ckpt"

# ==========================================
#              RUN AIFS
# ==========================================

python run_aifs_crps.py \
    --input_path "$RUN_IN_PATH" \
    --output_path "$RUN_OUT_PATH" \
    --date_str "$DATE_STR" \
    --exp_name "$EXP_NAME" \
    --checkpoint "$CHECKPOINT" \
    --n_members $N_MEMBERS \
    --variables $VARIABLES \
    --savetimes $SAVETIMES \
    --lead_time $LT
#    --deterministic
