#!/bin/bash

# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --job-name=aifs_run          # name of job
#SBATCH -C a100
#SBATCH -A udt@a100
##SBATCH --qos=qos_gpu_a100-dev      # uncomment for fast debug, time<2h
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=8           # number of cores per task (1/4 of the 4-GPUs V100 node)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=10:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=out_2026_may_20-%j.out    # name of output file
#SBATCH --error=out_2026_may_20-%j.out     # name of error file (here, in common with the output file)

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

EXP_NAME="may_2026_heatwave"
DATE_STR="2026052000"

N_MEMBERS=200
VARIABLES="2t 2d sp z t_500 z_500 tp" #variables to save
SAVETIMES="0 6 12 18" #at which times to save variables
LEADS="360 240 360"
BACKTRACKS="240 120 None"
PARIS_INDEX=67391

RUN_IN_PATH="/lustre/fswork/projects/rech/udt/udm13lc/inputs_aifs_ensemble_boosting"
RUN_OUT_PATH="/lustre/fsn1/projects/rech/udt/udm13lc/outputs_aifs_infinite_boosting"
CHECKPOINT="/lustre/fswork/projects/rech/udt/udm13lc/aifs-ens-crps-1.0.ckpt"
MASK="/lustre/fswork/projects/rech/udt/udm13lc/mask_eu.pkl"


# ==========================================
#              RUN AIFS
# ==========================================

python run_aifs_crps_infinite_boosting.py \
	--input_path "$RUN_IN_PATH" \
	--output_path "$RUN_OUT_PATH" \
	--date_str "$DATE_STR" \
	--exp_name "$EXP_NAME" \
	--checkpoint "$CHECKPOINT" \
	--n_members $N_MEMBERS \
	--variables $VARIABLES \
	--savetimes $SAVETIMES \
	--index_boosting $PARIS_INDEX \
	--leads $LEADS \
	--backtracks $BACKTRACKS \
	--mask "$MASK"
