#!/bin/bash

# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --job-name=aifs_run
#SBATCH -C a100
#SBATCH -A udt@a100
##SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:59
#SBATCH --output=out-%j.out
#SBATCH --error=out-%j.out

# ==========================================
#              PREPARE ENV
# ==========================================

module purge
module load pytorch-gpu/py3/2.8.0

set -x

# ==========================================
#          EXPERIMENT PARAMETERS
# ==========================================

EXP_NAME="return_levels_JJA"
DATE_STRS="2019071400 2019071300 2019071200 1947071700 1947071600 1947071500 1952062000 1952061900 1952061800 1990072400 1990072300 1990072200 2003073000 2003072900 2003072800 1957062500 1957062400 1957062300 2022070800 2022070700 2022070600 1976062200 1976062100 1976062000 1975072400 1975072300 1975072200 2015062000 2015061900 2015061800 2020072000 2020071900 2020071800 1949070200 1949070100 1949063000 1943080700 1943080600 1943080500 1998073100 1998073000 1998072900 2025062000 2025061900 2025061800 2026061300 2026061200 2026061100"
N_MEMBERS=576

# Surface variables
SURF_VARS="2t 2d sp tp"
# Temperature
T_VARS="t_250 t_300 t_400 t_500 t_600 t_700 t_850 t_925 t_1000"
# Geopotential
Z_VARS="z_500"
# U-component of wind
U_VARS="u_250 u_300 u_400 u_500 u_600 u_700 u_850 u_925 u_1000"
# V-component of wind
V_VARS="v_250 v_300 v_400 v_500 v_600 v_700 v_850 v_925 v_1000"
# Vertical velocity (omega)
W_VARS="w_250 w_300 w_400 w_500 w_600 w_700 w_850 w_925 w_1000"

VARIABLES="$SURF_VARS $T_VARS $Z_VARS $U_VARS $V_VARS $W_VARS" #variables to save

SAVETIMES="0 6 12 18"
FORWARDS="360 240 360" # f_j
LEADS="240 120 None"   # t_j
PARIS_INDEX=67391
q=0.05

RUN_IN_PATH="/lustre/fswork/projects/rech/udt/udm13lc/inputs_aifs_ensemble_boosting"
RUN_OUT_PATH="/lustre/fsn1/projects/rech/udt/udm13lc/outputs_aifs_iterative_boosting"
CHECKPOINT="/lustre/fswork/projects/rech/udt/udm13lc/aifs-ens-crps-1.0.ckpt"
MASK="/lustre/fswork/projects/rech/udt/udm13lc/mask_eu.pkl"

# ==========================================
#              RUN AIFS
# ==========================================

python run_iterative_boosting.py \
	--input_path "$RUN_IN_PATH" \
	--output_path "$RUN_OUT_PATH" \
	--date_strs $DATE_STRS \
	--exp_name "$EXP_NAME" \
	--checkpoint "$CHECKPOINT" \
	--n_members $N_MEMBERS \
	--variables $VARIABLES \
	--savetimes $SAVETIMES \
	--index_boosting $PARIS_INDEX \
	--forwards $FORWARDS \
	--leads $LEADS \
	--mask "$MASK" \
	--q $q
