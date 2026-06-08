#!/bin/bash

# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --partition=zen4
#SBATCH --time=0:30:00
#SBATCH --mem=32G

# ==========================================
#              PREPARE ENV
# ==========================================

module purge
module load pangeo-meso/2026.01.21

conda activate /scratchx/pchevali/python_envs/pangeo-earthkit 

# ==========================================
#          EXPERIMENT PARAMETERS
# ==========================================

EXP_NAME="boosting_2025_recent_heatwave"
DATE_INPUT="2025-06-20 00:00:00"

PERT_TYPE="uniform"
N_MEMBERS=50
#commas mean together and spaces separate experiments
SCALES="1e-3 1e-6"
# for example this will run 3 experiments: 'q' alone, 'u' and 'v' together, and all three together
PERT_FIELDS="q u,v q,u,v"

INPUTS_SAVE="/homedata/pchevali/AIFS_INPUTS/"

# ==========================================
#              RUN PREPROCESS
# ==========================================

python /home/pchevali/Stage2026/scripts_aifs_ens_boosting/aifs_inputs_dl+preprocess+perturbation.py \
    --date_input "$DATE_INPUT" \
    --n_members $N_MEMBERS \
    --perturbed_fields $PERT_FIELDS \
    --scales $SCALES \
    --exp_name "$EXP_NAME" \
    --pert_type "$PERT_TYPE" \
    --output_save "$INPUTS_SAVE"
