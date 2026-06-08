#!/bin/bash

# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --partition=zen4
#SBATCH --time=2:30:00
#SBATCH --mem=64G

# ==========================================
#              PREPARE ENV
# ==========================================

module purge
module load pangeo-meso/2026.01.21

conda activate /scratchx/pchevali/python_envs/pangeo-earthkit 

# ==========================================
#          EXPERIMENT PARAMETERS
# ==========================================

EXP_NAME="test_perturbation_deterministic"
DATE_INPUT="2025-06-25 00:00:00"
INPUTS_SAVE="/homedata/pchevali/AIFS_INPUTS/"

# ==========================================
#              RUN PREPROCESS
# ==========================================

python /home/pchevali/Stage2026/scripts_aifs_ens_boosting/aifs_inputs_dl+preprocess+perturbation-EDA.py \
    --date_input "$DATE_INPUT" \
    --exp_name "$EXP_NAME" \
    --output_save "$INPUTS_SAVE"
