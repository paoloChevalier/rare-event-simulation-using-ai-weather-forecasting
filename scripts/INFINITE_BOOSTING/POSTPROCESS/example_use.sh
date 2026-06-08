#!/bin/bash
# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --partition=zen16
#SBATCH --time=00:20:00
#SBATCH --mem=100G

# ==========================================
#              PREPARE ENV
# ==========================================

module purge
module load pangeo-meso/2026.01.21

conda activate /scratchx/pchevali/python_envs/pangeo-earthkit 

# ==========================================
#          EXPERIMENT PARAMETERS
# ==========================================

EXP_NAME="exp"
DATE_STR="2026051400"
N_MEMBERS=200
N_PHASES=3

REGRID_IN_DIR="/scratchx/pchevali/INFINITE_BOOSTING_CUTOUT/"
REGRID_OUT_DIR="/scratchx/pchevali/INFINITE_BOOSTING_CUTOUT_PROCESSED/"
MASK="/homedata/pchevali/mask_eu.pkl"

# ==========================================
#              RUN POSTPROCESS
# ==========================================

python infinite_boosting_postprocess.py \
    --input_dir "$REGRID_IN_DIR" \
    --output_dir "$REGRID_OUT_DIR" \
    --date_str "$DATE_STR" \
    --exp_name "$EXP_NAME" \
    --n_members $N_MEMBERS \
    --n_phases $N_PHASES \
    --mask "$MASK"
