#!/bin/bash
# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --partition=zen16
#SBATCH --time=00:30:00
#SBATCH --mem=240G

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
DATE_STR="2025062000"
N_MEMBERS=128

REGRID_IN_DIR="/scratchx/pchevali/AIFS_OUTPUTS/"
REGRID_OUT_DIR="/scratchx/pchevali/AIFS_OUTPUTS_REGRIDDED/"
MASK="/homedata/pchevali/mask_eu.pkl"


# ==========================================
#              RUN POSTPROCESS
# ==========================================

python aifs_outputs_process_crps.py \
    --input_dir "$REGRID_IN_DIR" \
    --output_dir "$REGRID_OUT_DIR" \
    --date_str "$DATE_STR" \
    --exp_name "$EXP_NAME" \
    --n_members $N_MEMBERS \
    --mask "$MASK"
