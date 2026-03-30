#!/bin/bash

# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --partition=zen16
#SBATCH --time=1:30:00
#SBATCH --mem=200G

# ==========================================
#              PREPARE ENV
# ==========================================

module purge
module load pangeo-meso/2026.01.21

conda activate /scratchx/pchevali/python_envs/pangeo-earthkit 

# ==========================================
#          EXPERIMENT PARAMETERS
# ==========================================

EXP_NAME="tests_reproducibility"
DATE_STR="2025062500"
N_MEMBERS=5

REGRID_IN_DIR="/homedata/pchevali/AIFS_OUTPUTS/"
REGRID_OUT_DIR="/homedata/pchevali/AIFS_OUTPUTS_REGRIDDED/"

# ==========================================
#              RUN POSTPROCESS
# ==========================================

python aifs_outputs_process.py \
    --input_dir "$REGRID_IN_DIR" \
    --output_dir "$REGRID_OUT_DIR" \
    --date_str "$DATE_STR" \
    --exp_name "$EXP_NAME" \
    --n_members $N_MEMBERS
