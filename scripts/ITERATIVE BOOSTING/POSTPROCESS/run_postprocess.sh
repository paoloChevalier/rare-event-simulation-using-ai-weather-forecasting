#!/bin/bash
# ==========================================
#              SBATCH SETTINGS
# ==========================================

#SBATCH --partition=zen16
#SBATCH --time=02:30:00
#SBATCH --mem=250G

# ==========================================
#              PREPARE ENV
# ==========================================

module purge
module load pangeo-meso/2026.01.21

conda activate /scratchx/pchevali/python_envs/pangeo-earthkit 

# ==========================================
#          EXPERIMENT PARAMETERS
# ==========================================

EXP_NAME="return_levels_JJA_run2"
N_MEMBERS=480
N_PHASES=3

REGRID_IN_DIR="/scratchx/pchevali/ITERATIVE_BOOSTING/"
REGRID_OUT_DIR="/scratchx/pchevali/ITERATIVE_BOOSTING_PROCESSED/"
MASK="/homedata/pchevali/mask_eu.pkl"

# ==========================================
#              RUN POSTPROCESS
# ==========================================

python iterative_boosting_postprocess.py \
	 --input_dir "$REGRID_IN_DIR" \
	 --output_dir "$REGRID_OUT_DIR" \
	 --exp_name "$EXP_NAME" \
	 --n_members $N_MEMBERS \
	 --n_phases $N_PHASES \
	 --mask "$MASK"
