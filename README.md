# rare-event-simulation-using-ai-weather-forecasting

This repository contains the code, notebooks and scripts supporting my Master's thesis at LSCE (CNRS), supervised by Pascal Yiou and Soulivanh Thao. The thesis evaluates the viability of ECMWF's AI-Enhanced Forecast Systems (AIFS) for studying climate extremes and for performing rare-event simulation algorithms that leverage AI-based forecasts.

This project has two main parts:

- Physical consistency and chaos analysis: evaluates AIFS (and IFS) for physical consistency and chaotic properties.
- Rare-event algorithms: implements and tests rare-event simulation methods (iterative boosting and simple boosting variants) using AIFS forecasts.

## Installation

The notebooks and input/postprocess scripts for the preliminary part and results were run using pangeo_2025.
All the remaining scripts were run on the Jean-Zay supercomputer and need at least a NVIDIA 1100 GPU with flash_attention support.

## Usage

### Notebooks
- Notebooks/ANNEX-lyapunov_exponents_ensemble_forecasts_ecmwf.ipynb: study of Lyapunov exponents in IFS and AIFS to assess chaotic behaviour.
- Notebooks/MAY_2026_HEATWAVE.ipynb: focused analysis of a May 2026 heatwave case study and diagnostics.
- Notebooks/ITERATIVE_BOOSTING_AIFS-Ens.ipynb: iterative boosting experiments using AIFS ensembles. contains functions to compute probabilities and reconstruct trajectories.
- Notebooks/ITERATIVE_BOOSTING_OU.ipynb: iterative-boosting experiments on Ornstein–Uhlenbeck / toy dynamical systems.
- Notebooks/ANNEX-ensemble_forecasts_outputs_aifs_crps_N320.ipynb: annex with outputs and diagnostics for AIFS-CRPS ensemble on N320 grid.
- Notebooks/ANNEX-ensemble_forecasts_outputs_aifs_single.ipynb: annex with outputs and diagnostics for AIFS-Deterministic runs.
- Notebooks/LORENZ63.ipynb: small Lorenz63 experiments and examples used for method illustration.
- Notebooks/ensemble_chaos_tools.py: helper module (used by notebooks) with plotting and chaos diagnostic utilities.

### Scripts
The repository includes several scripts for downloading inputs, running AIFS experiments, and post-processing results. All scripts are organized by experiment type and include example usage files.

- scripts/DL_INPUTS_ENS/
  - aifs_inputs_dl+preprocess.py — download and preprocess ERA5 inputs.
  - example_use.sh — example config.

- scripts/FIRST_EXPERIMENTS_AIFS_SINGLE/
  - runs_aifs_single.py — run AIFS-Deterministic forecasts for prepared input states.
  - DL_INPUTS_SINGLE/aifs_inputs_dl+preprocess+perturbation.py — download + preprocess + add perturbations (single/deterministic input prep).
  - POSTPROCESS/aifs_single_outputs_process.py — convert per-member .pkl outputs to regridded NetCDF and other postprocessing utilities.

- scripts/SIMPLE_BOOSTING/
  - run_aifs_crps_simple_boosting.py — run the simple boosting rare-event algorithm using AIFS-Ens.
  - example_use.sh — config
  - POSTPROCESS/simple_boosting_postprocess.py — postprocessing script; pkl to netcdf per variable.
  - POSTPROCESS/example_use.sh — example config

- scripts/ITERATIVE BOOSTING/
  - run_iterative_boosting.py — iterative boosting script for AIFS-Ens.
  - iterative_boosting_JJA.sh — example config for JJA
  - POSTPROCESS/iterative_boosting_postprocess.py — postprocessing script; pkl to netcdf per variable per phase.
  - POSTPROCESS/run_postprocess.sh — example config

## Data
- Many notebooks and scripts expect local datasets and pickles from experiments.
- ERA5 (available via Copernicus CDS): https://cds.climate.copernicus.eu/
- IFS and AIFS forecasts: ECMWF CDS or via Herbie: https://github.com/blaylockbk/Herbie
- AIFS checkpoints (Hugging Face):
  - AIFS-Deterministic: https://huggingface.co/ecmwf/aifs-single-1.0
  - AIFS-Ensemble (CRPS): https://huggingface.co/ecmwf/aifs-ens-1.0
    
## Acknowledgements

This project was provided with computer and storage resources by GENCI at
IDRIS thanks to the grant 2025-AD01116371 on the supercomputer
Jean Zay's H100 partition.

This study benefited from the IPSL Data and Computing Center ESPRI which is supported by CNRS, SU, CNES and Ecole Polytechnique

## Contributing

Contributions are welcomed.

## License

This repo is licensed under Apache 2.0 for now.
