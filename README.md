# rare-event-simulation-using-ai-weather-forecasting

This repository contains the work from my Master’s thesis at LSCE (CNRS), supervised by Pascal Yiou and Soulivanh Thao. The project evaluates the viability of AIFS from ECMWF for climate extremes. Exploring whether AI-based weather forecasting models (AIFS) can replace traditional dynamical models for simulating extreme climate events. 

This project 

- The first part focuses on the models physical consistency and chaotic properties, assessing the ability of AIFS-Deterministic to be physically consistent and chaotic, looking at energy transfers and using ensemble boosting (perturbed initial conditions) to evaluate Lyapunov exponents and times.
- The second part will aim at performing rare event algorithms using AIFS.

## Installation

The notebooks for the preliminary part and results were run using pangeo_2025.
All the remaining scripts were run on the Jean-Zay supercomputer and need atleast a NVIDIA H100 GPU with flash_attention support.

## Usage

### Notebooks

  - lyapunov_exponents_ensemble_forecasts_ecmwf.ipynb : Preliminary work using ECMWF's IFS and AIFS ensemble weather forecast to study the chaotic behaviour of these systems.
  - ensemble_chaos_tools.py : Various tools for analyzing chaos and dynamical/thermodynamical properties of forecasts, plot of trajectories, plot of logdists and computation of lyapunov exponents, plot of forecast maps. Create a `EnsembleChaos` object to call the different tools.

### Scripts
The repository includes several scripts for downloading inputs, running AIFS experiments, and post-processing results. All scripts are organized by experiment type and include example usage files.

#### Data Download & Preprocessing

- **`scripts/DL_INPUTS_ENS/`** - Download and preprocess inputs for AIFS Ensemble
  - `aifs_inputs_dl+preprocess.py`: Main script for downloading and preprocessing ensemble forecast inputs
  - `example_use.sh`: Example shell script demonstrating usage

#### First Experiments with AIFS Single/Deterministic

- **`scripts/FIRST_EXPERIMENTS_AIFS_SINGLE/`** - Initial experiments using AIFS-Deterministic model
  - `runs_aifs_single.py`: Main script to run AIFS-Deterministic forecasts
  - `example_use.sh`: Example usage
  - **`DL_INPUTS_SINGLE/`** - Input preparation with perturbations
    - `aifs_inputs_dl+preprocess+perturbation.py`: Download, preprocess, and apply perturbations to initial conditions
    - `aifs_inputs_dl+preprocess+perturbation-EDA.py`: Variant for Ensemble Data Assimilation (EDA)
    - `example_use.sh` and `example_use_EDA.sh`: Example usage files
  - **`POSTPROCESS/`** - Post-processing results
    - `aifs_single_outputs_process.py`: Process AIFS-Deterministic outputs

#### Simple Boosting Algorithm

- **`scripts/SIMPLE_BOOSTING/`** - Rare event simulation using simple boosting
  - `run_aifs_crps_simple_boosting.py`: Main script to run simple boosting algorithm with AIFS-CRPS
  - `example_use.sh`: Example usage
  - **`POSTPROCESS/`** - Post-processing results
    - `simple_boosting_postprocess.py`: Process simple boosting results

#### Infinite Boosting Algorithm

- **`scripts/INFINITE_BOOSTING/`** - Rare event simulation using infinite boosting
  - `run_aifs_crps_infinite_boosting.py`: Main script to run infinite boosting algorithm with AIFS-CRPS
  - `example_use.sh`: Example usage
  - **`POSTPROCESS/`** - Post-processing results
    - `infinite_boosting_postprocess.py`: Process infinite boosting results

## Data

The data needed is from IPSL-CM6A-LR and ERA5. Both available online [here](https://esgf-node.ipsl.upmc.fr/search/cmip6-ipsl/) and [here](https://cds.climate.copernicus.eu/).
The IFS and AIFS-Ensemble forecasts are available online [here](https://cds.climate.copernicus.eu/) or using [Herbie](https://github.com/blaylockbk/Herbie).
The AIFS-Deterministic version is available online from [HuggingFace](https://huggingface.co/ecmwf/aifs-single-1.0).
The AIFS-Ensemble(CRPS) version is available online from [HuggingFace](https://huggingface.co/ecmwf/aifs-ens-1.0).

## Acknowledgements

This project was provided with computer and storage resources by GENCI at
IDRIS thanks to the grant 2025-AD01116371 on the supercomputer
Jean Zay's H100 partition.

This study benefited from the IPSL Data and Computing Center ESPRI which is supported by CNRS, SU, CNES and Ecole Polytechnique

## Contributing

Contributions are welcomed.

## License

This repo is licensed under Apache 2.0 for now.
