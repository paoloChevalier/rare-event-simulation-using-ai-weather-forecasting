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

## Data

The data needed is from IPSL-CM6A-LR and ERA5. Both available online [here](https://esgf-node.ipsl.upmc.fr/search/cmip6-ipsl/) and [here](https://cds.climate.copernicus.eu/).
The IFS and AIFS-Ensemble(Diffusion) forecasts are available online [here](https://cds.climate.copernicus.eu/) or using [Herbie](https://github.com/blaylockbk/Herbie).
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
