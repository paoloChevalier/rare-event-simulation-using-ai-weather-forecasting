import datetime
from collections import defaultdict

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr

from tqdm import tqdm
from ecmwf.opendata import Client as OpendataClient

import copy
import os
import pickle
import multiurl #in some versions of earthkit they forgot to import it -_-
import multiurl.base #in some versions of earthkit they forgot to import it -_-

os.environ["CDSAPI_RC"] = "/home/pchevali/.cdsapirc"
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["stl1", "stl2", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]
PARAM_PL = ["z", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

def get_open_data(target_date, param, levelist=[]):
    """Fetches ERA5 reanalysis data from CDS for a target date and 6 hours before.

    Retrieves data on a 0.25-degree grid and interpolates it to the N320 Gaussian
    grid. For each parameter, the two time steps (t-6h and t) are stacked into
    a single array.
    
    Args:
        target_date (datetime.datetime): The target date and time for which to
            retrieve data. Data is also fetched 6 hours before this date.
        param (list[str]): List of parameter short names to retrieve
            (e.g. ``["10u", "10v", "2t"]``).
        levelist (list[int], optional): Pressure levels in hPa to retrieve.
            If non-empty, fetches from the pressure-level dataset; otherwise
            fetches from the single-level dataset. Defaults to ``[]``.
    
    Returns:
        dict[str, np.ndarray]: A mapping from parameter name to a stacked NumPy
            array of shape ``(2, N320_lat, N320_lon)``. Pressure-level parameters
            are keyed as ``"<param>_<level>"`` (e.g. ``"z_500"``), while
            surface parameters use the short name directly (e.g. ``"2t"``).
    """
    fields = defaultdict(list)
    # CDS separates surface and pressure level datasets
    dataset = "reanalysis-era5-pressure-levels" if levelist else "reanalysis-era5-single-levels"
    # Get the data for the current date and the previous date
    for date in [target_date - datetime.timedelta(hours=6), target_date]:
        request = {
            "product_type": "reanalysis",
            "param": param,
            "date": date.strftime("%Y-%m-%d"),
            "time": date.strftime("%H:%M"),
            "grid": [0.25, 0.25],
        }
        
        if levelist:
            request["pressure_level"] = levelist

        data = ekd.from_source("cds", dataset, **request)
        
        for f in data:
            assert f.to_numpy().shape == (721, 1440)
            values = f.to_numpy()
            
            # Interpolate the data from 0.25 to N320
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            
            # Add the values to the list
            name = (
                f"{f.metadata('param')}_{f.metadata('levelist')}"
                if levelist
                else f.metadata("param")
            )
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields

def build_initial_state(target_date):
    """Builds a complete initial model state by fetching all required ERA5 fields.

    Combines surface, soil, and pressure-level parameters into a single state
    dictionary. The fields are fetched at ``target_date`` and 6 hours before,
    and interpolated to the N320 Gaussian grid.
    
    Args:
        target_date (datetime.datetime): The target date and time for the
            initial state.
    
    Returns:
        dict: A state dictionary with the following keys:
    
            - ``"date"`` (datetime.datetime): The target date.
            - ``"fields"`` (dict[str, np.ndarray]): All fetched fields, keyed
              by parameter name (surface/soil) or ``"<param>_<level>"``
              (pressure-level). Each value is an array of shape
              ``(2, N320_lat, N320_lon)``.
    """
    fields = {}
    
    fields.update(get_open_data(target_date, param=PARAM_SFC))
    fields.update(get_open_data(target_date, param=PARAM_SOIL))
    fields.update(get_open_data(target_date, param=PARAM_PL, levelist=LEVELS))

    return dict(date=target_date, fields=fields)

def gaussian_pertubation(state, fields_to_perturbate=["q", "u", "v"], scale=1e-13):
    """Applies multiplicative Gaussian noise to selected fields of a model state.

    Creates a deep copy of the input state and perturbs the specified fields
    by multiplying each value by ``(1 + N(0, scale))``.
    
    Args:
        state (dict): A state dictionary as returned by ``build_initial_state``,
            containing a ``"fields"`` key mapping parameter names to NumPy arrays.
        fields_to_perturbate (list[str], optional): List of base variable names
            to perturb (e.g. ``["q", "u", "v"]``). Both exact key matches and
            pressure-level keys (e.g. ``"q_500"``) are perturbed. Defaults to
            ``["q", "u", "v"]``.
        scale (float, optional): Standard deviation of the Gaussian noise.
            Defaults to ``1e-13``.
    
    Returns:
        dict: A deep copy of ``state`` with the specified fields perturbed.
    """
    perturbed = copy.deepcopy(state)
    for key, arrays in perturbed["fields"].items():
        # extract the base variable name (e.g., 'q' from 'q_500')
        base_var = key.split("_")[0]
        # perturbate only what we want to
        if key in fields_to_perturbate or base_var in fields_to_perturbate:
            noise = np.random.normal(loc=0.0, scale=scale, size=arrays.shape)
            perturbed["fields"][key] = arrays * (1 + noise)
    return perturbed


def uniform_pertubation(state, fields_to_perturbate=["q", "u", "v"], scale=1e-13):
    """Applies multiplicative uniform noise to selected fields of a model state.

    Creates a deep copy of the input state and perturbs the specified fields
    by multiplying each value by ``(1 + scale * U(-0.5, 0.5))``.
    
    Args:
        state (dict): A state dictionary as returned by ``build_initial_state``,
            containing a ``"fields"`` key mapping parameter names to NumPy arrays.
        fields_to_perturbate (list[str], optional): List of base variable names
            to perturb (e.g. ``["q", "u", "v"]``). Both exact key matches and
            pressure-level keys (e.g. ``"q_500"``) are perturbed. Defaults to
            ``["q", "u", "v"]``.
        scale (float, optional): Scaling factor applied to the uniform noise
            drawn from ``U(-0.5, 0.5)``. Defaults to ``1e-13``.
    
    Returns:
        dict: A deep copy of ``state`` with the specified fields perturbed.
    """
    perturbed = copy.deepcopy(state)
    for key, arrays in perturbed["fields"].items():
        # extract the base variable name (e.g., 'q' from 'q_500')
        base_var = key.split("_")[0]
        # perturbate only what we want to
        if key in fields_to_perturbate or base_var in fields_to_perturbate:
            noise = np.random.uniform(-0.5, 0.5, size=arrays.shape)
            perturbed["fields"][key] = arrays * (1 + scale * noise)
    return perturbed

def build_and_save_initial_states(
    date_input, N_MEMBERS, perturbed_fields, scale, save_path, perturbation_fn=gaussian_pertubation
):
    """Builds and saves a control state and an ensemble of perturbed initial states.

    Fetches the ERA5-based control initial state for the given date, saves it
    as member ``00``, then generates ``N_MEMBERS`` perturbed copies using the
    provided perturbation function and saves each as a separate pickle file.
    
    Output files are written to ``<save_path>/<YYYYMMDDHH>/`` and follow the
    naming::
    
        init_state-<YYYYMMDDHH>-<field1>-<field2>-<NN>.pkl
    
    where ``<NN>`` is ``00`` for the control and ``01``–``NN`` for ensemble members.
    
    Args:
        date_input (str): Initial date and time as a string in the format
            ``"%Y-%m-%d %H:%M:%S"`` (e.g. ``"2019-07-10 00:00:00"``).
        N_MEMBERS (int): Number of perturbed ensemble members to generate.
        perturbed_fields (list[str]): Base variable names to perturb
            (e.g. ``["q", "u", "v"]``).
        scale (float): Perturbation scale passed to ``perturbation_fn``.
        save_path (str): Root directory under which output files are saved.
        perturbation_fn (callable, optional): Function used to perturb the
            control state. Must have the signature
            ``fn(state, fields_to_perturbate, scale) -> dict``.
            Defaults to :func:`gaussian_pertubation`.
    
    Returns:
        None, but exports all the .pkl.
    """
    DATE = datetime.datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
    date_str = DATE.strftime("%Y%m%d%H")
    output_dir = os.path.join(save_path, date_str)
    fields_str = "-".join(perturbed_fields)
    os.makedirs(output_dir, exist_ok=True)

    init_control_state = build_initial_state(DATE)

    with open(
        os.path.join(output_dir, f"init_state-{date_str}-{fields_str}-00.pkl"), "wb"
    ) as f:
        pickle.dump(init_control_state, f)

    for i in tqdm(range(1, N_MEMBERS + 1)):
        perturbed_state = perturbation_fn(
            init_control_state, fields_to_perturbate=perturbed_fields, scale=scale
        )
        with open(
            os.path.join(output_dir, f"init_state-{date_str}-{fields_str}-{i:02d}.pkl"),
            "wb",
        ) as f:
            pickle.dump(perturbed_state, f)

if __name__ == "__main__":
    date_input = "2019-07-10 00:00:00"
    N_MEMBERS = 50
    perturbed_fields = ["q", "u", "v"]
    scale = 1e-13
    OUTPUT_SAVE = "/homedata/pchevali/AIFS_INPUTS/"
    build_and_save_initial_states(
        date_input, N_MEMBERS, perturbed_fields, scale, OUTPUT_SAVE, uniform_pertubation
    )
