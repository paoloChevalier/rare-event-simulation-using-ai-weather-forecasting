import datetime
from collections import defaultdict

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr

from tqdm import tqdm
import os
import pickle
import multiurl
import multiurl.base
import argparse

os.environ["CDSAPI_RC"] = "/home/pchevali/.cdsapirc"
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["stl1", "stl2", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]
PARAM_PL = ["z", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

def get_open_data(target_date, param, member=0, levelist=[]):
    """Fetches ERA5 ensemble data from CDS for a target date and 6 hours before.

    Retrieves data on a 0.25-degree grid and interpolates it to the N320 Gaussian
    grid. For each parameter, the two time steps (t-6h and t) are stacked into
    a single array.
    
    Args:
        target_date (datetime.datetime): The target date and time for which to
            retrieve data. Data is also fetched 6 hours before this date.
        param (list[str]): List of parameter short names to retrieve
            (e.g. ``["10u", "10v", "2t"]``).
        member (int): The ensemble member number to fetch (0-9 for ERA5).
        levelist (list[int], optional): Pressure levels in hPa to retrieve.
            If non-empty, fetches from the pressure-level dataset; otherwise
            fetches from the single-level dataset. Defaults to ``[]``.
    
    Returns:
        dict[str, np.ndarray]: A mapping from parameter name to a stacked NumPy
            array of shape ``(2, N320_lat, N320_lon)``. 
    """
    fields = defaultdict(list)
    dataset = "reanalysis-era5-pressure-levels" if levelist else "reanalysis-era5-single-levels"
    
    # Get the data for the current date and the previous date
    for date in [target_date - datetime.timedelta(hours=6), target_date]:
        request = {
            "product_type": "ensemble_members",
            "member": str(member),
            "param": param,
            "date": date.strftime("%Y-%m-%d"),
            "time": date.strftime("%H:%M"),
            "grid": [0.25, 0.25], # ERA5 EDA is natively lower res, but CDS will interpolate
        }
        
        if levelist:
            request["pressure_level"] = levelist

        data = ekd.from_source("cds", dataset, **request)
                
        for f in data:
            # Filter the ensemble member locally, since CDS returns all 10
            if str(f.metadata("number")) != str(member):
                continue
                
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

def build_initial_state(target_date, member):
    """Builds a complete initial model state by fetching all required ERA5 fields for a specific member.
    """
    fields = {}
    
    fields.update(get_open_data(target_date, param=PARAM_SFC, member=member))
    fields.update(get_open_data(target_date, param=PARAM_SOIL, member=member))
    fields.update(get_open_data(target_date, param=PARAM_PL, member=member, levelist=LEVELS))

    return dict(date=target_date, member=member, fields=fields)

def download_and_save_ensemble(date_input, save_path, exp_name):
    """Downloads and saves all 10 true ERA5 ensemble members."""
    
    DATE = datetime.datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
    date_str = DATE.strftime("%Y%m%d%H")
    
    # ERA5 only has 10 ensemble members
    n_members = 10

    output_dir = os.path.join(save_path, f"{date_str}-{exp_name}-ERA5_EDA")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading all {n_members} ERA5 ensemble members for {date_input}...")

    for member in tqdm(range(n_members)):
        state = build_initial_state(DATE, member)
        with open(os.path.join(output_dir, f"init_state-{date_str}-{exp_name}-EDA-{member:02d}.pkl"), "wb") as f:
            pickle.dump(state, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date_input", type=str, required=True, help="Format: 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--output_save", type=str, required=True)
    args = parser.parse_args()
    
    download_and_save_ensemble(
        date_input=args.date_input,
        save_path=args.output_save,
        exp_name=args.exp_name
    )
    
    print("Done")
