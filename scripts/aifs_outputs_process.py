import os
import glob
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import earthkit.regrid as ekr

def pkl_ensemble_to_regridded_netcdf(input_dir, output_dir, date_str, n_members=50):
    """Converts an ensemble of AIFS forecast pkl files to one regridded NetCDF file per variable.

    Reads per-member pickle files produced by run_aifs(), regrids each field
    from the N320 Gaussian grid to a regular 0.25-degree lat/lon grid, and
    writes one NetCDF file per variable with dimensions (number, step, latitude, longitude).

    Args:
        input_dir (str): Directory containing the aifs_output-*.pkl files.
        output_dir (str): Directory where the output NetCDF files will be written.
        date_str (str): Initialisation date string in the format "YYYYMMDDHH".
        n_members (int, optional): Number of perturbed members (excluding the control).
            Total members processed = n_members + 1 (00 to n_members). Defaults to 50.

    Returns:
        None. Writes one NetCDF file per variable to output_dir, named
        ``aifs_ensemble-<date_str>-<fields_str>-<var>.nc``.
    """
    os.makedirs(output_dir, exist_ok=True)

    # get the perturbed fields string
    sample_file = os.path.basename(
        glob.glob(os.path.join(input_dir, f"aifs_output-{date_str}-*-00.pkl"))[0]
    )
    fields_str = sample_file.replace(f"aifs_output-{date_str}-", "").replace("-00.pkl", "")

    # load control member to get time steps and variable names
    control_path = os.path.join(input_dir, f"aifs_output-{date_str}-{fields_str}-00.pkl")
    with open(control_path, "rb") as f:
        control_data = pickle.load(f)

    # extract init_time, variable names, and number of steps from control member
    init_time = pd.Timestamp(control_data[0]["date"])
    variables = list(control_data[0]["fields"].keys())
    num_steps = len(control_data)

    # build step timedeltas and valid times from control member dates
    step_timedeltas = np.array(
        [pd.Timestamp(step["date"]) - init_time for step in control_data],
        dtype="timedelta64[ns]",
    )
    valid_times = np.array(
        [pd.Timestamp(step["date"]) for step in control_data],
        dtype="datetime64[ns]",
    )

    # lat/lon targets
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)

    # accumulate all members: {var: (n_members + 1, num_steps, 721, 1440)}
    all_data = {
        var: np.empty((n_members + 1, num_steps, 721, 1440), dtype=np.float32)
        for var in variables
    }

    for member in range(n_members + 1):
        pkl_file = os.path.join(
            input_dir, f"aifs_output-{date_str}-{fields_str}-{member:02d}.pkl"
        )
        with open(pkl_file, "rb") as f:
            member_data = pickle.load(f)

        for step_idx, step_dict in enumerate(member_data):
            for var, field_N320 in step_dict["fields"].items():
                all_data[var][member, step_idx] = ekr.interpolate(
                    field_N320,
                    {"grid": "N320"},
                    {"grid": [0.25, 0.25]},
                ).astype(np.float32)

    # write one .nc file per variable
    for var in variables:
        ds = xr.Dataset(
            data_vars={
                var: (["number", "step", "latitude", "longitude"], all_data[var])
            },
            coords={
                "number":     ("number",    np.arange(n_members + 1)),
                "time":       init_time,
                "step":       ("step",      step_timedeltas),
                "valid_time": ("step",      valid_times),
                "latitude":   ("latitude",  lats),
                "longitude":  ("longitude", lons),
            }
        )
        
        out_file = os.path.join(output_dir, f"aifs_ensemble-{date_str}-{fields_str}-{var}.nc")
        ds.to_netcdf(out_file)


if __name__ == "__main__":
    INPUT_DIR = "aaaaa"
    OUTPUT_DIR = "bbbbb"
    DATE_STR = "YYYYMMDDHH"
    N_MEMBERS = 50

    pkl_ensemble_to_regridded_netcdf(INPUT_DIR, OUTPUT_DIR, DATE_STR, N_MEMBERS)
