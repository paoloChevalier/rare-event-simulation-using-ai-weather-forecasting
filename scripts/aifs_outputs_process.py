import os
import glob
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import earthkit.regrid as ekr
import argparse

def pkl_ensemble_to_regridded_netcdf(input_dir, output_dir, date_str, exp_name, n_members=50):
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
    # Find the input directory by itself
    dir_pattern = os.path.join(input_dir, f"{date_str}-{exp_name}-*")
    found_dirs = glob.glob(dir_pattern)
    if not found_dirs:
        raise FileNotFoundError(f"No output directory found matching: {dir_pattern}")
        
    in_dir = found_dirs[0]
    dir_name = os.path.basename(in_dir)
    
    print(f"[1/4] Starting processing for {dir_name} ({n_members + 1} members)...")
    
    # create the exact same directory structure for the NetCDF outputs
    out_dir = os.path.join(output_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # load control member
    control_path = os.path.join(in_dir, f"aifs_output-{dir_name}-00.pkl")
    
    with open(control_path, "rb") as f:
        control_data = pickle.load(f)

    # extract init_time, variable names, and number of steps from control member
    init_time = pd.Timestamp(control_data[0]["date"])
    variables = list(control_data[0]["fields"].keys())
    num_steps = len(control_data)
    
    print(f"[2/4] Control loaded. Variables: {variables}, Steps: {num_steps}")

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
        print(f"[3/4] Regridding member {member:02d}/{n_members}...", flush=True)
        pkl_file = os.path.join(in_dir, f"aifs_output-{dir_name}-{member:02d}.pkl")
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
        print(f"[4/4] Writing variable '{var}' to NetCDF...", flush=True)
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
        out_file = os.path.join(out_dir, f"aifs_ensemble-{dir_name}-{var}.nc")
        ds.to_netcdf(out_file)
        print(f"      Saved: {out_file}")
    print(f"All variables written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--date_str", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_members", type=int, required=True)
    args = parser.parse_args()

    pkl_ensemble_to_regridded_netcdf(
        args.input_dir, args.output_dir, args.date_str, args.exp_name, args.n_members
    )
