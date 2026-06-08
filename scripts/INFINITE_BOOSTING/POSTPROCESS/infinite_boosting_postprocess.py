import os
import glob
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import argparse

def pkl_ensemble_to_raw_netcdf(input_dir, output_dir, date_str, exp_name, n_members, n_phases,mask=None):
    """Converts an ensemble of AIFS forecast pkl files to one NetCDF file per variable.
    Reads per-member pickle files produced by run_aifs(), retains the N320 1D values,
    and writes one NetCDF file per variable with dimensions (number, step, values).

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
    dir_pattern = os.path.join(input_dir, f"{date_str}-{exp_name}*")
    found_dirs = glob.glob(dir_pattern)
    if not found_dirs:
        raise FileNotFoundError(f"No output directory found matching: {dir_pattern}")

    dir_name = os.path.basename(found_dirs[0])
    print(f"Starting processing for {dir_name} ({n_members + 1} members and {n_phases} phases)...")

    # create the exact same directory structure for the NetCDF outputs
    out_dir = os.path.join(output_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    for phase in range(1,n_phases+1):
        # load first file to get init_time, variables, and number of steps from it
        control_path = os.path.join(found_dirs[0], f"aifs_outputs_phase{phase}-{dir_name}-00.pkl")

        with open(control_path, "rb") as f:
            control_data = pickle.load(f)

        # extract init_time, variable names, and number of steps from control member
        init_time = pd.Timestamp(control_data[0]["date"])
        num_steps = len(control_data)

        # collect ALL unique variables across all steps (accumulated fields like 'tp' are missing at step 0)
        variables_set = set()
        for step in control_data:
            variables_set.update(step["fields"].keys())
        variables = list(variables_set)

        print(f"First member of phase {phase} loaded. Variables: {variables}, Steps: {num_steps}")

        # build step timedeltas and valid times from control member dates

        valid_times = pd.to_datetime([step["date"] for step in control_data])
        step_timedeltas = valid_times - init_time

        # lat/lon targets
        with open("/homedata/pchevali/n320_coordinates.pkl", "rb") as f:
            coords = pickle.load(f)
        lats, lons = coords["latitude"], coords["longitude"]
        n_values = len(lats) if mask is None else len(lats[mask])
        
        # export z only once if present (since it's only on step 0 and same for everyone)
        if "z" in variables:
            z_data = control_data[0]["fields"]["z"]
            ds_z = xr.Dataset(
                data_vars={"z": (["values"], z_data)},
                coords={
                    "latitude":  ("values", lats if mask is None else lats[mask]),
                    "longitude": ("values", lons if mask is None else lons[mask]),
                }
            )
            z_out_file = os.path.join(out_dir, f"infinite_boosting_phase{phase}-{dir_name}-z.nc")
            ds_z.to_netcdf(z_out_file)
            print(f"Saved: {z_out_file}")
            variables.remove("z")

        # accumulate all members: {var: (n_members + 1, num_steps, 542080)}
        shape = (n_members + 1, num_steps, n_values)
        all_data = {var: np.full(shape, np.nan, dtype=np.float32) for var in variables}

        for member in range(n_members + 1):
            print(f"Processing member {member:02d}/{n_members} of phase {phase}/{n_phases}...", flush=True)
            pkl_file = os.path.join(found_dirs[0], f"aifs_outputs_phase{phase}-{dir_name}-{member:02d}.pkl")
            with open(pkl_file, "rb") as f:
                member_data = pickle.load(f)

            for step_idx, step_dict in enumerate(member_data):
                for var, field_N320 in step_dict["fields"].items():
                    if var != "z":
                        all_data[var][member, step_idx] = field_N320.astype(np.float32)

        # write one .nc file per variable
        for var in variables:
            print(f"Writing variable '{var}' to NetCDF...", flush=True)
            ds = xr.Dataset(
                data_vars={
                    var: (["number", "step", "values"], all_data[var])
                },
                coords={
                    "number":     ("number",    np.arange(n_members + 1)),
                    "time":       init_time,
                    "step":       ("step",      step_timedeltas),
                    "valid_time": ("step",      valid_times),
                    "latitude":  ("values", lats if mask is None else lats[mask]),
                    "longitude": ("values", lons if mask is None else lons[mask]),
                }
            )
            out_file = os.path.join(out_dir, f"infinite_boosting_phase{phase}-{dir_name}-{var}.nc")
            ds.to_netcdf(out_file)
            print(f"      Saved: {out_file}")
        print(f"All variables for phase {phase} written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--date_str", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_members", type=int, required=True)
    parser.add_argument("--n_phases",type=int, required=True)
    parser.add_argument("--mask", type=str,default=None)
    args = parser.parse_args()

    if args.mask is not None:
        with open( args.mask, "rb") as f:
            mask = pickle.load(f)
    else:
        mask=None

    pkl_ensemble_to_raw_netcdf(
        args.input_dir, args.output_dir, args.date_str, args.exp_name, args.n_members, args.n_phases, mask
    )
