import os
import pickle
import json
import numpy as np
import xarray as xr
import pandas as pd
import argparse

def pkl_ensemble_to_raw_netcdf(input_dir, output_dir, exp_name, n_members, n_phases, mask=None):

    in_dir = os.path.join(input_dir, exp_name)
    out_dir = os.path.join(output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # tree
    with open(os.path.join(in_dir, f"{exp_name}_lineage.json")) as f:
        lineage = json.load(f)
    root_times = {}
    parent_map = {}
    for phase_info in lineage["phases"]:
        for m in phase_info["members"]:
            m_id = m["member_id"]
            parent = m["parent_id"]

            parent_map[m_id] = parent
            # get root initialisation time from parent, or if it's the first phase the date of initialisation
            if parent in root_times:
                root_times[m_id] = root_times[parent]
            else:
                root_times[m_id] = pd.to_datetime(parent, format="%Y%m%d%H")
    # load lat/lon coordinates
    with open("/homedata/pchevali/n320_coordinates.pkl", "rb") as f:
        coords = pickle.load(f)
    lats = coords["latitude"][mask] if mask is not None else coords["latitude"]
    lons = coords["longitude"][mask] if mask is not None else coords["longitude"]

    # process each phase
    for phase in range(1, n_phases + 1):
        print(f"\n=== Processing Phase {phase} ===")

        phase_data, m_ids = [], []
        for i in range(n_members):
            m_id = f"p{phase}_m{i:03d}"
            m_ids.append(m_id)
            with open(os.path.join(in_dir, f"aifs_outputs_phase{phase}-{exp_name}-{m_id}.pkl"), "rb") as f:
                phase_data.append(pickle.load(f))

        # coordinates
        variables = list(phase_data[0][0]["fields"].keys())
        times = [root_times[m_id] for m_id in m_ids]
        parents = [parent_map[m_id] for m_id in m_ids]
        valid_times = [[step["date"] for step in member] for member in phase_data]
        step_deltas = [vt - times[0] for vt in valid_times[0]] #steps are always the same

        # save 'z' once if present (static)
        if "z" in variables:
            xr.Dataset(
                {"z": (["values"], phase_data[0][0]["fields"]["z"])},
                coords={"latitude": ("values", lats), "longitude": ("values", lons)}
            ).to_netcdf(os.path.join(out_dir, f"infinite_boosting_phase{phase}-{exp_name}-z.nc"))
            variables.remove("z")
            print("Saved static 'z' field.")

        # savevariables
        for var in variables:
            # Extract data (number, step, values)
            var_data = np.array([[step["fields"][var] for step in m] for m in phase_data])

            ds = xr.Dataset(
                data_vars={var: (["number", "step", "values"], var_data)},
                coords={
                    "number": np.arange(n_members),
                    "member_id": ("number", m_ids),
                    "parent_id": ("number", parents),
                    "time": ("number", times),
                    "step": ("step", step_deltas),
                    "valid_time": (["number", "step"], valid_times),
                    "latitude": ("values", lats),
                    "longitude": ("values", lons),
                }
            )

            out_path = os.path.join(out_dir, f"iterative_boosting_phase{phase}-{exp_name}-{var}.nc")
            ds.to_netcdf(out_path)
            print(f"Saved {var}.nc")
            del var_data, ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_members", type=int, required=True)
    parser.add_argument("--n_phases", type=int, required=True)
    parser.add_argument("--mask", type=str, default=None)
    args = parser.parse_args()

    mask = pickle.load(open(args.mask, "rb")) if args.mask else None
    pkl_ensemble_to_raw_netcdf(
        args.input_dir, args.output_dir, args.exp_name, args.n_members, args.n_phases, mask
    )
