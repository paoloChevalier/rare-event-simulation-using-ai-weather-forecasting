import datetime
import numpy as np
import os
import pickle
import argparse
import copy
import glob
from anemoi.inference.runners.simple import SimpleRunner

def infinite_boosting(
    input_path,
    output_path,
    date_str,
    exp_name,
    checkpoint,
    phases,
    target_variables=["2t", "10u", "10v", "z_500"],
    times_to_save=[0, 6, 12, 18],
    N_MEMBERS=50,
    INDEX=67391,
    mask=None,
):
    """Runs the infinite boosting with the AIFS model for a given set of initial states and saves the forecasts.

    Args:
        input_path (str): Directory containing the input .pkl files with initial states.
        output_path (str): Directory where the output .pkl files with forecasts will be saved.
        date_str (str): Date string in the format "YYYYMMDDHH" corresponding to the initial states.
        checkpoint (str): Path to the AIFS model checkpoint file.
        lead_time (int, optional): Lead time in hours for the forecast. Defaults to 720 (30 days).
        phases (list({"lead_time": int, "backtrack": int})) : description of the different phases
        target_variables (list[str], optional): List of variable names to include in the output forecasts. Defaults to ["2t", "10u", "10v", "z_500"].
        times_to_save (list[int], optional): List of hours (0-23) at which to save the forecast steps. Defaults to [0, 6, 12, 18] (every 6 hours).
        N_MEMBERS (int, optional): Number of ensemble members to process (not including the control member 00). Defaults to 50
        INDEX (int, optional): index in the N320 grid where to do the boosting
        mask (np.array[boolean]): a mask indicating on which zone should the outputs be saved


    Returns:
        None, but saves the forecast outputs as .pkl files in the specified output directory.
    """
    # load files
    dir_name=f"{date_str}-{exp_name}"
    dir_pattern = os.path.join(input_path, f"{date_str}-{exp_name}")
    found_dir = glob.glob(dir_pattern)[0]
    if not found_dir:
        raise FileNotFoundError(f"No input directory found matching: {dir_pattern}")

    # creates the output directry if it doesn't exist
    out_dir = os.path.join(output_path, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    input_filename = os.path.join(found_dir, f"init_state-{dir_name}.pkl")
    # load the initial state from the input file
    with open(input_filename, "rb") as f:
        input_state = pickle.load(f)

    # loading the model
    runner = SimpleRunner(checkpoint, device="cuda")

    # initialisation 

    current_init_state = input_state

    for phase_idx, phase in enumerate(phases):
        lead_time = phase["lead_time"]
        backtrack = phase["backtrack"]
        is_last = phase_idx == len(phases) - 1

        start_date = current_init_state["date"]

        if not is_last and backtrack is not None:
            backtrack_date = start_date + datetime.timedelta(
                hours=lead_time - backtrack
            )
        else:
            backtrack_date = None

        max_temp_at_index = -999.0
        best_member_id = None
        best_backtrack_prev = None  # state 6h before the backtrack state
        best_backtrack_curr = None  # backtrack state

        print(
            f"\n=== PHASE {phase_idx + 1} : lead {lead_time}h, backtrack {backtrack}h ==="
        )

        for i in range(N_MEMBERS + 1):
            member_id = f"{i:02d}"

            backtrack_prev = None
            backtrack_curr = None
            prev_state = None
            member_max_temp = -999.0

            # run the model and collect the forecast steps
            forecast_steps = []

            # save initial state as the first step of the forecast
            forecast_steps.append(
                {
                    "date": current_init_state["date"],
                    "fields": {
                        k: v[1][mask] if mask is not None else v[1]
                        for k, v in input_state["fields"].items()
                        if k in target_variables
                    },
                }
            )

            # forecast
            for state in runner.run(
                input_state=current_init_state, lead_time=lead_time
            ):
                if backtrack_date is not None and state["date"] == backtrack_date:
                    backtrack_prev = copy.deepcopy(prev_state)
                    backtrack_curr = copy.deepcopy(state)
                # filter the states to be saved and exported
                if state["date"].hour in times_to_save:
                    forecast_steps.append(
                        {
                            "date": state["date"],
                            "fields": {
                                k: v[mask] if mask is not None else v
                                for k, v in state["fields"].items()
                                if k in target_variables
                            },
                        }
                    )

                # Track max temp and update prev_step_state AFTER extraction
                current_temp = state["fields"]["2t"].flatten()[INDEX].item()
                if current_temp > member_max_temp:
                    member_max_temp = current_temp
                prev_state = copy.deepcopy(state)

            # switch our max ensemble member if the observable was bigger for this member
            if member_max_temp > max_temp_at_index:
                max_temp_at_index = member_max_temp
                best_member_id = member_id
                if backtrack_prev is not None and backtrack_curr is not None:
                    best_backtrack_prev = backtrack_prev
                    best_backtrack_curr = backtrack_curr

            output_filename = os.path.join(
                output_path,
                f"aifs_outputs_phase{phase_idx + 1}-{dir_name}-{member_id}.pkl",
            )
            with open(output_filename, "wb") as f_out:
                pickle.dump(forecast_steps, f_out)

        print(
            f"Meilleur membre phase {phase_idx + 1} : {best_member_id} (température at selected index = {(max_temp_at_index - 273.15):.2f})"
        )

        # formatting the input state for the next phase

        if (
            not is_last
            and best_backtrack_prev is not None
            and best_backtrack_curr is not None
        ):
            new_init = {"date": best_backtrack_curr["date"], "fields": {}}
            for k in current_init_state["fields"].keys():
                if k in best_backtrack_curr["fields"]:
                    new_init["fields"][k] = np.stack(
                        [
                            best_backtrack_prev["fields"][k],
                            best_backtrack_curr["fields"][k],
                        ],
                        axis=0,
                    )
                else:
                    # static variables
                    new_init["fields"][k] = current_init_state["fields"][k]
            current_init_state = new_init


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--date_str", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_members", type=int, required=True)
    parser.add_argument("--savetimes", nargs="+", type=int, default=[0, 6, 12, 18])
    parser.add_argument("--variables", nargs="+", default=["2t", "10u", "10v", "z_500"])
    parser.add_argument("--index_boosting", type=int, required=True)
    parser.add_argument("--leads", nargs="+", type=int, required=True)
    parser.add_argument("--backtracks", nargs="+", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)

    args = parser.parse_args()

    phases = []
    for lead, backtrack_str in zip(args.leads, args.backtracks):
        if backtrack_str == "None":
            backtrack = None
        else:
            backtrack = int(backtrack_str)
        phases.append({"lead_time": lead, "backtrack": backtrack})

    if args.mask is not None:
        with open(args.mask, "rb") as f:
            mask = pickle.load(f)
    else:
        mask = None

    infinite_boosting(
        args.input_path,
        args.output_path,
        args.date_str,
        args.exp_name,
        args.checkpoint,
        phases,
        args.variables,
        args.savetimes,
        args.n_members,
        args.index_boosting,
        mask,
    )
