import datetime
import numpy as np
import os
import pickle
import argparse
import copy
import glob
from anemoi.inference.runners.simple import SimpleRunner
import json

def infinite_boosting(
    input_path,
    output_path,
    date_strs,
    exp_name,
    checkpoint,
    phases,
    target_variables=["2t", "10u", "10v", "z_500"],
    times_to_save=[0, 6, 12, 18],
    time_to_boost=12,
    N_MEMBERS=50,
    INDEX=67391,
    mask=None,
    q=0.10,
):
    """.

    Args:
        input_path (str): Directory containing the input .pkl files with initial states.
        output_path (str): Directory where the output .pkl files with forecasts will be saved.
        date_str (str): Date string in the format "YYYYMMDDHH" corresponding to the initial states.
        exp_name (str): Name of the experiment.
        checkpoint (str): Path to the AIFS model checkpoint file.
        phases (list[dict]): List of dictionaries containing 'forward_time' and 'lead' for each phase.
        target_variables (list[str], optional): List of variable names to include in the outputs. Defaults to ["2t", "10u", "10v", "z_500"].
        times_to_save (list[int], optional): List of hours (0-23) at which to save the forecast steps. Defaults to [0, 6, 12, 18].
        N_MEMBERS (int, optional): Number of ensemble members to process. Defaults to 50.
        INDEX (int, optional): index in the N320 grid where to do the boosting.
        mask (np.array[boolean]): a mask indicating on which zone should the outputs be saved.
        q (float, optional): Fraction of top-performing members to keep per phase. Defaults to 0.10.
    """
    # load files
    AC_0_t = []

    #create output dir
    out_dir = os.path.join(output_path, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    for date_str in date_strs:
        dir_name = f"{date_str}-{exp_name}"
        dir_pattern = os.path.join(input_path, f"{date_str}-{exp_name}")
        found_dir = glob.glob(dir_pattern)[0]
        if not found_dir:
            raise FileNotFoundError(f"No input directory found matching: {dir_pattern}")

        input_filename = os.path.join(found_dir, f"init_state-{dir_name}.pkl")
        # load the initial state from the input file
        with open(input_filename, "rb") as f:
            input_state = pickle.load(f)
        AC_0_t.append(input_state)

    # loading the model
    runner = SimpleRunner(checkpoint, device="cuda")

    # initialisation: maintain a list of states to branch from
    init_states_for_phase = [
            {"parent_id": date_strs[idx], "anemoi_state": state}
            for idx, state in enumerate(AC_0_t)
        ]

    tree_history = {
            "experiment": exp_name,
            "phases": []
        }

    del AC_0_t

    for phase_idx, phase in enumerate(phases):
        forward_time = phase["forward_time"]
        lead = phase["lead"]
        is_last = phase_idx == len(phases) - 1

        print(
            f"\n=== PHASE {phase_idx + 1} : forward {forward_time}h, lead {lead}h ==="
        )

        n_batch = int(N_MEMBERS / len(init_states_for_phase))
        k_survivors = max(1, int((n_batch * len(init_states_for_phase)) * q))

        survivors = []
        phase_tracking = {
            "phase_id": phase_idx + 1,
            "T_ref": None,
            "members": []
        }

        member_counter = 0
        for current_wrapper in init_states_for_phase:
            current_init_state = current_wrapper["anemoi_state"]
            parent_id = current_wrapper["parent_id"]
            for _ in range(n_batch):
                member_id = f"p{phase_idx + 1}_m{member_counter:03d}"
                member_counter += 1

                start_date = current_init_state["date"]

                if not is_last and lead is not None:
                    lead_date = start_date + datetime.timedelta(
                        hours=forward_time - lead
                    )
                else:
                    lead_date = None

                lead_prev = None
                lead_curr = None
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
                            for k, v in current_init_state["fields"].items()
                            if k in target_variables
                        },
                    }
                )

                # forecast
                for state in runner.run(
                    input_state=current_init_state, lead_time=forward_time
                ):
                    if lead_date is not None and state["date"] == lead_date:
                        lead_prev = copy.deepcopy(prev_state)
                        lead_curr = copy.deepcopy(state)

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
                    if state["date"].hour == time_to_boost:
                        current_temp = state["fields"]["2t"].flatten()[INDEX].item()
                        if current_temp > member_max_temp:
                            member_max_temp = current_temp
                    prev_state = copy.deepcopy(state)

                output_filename = os.path.join(
                    out_dir,
                    f"aifs_outputs_phase{phase_idx + 1}-{exp_name}-{member_id}.pkl",
                )
                with open(output_filename, "wb") as f_out:
                    pickle.dump(forecast_steps, f_out)

                del forecast_steps
                del prev_state

                phase_tracking["members"].append({
                                "member_id": member_id,
                                "parent_id": parent_id,
                                "max_temp": member_max_temp,
                                "file_path": output_filename
                            })

                # Append the new member to survivors list
                survivors.append({
                    "member_id": member_id,
                    "max_temp": member_max_temp,
                    "lead_prev": lead_prev,
                    "lead_curr": lead_curr,
                    "original_init_state": current_init_state
                })

                # Sort descending by max_temp
                survivors.sort(key=lambda x: x["max_temp"], reverse=True)

                # If we have more members than our limit, pop the worst one to free up RAM
                if len(survivors) > k_survivors:
                    survivors.pop()

        current_t_ref = survivors[-1]['max_temp']
        phase_tracking["T_ref"] = current_t_ref
        tree_history["phases"].append(phase_tracking)

        # By the time the loop ends, 'survivors' already contains exactly the top fraction requested
        print(f"Top member phase {phase_idx + 1}: {survivors[0]['member_id']} (Temp = {(survivors[0]['max_temp'] - 273.15):.2f}°C)")
        print(f"Threshold for top {q*100}% (T_ref) was {(current_t_ref - 273.15):.2f}°C")

        # Format the next initial states from the survivors
        if not is_last:
            init_states_for_phase = [] # Reset for the next phase
            for survivor in survivors:
                if survivor["lead_prev"] is not None and survivor["lead_curr"] is not None:
                    new_init = {"date": survivor["lead_curr"]["date"], "fields": {}}
                    for k in survivor["original_init_state"]["fields"].keys():
                        if k in survivor["lead_curr"]["fields"]:
                            new_init["fields"][k] = np.stack(
                                [
                                    survivor["lead_prev"]["fields"][k],
                                    survivor["lead_curr"]["fields"][k],
                                ],
                                axis=0,
                            )
                        else:
                            # static variables
                            new_init["fields"][k] = survivor["original_init_state"]["fields"][k]
                    init_states_for_phase.append({
                        "parent_id": survivor["member_id"],
                        "anemoi_state": new_init
                    })
        del survivors

    lineage_file = os.path.join(out_dir, f"{exp_name}_lineage.json")
    with open(lineage_file, "w") as f:
        json.dump(tree_history, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--date_strs", nargs="+", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_members", type=int, required=True)
    parser.add_argument("--savetimes", nargs="+", type=int, default=[0, 6, 12, 18])
    parser.add_argument("--variables", nargs="+", default=["2t", "10u", "10v", "z_500"])
    parser.add_argument("--index_boosting", type=int, required=True)
    parser.add_argument("--forwards", nargs="+", type=int, required=True)
    parser.add_argument("--leads", nargs="+", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--q", type=float, default=0.10)

    args = parser.parse_args()

    phases = []
    for forward, lead_str in zip(args.forwards, args.leads):
        if lead_str == "None":
            lead = None
        else:
            lead = int(lead_str)
        # Perfectly aligns with f_j and t_j terminology
        phases.append({"forward_time": forward, "lead": lead})

    if args.mask is not None:
        with open(args.mask, "rb") as f:
            mask = pickle.load(f)
    else:
        mask = None

    infinite_boosting(
        input_path=args.input_path,
        output_path=args.output_path,
        date_strs=args.date_strs,
        exp_name=args.exp_name,
        checkpoint=args.checkpoint,
        phases=phases,
        target_variables=args.variables,
        times_to_save=args.savetimes,
        time_to_boost=12,
        N_MEMBERS=args.n_members,
        INDEX=args.index_boosting,
        mask=mask,
        q=args.q,
    )
