##imports
import datetime
from collections import defaultdict
import os
# if usage is too expensive on ram
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' 
#os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = '16'
import pickle
import glob

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

def run_aifs(input_path, output_path, date_str, checkpoint, lead_time=720, target_variables=["2t", "10u", "10v", "z_500"], times_to_save=[0, 6, 12, 18], N_MEMBERS=50):
    """Runs the AIFS model for a given set of initial states and saves the forecasts.
    
    Args:
        input_path (str): Directory containing the input .pkl files with initial states.
        output_path (str): Directory where the output .pkl files with forecasts will be saved.
        date_str (str): Date string in the format "YYYYMMDDHH" corresponding to the initial states.
        checkpoint (str): Path to the AIFS model checkpoint file.
        lead_time (int, optional): Lead time in hours for the forecast. Defaults to 720 (30 days).
        target_variables (list[str], optional): List of variable names to include in the output forecasts. Defaults to ["2t", "10u", "10v", "z_500"].
        times_to_save (list[int], optional): List of hours (0-23) at which to save the forecast steps. Defaults to [0, 6, 12, 18] (every 6 hours).
        N_MEMBERS (int, optional): Number of ensemble members to process (not including the control member 00). Defaults to 50
    
    Returns:
        None, but saves the forecast outputs as .pkl files in the specified output directory.
    """
    # create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # get the perturbed fields string from one of the input files
    sample_file = os.path.basename(glob.glob(os.path.join(input_path, f"init_state-{date_str}-*-00.pkl"))[0])
    fields_str = sample_file.replace(f"init_state-{date_str}-", "").replace("-00.pkl", "")

    #loading the model
    runner = SimpleRunner(checkpoint, device="cuda")

    #for each member, load the initial state, run the model and save the output
    for i in range(N_MEMBERS + 1):
        # format the member id to be two digits, e.g. 00, 01, ..., 50 and construct the input filename
        member_id = f"{i:02d}"
        input_filename = os.path.join(input_path, f"init_state-{date_str}-{fields_str}-{member_id}.pkl")
        
        # load the initial state from the input file
        with open(input_filename,"rb") as f:
            input_state = pickle.load(f)

        # run the model and collect the forecast steps
        forecast_steps = []   
        #save initial state as the first step of the forecast
        forecast_steps.append({
            "date": input_state["date"],
            "fields": {k: v[1] for k, v in input_state["fields"].items() if k in target_variables}
        })
        #forecast
        for state in runner.run(input_state=input_state, lead_time=lead_time):
            # filter the state to only include the target variables
            if state["date"].hour in times_to_save:
                filtered_state = {
                    "date": state["date"],
                    "fields": {k: v for k, v in state["fields"].items() if k in target_variables}
                }
                forecast_steps.append(filtered_state)
        
        # save the forecast steps to the output file
        output_filename = os.path.join(output_path, f"aifs_output-{date_str}-{fields_str}-{member_id}.pkl")
        with open(output_filename, "wb") as f_out:
            pickle.dump(forecast_steps, f_out)
    
if __name__ == "__main__":
    INPUT_PATH = "aaaaa"
    OUTPUT_PATH = "aaaa"
    DATE_STR = "YYYYMMDDHH"
    CHECKPOINT = "path_to_chkpt.ckpt"
    LEAD_TIME = 720
    N_MEMBERS = 50
    SAVETIMES = [0, 6 ,12 ,18] #could be used to only save certain forecast steps, e.g. every 24 hours with [18], time of day to save
    VARIABLES = ["2t", "10u", "10v", "z_500"]
    run_aifs(INPUT_PATH, OUTPUT_PATH, DATE_STR, CHECKPOINT, LEAD_TIME, VARIABLES, SAVETIMES, N_MEMBERS)
