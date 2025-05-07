import copy
import json
import os
import time
from subprocess import run

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1) Configuration Parameters
# ------------------------------------------------------------------
attribute_list = ["flight", "actArrTime", "schedArrTime", "actDepTime", "schedDepTime"]
# attribute_list = [
#     "ProviderNumber",
#     "HospitalName",
#     "Address1",
#     "Address2",
#     "Address3",
#     "City",
#     "State",
#     "ZipCode",
#     "CountyName",
#     "PhoneNumber",
#     "HospitalType",
#     "HospitalOwner",
#     "EmergencyService",
#     "Condition",
#     "MeasureCode",
#     "MeasureName",
#     "Score",
#     "Sample",
#     "Stateavg"
# ]
# You can specify the desired range of thresholds to be generated.
threshold_search_space = {
    "flight": [0.00, 0.03, 0.06, 0.09, 0.12, 0.13, 0.14, 0.15, 0.20],
    "actArrTime": [0.00, 0.14, 0.31, 0.32, 0.40, 0.41, 0.48, 0.49, 0.50],
    "schedArrTime": [0.50, 0.53, 0.54, 0.58, 0.59, 0.60, 0.67, 0.68, 0.70],
    "actDepTime": [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00],
    "schedDepTime": [0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80],
}
# threshold_search_space = {
#     attr: [round(x * 0.1, 1) for x in range(11)]  # 生成 [0.0, 0.1, ..., 1.0]
#     for attr in attribute_list
# }


initial_thresholds = {
    "flight": 0.0,
    "actArrTime": 0.0,
    "schedArrTime": 0.0,
    "actDepTime": 0.0,
    "schedDepTime": 0.0,
}

thresholds_file = "path/PED/data/Flight/attribute_thresholds.txt"
result_file = "path/PED/data/Flight/result.csv"
run_script = "path/PED/codes/PEDandIPED/PED.py"


# ------------------------------------------------------------------
# 2) Invoke detection script and read F1 score
# ------------------------------------------------------------------
def run_pipeline_with_thresholds(threshold_dict):
    with open(thresholds_file, "w") as f:
        json.dump(threshold_dict, f)

    start = time.time()
    result = run(["python", run_script], capture_output=True, text=True)
    duration = time.time() - start

    if result.returncode != 0:
        print("Error running pipeline:")
        print(result.stderr)
        return -1

    df = pd.read_csv(result_file)
    f1 = df["F1"].values[0]
    print(f"Run completed in {duration:.2f}s with F1={f1:.4f}")
    return f1


# ------------------------------------------------------------------
# 3) Greedy Hyperparameter Tuning Main Function
# ------------------------------------------------------------------
def greedy_tune_thresholds(initial_thresholds, max_rounds=1):
    best_thresholds = copy.deepcopy(initial_thresholds)

    for round_idx in range(max_rounds):
        print(f"\n===== Round {round_idx + 1} =====")
        updated = False

        for attr in attribute_list:
            best_f1 = -1
            best_t = best_thresholds[attr]

            for t in threshold_search_space[attr]:
                trial_thresholds = copy.deepcopy(best_thresholds)
                trial_thresholds[attr] = t
                print(f"Testing {attr}={t} ... ", end="", flush=True)
                f1 = run_pipeline_with_thresholds(trial_thresholds)

                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            if best_t != best_thresholds[attr]:
                print(f"Update: {attr} = {best_thresholds[attr]} → {best_t}")
                best_thresholds[attr] = best_t
                updated = True
            else:
                print(f"No improvement for {attr}")

        if not updated:
            print("No update in this round. Stopping early.")
            break

    return best_thresholds


# ------------------------------------------------------------------
# 4) Perform Hyperparameter Tuning
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting threshold tuning...")
    best_thresholds = greedy_tune_thresholds(initial_thresholds)

    print("\n===== Final Best Thresholds =====")
    for attr, val in best_thresholds.items():
        print(f"{attr}: {val}")

    with open(thresholds_file, "w") as f:
        json.dump(best_thresholds, f)
    print(f"Saved best thresholds to {thresholds_file}")
