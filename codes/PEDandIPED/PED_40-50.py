import os
import time
import pandas as pd
from utils.dd_utils import extract_attributes_thresholds, find_violating_pairs_parallel
from utils.difference import generate_difference_file
from utils.distance_utils import build_all_distance_lookups_for_block
from utils.evaluation_utils import calculate_precision_recall_f1, check_all_pairs_covered
from utils.MWVC import build_graph_parallel
from utils.preprocessing import generate_probability_files
from utils.cluster_utils import generate_partial_clusters

if __name__ == "__main__":
    dirty_file = "path/PED/data/MIMIC/dirty.csv"
    dd_constraints_file = "path/PED/data/MIMIC/dd_constraints_dd.txt"
    attribute_thresholds_file = "path/PED/data/MIMIC/attribute_thresholds.txt"
    difference_file = "path/PED/data/MIMIC/difference.csv"
    probability_file = "path/PED/data/MIMIC/predicted_probabilities.csv"
    block_error_output = "path/PED/data/MIMIC/block_errors_row_40_49.txt"

    with open(dd_constraints_file, "r", encoding="utf-8") as f:
        dd_constraints = eval(f.read())
    with open(attribute_thresholds_file, "r", encoding="utf-8") as f:
        attribute_thresholds = eval(f.read())
    print("Attribute thresholds:", attribute_thresholds)

    FIXED_NUM_PROCESSES = 16

    sorted_thresholds = extract_attributes_thresholds(dd_constraints)
    print("Extracted thresholds.")
    print("Sorted thresholds:", sorted_thresholds)

    diff_df = pd.read_csv(difference_file)
    ground_truth_errors = set(
        f"t{row['Index']}.{row['Attribute']}" for _, row in diff_df.iterrows()
    )

    dirty_df = pd.read_csv(dirty_file, dtype=str)
    dataset_scheme = list(dirty_df.columns)
    data_instance = dirty_df.to_dict(orient="records")
    length = len(data_instance)
    print(f"Dataset length: {length}")

    block_rows = 50
    block_cols = 50
    row_blocks = [list(range(k, length, block_rows)) for k in range(block_rows)]
    col_blocks = [list(range(k, length, block_cols)) for k in range(block_cols)]

    global_error_set = set()
    total_start_time = time.time()

    for i in range(40,50):
        for j in range(block_cols):
            row_indices = row_blocks[i]
            col_indices = col_blocks[j]
            print(f"Processing block (row {i+1}/{block_rows}, col {j+1}/{block_cols})...")

            block_data = [data_instance[r] for r in row_indices]
            col_data = [data_instance[c] for c in col_indices]

            start_time = time.time()
            distance_lookups = build_all_distance_lookups_for_block(
                block_data, col_data, sorted_thresholds, FIXED_NUM_PROCESSES
            )
            print(f"Time for building distance lookups: {time.time() - start_time:.2f}s")

            start_time = time.time()
            clusters_block = generate_partial_clusters(
                row_indices, data_instance, sorted_thresholds, distance_lookups, FIXED_NUM_PROCESSES, col_indices=col_indices
            )
            print(f"  Time for generating clusters: {time.time() - start_time:.2f}s")

            start_time = time.time()
            violating_pairs, hyper_edges = find_violating_pairs_parallel(
                dd_constraints, clusters_block, FIXED_NUM_PROCESSES, length, delta_n=0
            )
            print(f"Time for finding violating pairs: {time.time() - start_time:.2f}s")

            start_time = time.time()
            block_errors = build_graph_parallel(
                hyper_edges, probability_file, FIXED_NUM_PROCESSES, attribute_thresholds
            )
            global_error_set.update(block_errors)
            with open(block_error_output, "a", encoding="utf-8") as f:
                for cell in block_errors:
                    f.write(cell + "\n")

            # print(f"Time for building graph: {time.time() - start_time:.2f}s")
            print(f"Time for building graph: {time.time() - start_time:.2f}s")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    precision, recall, f1_score_val = calculate_precision_recall_f1(global_error_set, ground_truth_errors)

    print(f"Metrics => Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score_val:.4f}")

    results = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score_val,
        "TotalTime": round(total_time, 2),
    }

    df_results = pd.DataFrame([results])
    df_results.to_csv("path/PED/data/ICU/block_errors_row_40_49/result.csv", index=False)
    print("Results saved.")
