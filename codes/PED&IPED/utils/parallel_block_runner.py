import os
import time
import pandas as pd
from multiprocessing import Process
from utils.dd_utils import extract_attributes_thresholds, find_violating_pairs_parallel
from utils.distance_utils import build_all_distance_lookups_for_block
from utils.evaluation_utils import calculate_precision_recall_f1
from utils.cluster_utils import generate_partial_clusters
from utils.MWVC import build_graph_parallel

def process_block_range(start_row_block, end_row_block, data_instance, sorted_thresholds,
                        dd_constraints, attribute_thresholds, probability_file, output_prefix,
                        num_processes=16):
    length = len(data_instance)
    block_rows = 50
    block_cols = 50
    row_blocks = [list(range(k, length, block_rows)) for k in range(block_rows)]
    col_blocks = [list(range(k, length, block_cols)) for k in range(block_cols)]

    local_error_set = set()
    start_time_total = time.time()

    for i in range(start_row_block, end_row_block):
        for j in range(block_cols):
            row_indices = row_blocks[i]
            col_indices = col_blocks[j]
            print(f"Processing block (row {i+1}/{block_rows}, col {j+1}/{block_cols})...")

            block_data = [data_instance[r] for r in row_indices]
            col_data = [data_instance[c] for c in col_indices]

            distance_lookups = build_all_distance_lookups_for_block(
                block_data, col_data, sorted_thresholds, num_processes
            )

            clusters_block = generate_partial_clusters(
                row_indices, data_instance, sorted_thresholds,
                distance_lookups, num_processes, col_indices=col_indices
            )

            violating_pairs, hyper_edges = find_violating_pairs_parallel(
                dd_constraints, clusters_block, num_processes, length, delta_n=0
            )

            block_errors = build_graph_parallel(
                hyper_edges, probability_file, num_processes, attribute_thresholds
            )

            local_error_set.update(block_errors)

            block_error_output = f"{output_prefix}/block_errors_row_{start_row_block}_{end_row_block - 1}.txt"
            with open(block_error_output, "a", encoding="utf-8") as f:
                for cell in block_errors:
                    f.write(cell + "\n")

    total_time = time.time() - start_time_total
    result_csv = f"{output_prefix}/block_errors_row_{start_row_block}_{end_row_block - 1}/partial_result.csv"
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    pd.DataFrame([{"TotalTime": round(total_time, 2)}]).to_csv(result_csv, index=False)

    print(f"Block {start_row_block}~{end_row_block - 1} finished in {total_time:.2f}s. Errors: {len(local_error_set)}")
