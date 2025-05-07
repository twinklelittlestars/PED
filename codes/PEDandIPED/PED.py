import os
import time
import pandas as pd
from multiprocessing import Process
from utils.dd_utils import extract_attributes_thresholds, find_violating_pairs_parallel
from utils.parallel_block_runner import process_block_range
from utils.distance_utils import build_all_distance_lookups
from utils.cluster_utils import generate_clusters_parallel
from utils.MWVC import build_graph_parallel
from utils.evaluation_utils import calculate_precision_recall_f1, check_all_pairs_covered
from utils.difference import generate_difference_file
from utils.preprocessing import generate_probability_files

if __name__ == "__main__":
    dirty_file = "path/PED/data/Flight/dirty.csv"
    clean_file = "path/PED/data/Flight/clean.csv"
    dd_constraints_file = "path/PED/data/Flight/dd_constraints_dd.txt"
    attribute_thresholds_file = "path/PED/data/Flight/attribute_thresholds.txt"
    probability_file = "path/PED/data/Flight/predicted_probabilities.csv"
    difference_file = "path/PED/data/Flight/difference.csv"
    output_prefix = "path/PED/data/Flight"

    num_processes = 16
    block_rows = 50

    dirty_df = pd.read_csv(dirty_file, dtype=str)
    data_instance = dirty_df.to_dict(orient="records")
    length = len(data_instance)
    print("length of data_instance:", length)
    dirty_df = dirty_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    dirty_df = dirty_df.replace('', pd.NA).dropna()
    dirty_df.to_csv(dirty_file, index=False)

    with open(dd_constraints_file, "r", encoding="utf-8") as f:
        dd_constraints = eval(f.read())
    with open(attribute_thresholds_file, "r", encoding="utf-8") as f:
        attribute_thresholds = eval(f.read())
    sorted_thresholds = extract_attributes_thresholds(dd_constraints)

    data_instance = dirty_df.to_dict(orient="records")
    length = len(data_instance)

    if length >= 400000:
        row_block_ranges = [(i, i + 1) for i in range(block_rows)]

        print("Launching block-wise parallel processing with multiprocessing.Process...")
        start_time = time.time()

        processes = []
        for start, end in row_block_ranges:
            p = Process(target=process_block_range, args=(
                start, end, data_instance, sorted_thresholds,
                dd_constraints, attribute_thresholds,
                probability_file, output_prefix, num_processes
            ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        total_time = time.time() - start_time
        print(f"All blocks completed in {total_time:.2f} seconds.")

    else:
        # If you want to regenerate the difference.csv file, you can uncomment this line.
        # generate_difference_file(dirty_file, clean_file, difference_file)
        diff_df = pd.read_csv(difference_file)
        ground_truth_errors = set(
            f"t{row['Index']}.{row['Attribute']}" for _, row in diff_df.iterrows()
        )
        print(f"Number of ground_truth_errors: {len(ground_truth_errors)}")

        dataset_scheme = list(dirty_df.columns)
        
        # If you want to regenerate the predicted_probabilities.csv file, you can uncomment this line.
        # generate_probability_files(dirty_file, probability_file, dataset_scheme)

        start_time_lookup = time.time()
        distance_lookups = build_all_distance_lookups(
            data_instance, sorted_thresholds, num_processes
        )
        build_lookup_time = time.time() - start_time_lookup
        print(f"Distance lookup table built in {build_lookup_time:.2f} seconds.")

        start_time_cluster = time.time()
        clusters = generate_clusters_parallel(
            data_instance, sorted_thresholds, distance_lookups, num_processes
        )
        cluster_time = time.time() - start_time_cluster
        print(f"Clusters generated in {cluster_time:.2f} seconds.")

        start_time_find = time.time()
        find_violating_pair, hyper_edges = find_violating_pairs_parallel(
            dd_constraints, clusters, num_processes, length, delta_n=0
        )
        find_vio_pairs_time = time.time() - start_time_find
        print(f"Violating pairs found in {find_vio_pairs_time:.2f} seconds.")

        start_time_edet = time.time()
        E_final = build_graph_parallel(
            hyper_edges, probability_file, num_processes, attribute_thresholds
        )
        edet_time = time.time() - start_time_edet
        print(f"Error detection completed in {edet_time:.2f} seconds.")

        total_time = build_lookup_time + cluster_time + find_vio_pairs_time + edet_time
        print(f"Total time for pipeline: {total_time:.2f} seconds")

        precision, recall, f1_score_val = calculate_precision_recall_f1(
            E_final, ground_truth_errors
        )
        coverage_msg = check_all_pairs_covered(find_violating_pair, E_final)

        print(f"Metrics => Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score_val:.4f}")
        print("Coverage check:", coverage_msg)

        results = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1_score_val,
            "BuildLookupTime": build_lookup_time,
            "ClusterTime": cluster_time,
            "FindVioPairsTime": find_vio_pairs_time,
            "EDetTime": edet_time,
            "TotalTime": total_time,
            "Coverage": coverage_msg,
        }
        df_results = pd.DataFrame([results])
        result_path = os.path.join(output_prefix, "result.csv")
        df_results.to_csv(result_path, index=False)
        print(f"Saved results to: {result_path}")
