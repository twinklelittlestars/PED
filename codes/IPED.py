import os
import time

import pandas as pd
import xlsxwriter
from MWVC import build_graph_parallel
from utils.cluster_utils import (
    generate_clusters_parallel,
    generate_clusters_parallel_incremental,
)
from utils.dd_utils import (
    build_hyper_edge_for_pair,
    extract_attributes_thresholds,
    find_violating_pairs_parallel,
)
from utils.difference import generate_difference_file
from utils.distance_utils import (
    build_all_distance_lookups,
    incremental_build_all_distance_lookups,
)
from utils.evaluation_utils import (
    calculate_precision_recall_f1,
    check_all_pairs_covered,
)
from utils.preprocessing import generate_probability_files

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Read dd_constraints from a text file
    # ------------------------------------------------------------------
    dd_constraints_file = "Incremental_data\dd_constraints.txt"
    with open(dd_constraints_file, "r", encoding="utf-8") as f:
        dd_constraints = eval(f.read())

    # ------------------------------------------------------------------
    # 2) Read attribute_thresholds from a text file
    # ------------------------------------------------------------------
    attribute_thresholds_file = "Incremental_data\attribute_thresholds.txt"
    with open(attribute_thresholds_file, "r", encoding="utf-8") as f:
        attribute_thresholds = eval(f.read())

    # ------------------------------------------------------------------
    # 3) Number of processes
    # ------------------------------------------------------------------
    FIXED_NUM_PROCESSES = 16

    # ------------------------------------------------------------------
    # 4) Extract thresholds from dd_constraints
    # ------------------------------------------------------------------
    start_time = time.time()
    sorted_thresholds = extract_attributes_thresholds(dd_constraints)
    print("Time for extracting thresholds:", time.time() - start_time)

    # ------------------------------------------------------------------
    # 5) Data size settings for incremental experiments
    # ------------------------------------------------------------------
    data_pairs = [
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
        (100, 120),
        (120, 140),
        (140, 160),
        (160, 180),
        (180, 200),
    ]

    results_summary = []

    # ------------------------------------------------------------------
    # 6) Main loop over data_pairs
    # ------------------------------------------------------------------
    for base_size, inc_size in data_pairs:
        print(
            f"\n========== Processing base: {base_size}k → incremental: {inc_size}k =========="
        )

        # The directory for this incremental step
        base_dir = os.path.join("Incremental_data", f"Soccer_{inc_size}k")

        # Key files: throughout data + difference
        throughout_dirty_file = os.path.join(base_dir, "throughout_dirty.csv")
        throughout_clean_file = os.path.join(base_dir, "throughout_clean.csv")
        difference_file = os.path.join(base_dir, "difference.csv")

        print("Generating difference file for throughout dataset...")
        generate_difference_file(
            throughout_dirty_file, throughout_clean_file, difference_file
        )

        # Read difference file to get ground-truth errors
        difference_data = pd.read_csv(difference_file)
        ground_truth_errors = set(
            f"t{row['Index']}.{row['Attribute']}"
            for _, row in difference_data.iterrows()
        )
        print(f"Number of ground_truth_errors: {len(ground_truth_errors)}")

        # ------------------------------------------------------------------
        # (Optional) Generate a probability file from throughout_dirty.csv
        # ------------------------------------------------------------------
        print("Generating probability file from throughout dataset...")
        # 1) Derive dataset scheme from 'throughout_dirty.csv' columns
        throughout_df = pd.read_csv(throughout_dirty_file, dtype=str)
        dataset_scheme = list(throughout_df.columns)

        # 2) Probability file
        throughout_prob_file = os.path.join(base_dir, "predicted_probabilities.csv")

        # 3) Generate probability CSV
        generate_probability_files(
            throughout_dirty_file, throughout_prob_file, dataset_scheme
        )

        # ------------------------------------------------------------------
        # Now handle base portion
        # ------------------------------------------------------------------
        base_dirty_file = os.path.join(base_dir, "dirty_base.csv")
        base_df = pd.read_csv(base_dirty_file)
        data_instance = base_df.to_dict(orient="records")
        length_base = len(data_instance)
        print(f"Base data has {length_base} rows")

        st1 = time.time()
        distance_lookups = build_all_distance_lookups(
            data_instance, sorted_thresholds, FIXED_NUM_PROCESSES
        )
        et1 = time.time()
        build_lookup_time_base = et1 - st1

        st2 = time.time()
        clusters_base = generate_clusters_parallel(
            data_instance, sorted_thresholds, distance_lookups, FIXED_NUM_PROCESSES
        )
        et2 = time.time()
        cluster_time_base = et2 - st2

        st3 = time.time()
        find_violating_pair_base, hyper_edges_base = find_violating_pairs_parallel(
            dd_constraints, clusters_base, FIXED_NUM_PROCESSES, length_base, delta_n=0
        )
        et3 = time.time()
        find_vio_pairs_time_base = et3 - st3

        st4 = time.time()
        # pass throughout_prob_file to build_graph_parallel as the "probabilities_file"
        E_final_base = build_graph_parallel(
            hyper_edges_base,
            throughout_prob_file,  # use the newly generated prob CSV
            FIXED_NUM_PROCESSES,
            attribute_thresholds=attribute_thresholds,
        )
        et4 = time.time()
        edet_time_base = et4 - st4

        total_time_base = (
            build_lookup_time_base
            + cluster_time_base
            + find_vio_pairs_time_base
            + edet_time_base
        )
        print(f"Base data finished, total time: {total_time_base:.2f} seconds")

        # ------------------------------------------------------------------
        # Now handle incremental portion
        # ------------------------------------------------------------------
        inc_dirty_file = os.path.join(base_dir, "incremental_dirty.csv")
        inc_df = pd.read_csv(inc_dirty_file)
        inc_data = inc_df.to_dict(orient="records")
        length_inc = len(inc_data)
        print(f"Incremental data has {length_inc} rows")

        st5 = time.time()
        distance_lookups_updated = incremental_build_all_distance_lookups(
            distance_lookups, sorted_thresholds, inc_data, FIXED_NUM_PROCESSES
        )
        et5 = time.time()
        build_delta_lookup_time_inc = et5 - st5

        st6 = time.time()
        clusters_inc = generate_clusters_parallel_incremental(
            data_instance,
            inc_data,
            sorted_thresholds,
            distance_lookups_updated,
            FIXED_NUM_PROCESSES,
        )
        et6 = time.time()
        cluster_time_inc = et6 - st6

        st7 = time.time()
        find_violating_pair_inc, hyper_edges_inc = find_violating_pairs_parallel(
            dd_constraints,
            clusters_inc,
            FIXED_NUM_PROCESSES,
            length_base,
            delta_n=length_inc,
        )
        et7 = time.time()
        find_vio_pairs_time_inc = et7 - st7

        st8 = time.time()
        # again pass throughout_prob_file to build_graph_parallel
        E_final_inc = build_graph_parallel(
            hyper_edges_inc,
            throughout_prob_file,
            FIXED_NUM_PROCESSES,
            attribute_thresholds=attribute_thresholds,
        )
        et8 = time.time()
        edet_time_inc = et8 - st8

        # Combine
        E_final_total = E_final_base.union(E_final_inc)
        total_time_inc = (
            build_delta_lookup_time_inc
            + cluster_time_inc
            + find_vio_pairs_time_inc
            + edet_time_inc
        )
        print(f"Incremental total time({inc_size}k): {total_time_inc:.2f} seconds")

        # Evaluate
        precision, recall, f1_score_val = calculate_precision_recall_f1(
            E_final_total, ground_truth_errors
        )
        print(
            f"Final combined metrics => P={precision:.4f}, R={recall:.4f}, F1={f1_score_val:.4f}"
        )

        results_summary.append(
            {
                "Base_Size(k)": base_size,
                "Inc_Size(k)": inc_size,
                "Length_Original": length_base,
                "Length_Incremental": length_inc,
                "Time_Original_Total": total_time_base,
                "Time_Inc_Total": total_time_inc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1_score_val,
            }
        )

    # ------------------------------------------------------------------
    # Summarize results
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results_summary)
    print("\n====== All Results Summary ======")
    print(df_results)

    excel_file = "incremental_final_results.xlsx"
    with xlsxwriter.Workbook(excel_file) as writer:
        df_results.to_excel(writer, index=False, sheet_name="Incremental")
        workbook = writer.book
        worksheet = writer.sheets["Incremental"]
        float_format = workbook.add_format({"num_format": "0.0000"})
        text_format = workbook.add_format({"num_format": "@"})
        for idx, col in enumerate(df_results.columns):
            if col in [
                "Time_Original_Total",
                "Time_Inc_Total",
                "Precision",
                "Recall",
                "F1",
            ]:
                worksheet.set_column(idx, idx, 18, float_format)
            else:
                worksheet.set_column(idx, idx, 14, text_format)

    print(f"\nThe final incremental results have been saved to '{excel_file}'")
