import multiprocessing
from collections import defaultdict


def extract_attributes_thresholds(dd_constraints):
    attribute_thresholds = defaultdict(set)
    for condition, conclusion in dd_constraints:
        for attr, (op, threshold) in condition.items():
            attribute_thresholds[attr].add(threshold)
        conclusion_attr, (conclusion_op, conclusion_threshold) = conclusion
        attribute_thresholds[conclusion_attr].add(conclusion_threshold)

    sorted_thresholds = {
        attr: sorted(list(thresholds), reverse=True)
        for attr, thresholds in attribute_thresholds.items()
    }
    return sorted_thresholds


def build_hyper_edge_for_pair(pair, dd):
    i_idx, j_idx = pair
    left_side_dict = dd[0]
    right_side_attr, (right_op, right_threshold) = dd[1]
    edge = []

    for attr, (op, threshold) in left_side_dict.items():
        edge.append(f"t{i_idx}.{attr}")
        edge.append(f"t{j_idx}.{attr}")

    edge.append(f"t{i_idx}.{right_side_attr}")
    edge.append(f"t{j_idx}.{right_side_attr}")

    return edge


def process_dd_constraint_task(task_list, n, delta_n):
    processor_violating_pairs = set()
    processor_hyper_edges = []

    for task in task_list:
        LeftAttributes_LE = task["LeftAttributes_LE"]
        LeftAttributes_GT = task["LeftAttributes_GT"]
        attr, threshold, filtered_clusters_right, op_type = task["RightAttributes"]

        intersection_result = None
        if LeftAttributes_LE:
            for attr, threshold, filtered_clusters in LeftAttributes_LE:
                pairs = set()
                for i, cluster_l in filtered_clusters:
                    pairs.update((i, j) for j in cluster_l)
                if intersection_result is None:
                    intersection_result = pairs
                else:
                    intersection_result &= pairs

        if intersection_result:
            for attr, threshold, filtered_clusters in LeftAttributes_GT:
                for i, cluster_l in filtered_clusters:
                    difference_pairs = set((i, j) for j in cluster_l)
                    intersection_result -= difference_pairs

        if intersection_result and task["RightAttributes"]:
            attr, threshold, filtered_clusters, op_type = task["RightAttributes"]
            right_pairs = set()
            for i, cluster_l in filtered_clusters:
                right_pairs.update((i, j) for j in cluster_l)

            if op_type == 0:
                intersection_result -= right_pairs
            elif op_type == 1:
                intersection_result &= right_pairs

        if intersection_result:
            processor_violating_pairs.update(intersection_result)

    return {
        "violating_pairs": processor_violating_pairs,
    }


def find_violating_pairs_parallel(dd_constraints, clusters, num_processes, n, delta_n):
    all_violating_pairs = set()
    all_hyper_edges = []
    task_info = []
    for idx, dd in enumerate(dd_constraints):
        tasks = [[] for _ in range(num_processes)]

        LeftAttributes_LE = [
            (attr, threshold) for attr, (op, threshold) in dd[0].items() if op == "<="
        ]
        LeftAttributes_GT = [
            (attr, threshold) for attr, (op, threshold) in dd[0].items() if op == ">"
        ]
        RightAttribute, (right_op, right_threshold) = dd[1]

        min_max_i = float("inf")

        if min_max_i == float("inf"):
            min_max_i = n + delta_n - 2

        for processor_id in range(num_processes):
            processor_task = {
                "LeftAttributes_LE": [],
                "LeftAttributes_GT": [],
                "RightAttributes": None,
            }

            if LeftAttributes_LE:
                for attr, threshold in LeftAttributes_LE:
                    if attr in clusters and threshold in clusters[attr]:
                        attribute_clusters = clusters[attr][threshold]
                        filtered_clusters = [
                            (i, cluster_l)
                            for i, cluster_l in attribute_clusters
                            if i >= processor_id
                            and i <= min_max_i
                            and (i - processor_id) % num_processes == 0
                        ]

                        processor_task["LeftAttributes_LE"].append(
                            (attr, threshold, filtered_clusters)
                        )

            if not LeftAttributes_LE and delta_n == 0:
                all_pairs = {
                    (i, j)
                    for i in range(n)
                    for j in range(i + 1, n)
                    if i >= processor_id
                    and i <= min_max_i
                    and (i - processor_id) % num_processes == 0
                }
                processor_task["LeftAttributes_LE"].append(
                    ("all_pairs", 0, [(i, [j]) for i, j in all_pairs])
                )
            elif not LeftAttributes_LE and delta_n > 0:
                incremental_pairs = {
                    (i, j)
                    for i in range(n)
                    for j in range(n, n + delta_n)
                    if i >= processor_id
                    and i <= min_max_i
                    and (i - processor_id) % num_processes == 0
                }
                incremental_pairs.update(
                    {
                        (i, j)
                        for i in range(n, n + delta_n)
                        for j in range(i + 1, n + delta_n)
                        if i >= processor_id
                        and i <= min_max_i
                        and (i - processor_id) % num_processes == 0
                    }
                )
                processor_task["LeftAttributes_LE"].append(
                    ("incremental_pairs", 0, [(i, [j]) for i, j in incremental_pairs])
                )

            if LeftAttributes_GT:
                for attr, threshold in LeftAttributes_GT:
                    if attr in clusters and threshold in clusters[attr]:
                        attribute_clusters = clusters[attr][threshold]
                        filtered_clusters = [
                            (i, cluster_l)
                            for i, cluster_l in attribute_clusters
                            if i >= processor_id
                            and i <= min_max_i
                            and (i - processor_id) % num_processes == 0
                        ]
                        processor_task["LeftAttributes_GT"].append(
                            (attr, threshold, filtered_clusters)
                        )

            if (
                RightAttribute in clusters
                and right_threshold in clusters[RightAttribute]
            ):
                attribute_clusters = clusters[RightAttribute][right_threshold]
                filtered_clusters = [
                    (i, cluster_l)
                    for i, cluster_l in attribute_clusters
                    if i >= processor_id
                    and i <= min_max_i
                    and (i - processor_id) % num_processes == 0
                ]
                op_type = 0 if dd[1][1][0] == "<=" else 1
                processor_task["RightAttributes"] = (
                    RightAttribute,
                    right_threshold,
                    filtered_clusters,
                    op_type,
                )

            tasks[processor_id].append(processor_task)

        task_info.append(
            {
                "Constraint": idx,
                "Tasks": tasks,
            }
        )

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                process_dd_constraint_task,
                [(task_list, n, delta_n) for task_list in tasks],
            )

        violating_pairs = set()
        for result in results:
            violating_pairs.update(result["violating_pairs"])

        hyper_edges = []
        for pair in violating_pairs:
            edge = build_hyper_edge_for_pair(pair, dd)
            hyper_edges.append(edge)
        all_violating_pairs.update(violating_pairs)
        all_hyper_edges.extend(hyper_edges)

        print(f"Constraint {idx} violating pairs: {len(violating_pairs)}")

    return (
        all_violating_pairs,
        all_hyper_edges,
    )
