import multiprocessing
import time

import pandas as pd

g_data_instance = None
g_distance_lookups = None
g_attributes = None
g_sorted_thresholds = None
g_n_old = 0
g_n_total = 0


def init_child_process(data_inst, dist_lookups, attributes, sorted_thres):
    global g_data_instance, g_distance_lookups, g_attributes, g_sorted_thresholds
    g_data_instance = data_inst
    g_distance_lookups = dist_lookups
    g_attributes = attributes
    g_sorted_thresholds = sorted_thres


def init_child_process_incremental(
    all_data_inst, dist_lookups, attributes, sorted_thres, n_old
):
    global g_data_instance, g_distance_lookups, g_attributes, g_sorted_thresholds
    global g_n_old, g_n_total

    g_data_instance = all_data_inst
    g_distance_lookups = dist_lookups
    g_attributes = attributes
    g_sorted_thresholds = sorted_thres
    g_n_old = n_old
    g_n_total = len(all_data_inst)


def compute_clusters_for_chunk(row_indices):
    global g_data_instance, g_distance_lookups, g_attributes, g_sorted_thresholds
    local_clusters = {}
    n = len(g_data_instance)

    for attr in g_attributes:
        max_threshold = g_sorted_thresholds[attr][0]
        if attr not in local_clusters:
            local_clusters[attr] = {}
        local_clusters[attr][max_threshold] = []

    for i in row_indices:
        val_i_dict = g_data_instance[i]
        for attr in g_attributes:
            max_threshold = g_sorted_thresholds[attr][0]
            val_i = val_i_dict[attr]
            if pd.isnull(val_i):
                continue

            cluster_l = []
            for j in range(i + 1, n):
                val_j = g_data_instance[j][attr]
                if pd.isnull(val_j):
                    continue

                dist = g_distance_lookups[attr].get((val_i, val_j), None)
                if dist is None:
                    dist = g_distance_lookups[attr].get((val_j, val_i), float("inf"))
                if dist <= max_threshold:
                    cluster_l.append(j)

            if cluster_l:
                local_clusters[attr][max_threshold].append((i, cluster_l))

    return local_clusters


def compute_clusters_for_chunk_incremental(row_indices):
    global g_data_instance, g_distance_lookups, g_attributes, g_sorted_thresholds
    global g_n_old, g_n_total

    local_clusters = {}
    for attr in g_attributes:
        max_threshold = g_sorted_thresholds[attr][0]
        if attr not in local_clusters:
            local_clusters[attr] = {}
        local_clusters[attr][max_threshold] = []

    for i in row_indices:
        if i < g_n_old:
            start_j = g_n_old
            end_j = g_n_total
        else:
            start_j = i + 1
            end_j = g_n_total

        val_i_dict = g_data_instance[i]
        for attr in g_attributes:
            max_threshold = g_sorted_thresholds[attr][0]
            val_i = val_i_dict[attr]
            if pd.isnull(val_i):
                continue

            cluster_l = []
            for j in range(start_j, end_j):
                val_j = g_data_instance[j][attr]
                if pd.isnull(val_j):
                    continue

                dist = g_distance_lookups[attr].get((val_i, val_j), None)
                if dist is None:
                    dist = g_distance_lookups[attr].get((val_j, val_i))
                if dist is not None and dist <= max_threshold:
                    cluster_l.append(j)

            if cluster_l:
                local_clusters[attr][max_threshold].append((i, cluster_l))

    return local_clusters


def generate_clusters_parallel(
    data_instance, sorted_thresholds, distance_lookups, num_processes
):
    n = len(data_instance)
    attributes = list(sorted_thresholds.keys())

    row_splits = [[] for _ in range(num_processes)]
    for i in range(n - 1):
        proc_id = i % num_processes
        row_splits[proc_id].append(i)

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_child_process,
        initargs=(data_instance, distance_lookups, attributes, sorted_thresholds),
    ) as pool:
        results = pool.map(compute_clusters_for_chunk, row_splits)

    clusters = {}
    for attr in attributes:
        max_threshold = sorted_thresholds[attr][0]
        clusters[attr] = {max_threshold: []}

    for partial_clusters in results:
        for attr, threshold_dict in partial_clusters.items():
            for threshold, cluster_list in threshold_dict.items():
                clusters[attr][threshold].extend(cluster_list)

    for attr in attributes:
        thresholds = sorted_thresholds[attr]
        if len(thresholds) <= 1:
            continue

        max_threshold = thresholds[0]
        attribute_clusters = {max_threshold: clusters[attr][max_threshold]}

        for t in thresholds[1:]:
            new_clusters = []
            prev_clusters = attribute_clusters[max_threshold]
            for i, cluster_j_list in prev_clusters:
                val_i = data_instance[i][attr]
                if pd.isnull(val_i):
                    continue

                filtered_j_list = []
                for j in cluster_j_list:
                    val_j = data_instance[j][attr]
                    if pd.isnull(val_j):
                        continue
                    dist = distance_lookups[attr].get((val_i, val_j), None)
                    if dist is None:
                        dist = distance_lookups[attr].get((val_j, val_i), float("inf"))
                    if dist <= t:
                        filtered_j_list.append(j)

                if filtered_j_list:
                    new_clusters.append((i, filtered_j_list))

            attribute_clusters[t] = new_clusters
            max_threshold = t

        clusters[attr] = attribute_clusters

    return clusters


def generate_clusters_parallel_incremental(
    original_data, delta_data, sorted_thresholds, distance_lookups, num_processes
):
    if isinstance(original_data, pd.DataFrame):
        original_data = original_data.to_dict(orient="records")
    if isinstance(delta_data, pd.DataFrame):
        delta_data = delta_data.to_dict(orient="records")

    n_old = len(original_data)
    n_new = len(delta_data)
    all_data = original_data + delta_data
    n_total = n_old + n_new
    attributes = list(sorted_thresholds.keys())

    row_splits = [[] for _ in range(num_processes)]
    for i in range(n_total - 1):
        proc_id = i % num_processes
        row_splits[proc_id].append(i)

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_child_process_incremental,
        initargs=(all_data, distance_lookups, attributes, sorted_thresholds, n_old),
    ) as pool:
        results = pool.map(compute_clusters_for_chunk_incremental, row_splits)

    clusters = {}
    for attr in attributes:
        max_threshold = sorted_thresholds[attr][0]
        clusters[attr] = {max_threshold: []}

    for partial_clusters in results:
        for attr, threshold_dict in partial_clusters.items():
            for threshold, cluster_list in threshold_dict.items():
                clusters[attr][threshold].extend(cluster_list)

    for attr in attributes:
        thresholds = sorted_thresholds[attr]
        if len(thresholds) <= 1:
            continue

        max_threshold = thresholds[0]
        attribute_clusters = {max_threshold: clusters[attr][max_threshold]}

        for t in thresholds[1:]:
            new_clusters = []
            prev_clusters = attribute_clusters[max_threshold]
            for i, cluster_j_list in prev_clusters:
                val_i = all_data[i][attr]
                if pd.isnull(val_i):
                    continue

                filtered_j_list = []
                for j in cluster_j_list:
                    val_j = all_data[j][attr]
                    if pd.isnull(val_j):
                        continue

                    dist = distance_lookups[attr].get((val_i, val_j), None)
                    if dist is None:
                        dist = distance_lookups[attr].get((val_j, val_i), float("inf"))
                    if dist <= t:
                        filtered_j_list.append(j)

                if filtered_j_list:
                    new_clusters.append((i, filtered_j_list))

            attribute_clusters[t] = new_clusters
            max_threshold = t

        clusters[attr] = attribute_clusters

    return clusters
