import ctypes
import itertools
import multiprocessing
import os
import re
import string
import time
from collections import defaultdict

import EDet

# import dill
import numpy as np
import pandas as pd
from MWVC import build_graph_parallel

# multiprocessing.set_start_method("spawn", force=True)


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


# STRING_COLUMNS = {"name", "dept", "manager"}
# INT_COLUMNS = {"salary"}
STRING_COLUMNS = {
    "src",
    "flight",
    "schedDepTime",
    "actDepTime",
    "schedArrTime",
    "actArrTime",
}
INT_COLUMNS = {
    "id",
}

# STRING_COLUMNS = {
#     "name",
#     "surname",
#     "birthyear",
#     "birthplace",
#     "position",
#     "team",
#     "city",
#     "stadium",
#     "manager",
# }
# INT_COLUMNS = {"season"}
# STRING_COLUMNS = {
#     "HospitalName",
#     "Address1",
#     "Address2",
#     "Address3",
#     "City",
#     "State",
#     "CountyName",
#     "HospitalType",
#     "HospitalOwner",
#     "EmergencyService",
#     "Condition",
#     "MeasureCode",
#     "MeasureName",
#     "Score",
#     "Sample",
#     "Stateavg",
#     "ProviderNumber",
#     "ZipCode",
#     "PhoneNumber",
# }
# INT_COLUMNS = {
#     # "ProviderNumber",
#     # "ZipCode",
#     # "PhoneNumber",
#     "name",
# }
# STRING_COLUMNS = {
#     "name",
#     "surname",
#     "birthyear",
#     "birthplace",
#     "position",
#     "team",
#     "city",
#     "stadium",
#     "manager",
# }
# INT_COLUMNS = {"season"}


def clean_value(value):
    if isinstance(value, str):
        # value = re.sub(r"\(.*?\)", "", value)
        value = value.strip()
    return value


def preprocess_values(values):
    return [clean_value(value) for value in values]


def calculate_difference(value1, value2, col_name):
    # value1 = clean_value(value1) if isinstance(value1, str) else str(value1)
    # value2 = clean_value(value2) if isinstance(value2, str) else str(value2)
    # value1 = 0 if pd.isna(value1) else value1
    # value2 = 0 if pd.isna(value2) else value2

    if col_name in STRING_COLUMNS:
        value1 = str(value1)  # 确保是字符串
        value2 = str(value2)
        return levenshtein_distance(value1, value2)
    if col_name in INT_COLUMNS:
        return abs(int(float(value1)) - int(float(value2)))

    raise ValueError(f"Column '{col_name}' has an unknown data type.")


# 使用动态规划矩阵的滚动数组优化方法，仅存储两列，降低空间复杂度
def levenshtein_distance(str1, str2):
    str1, str2 = str1.replace(" ", ""), str2.replace(" ", "")
    len1, len2 = len(str1), len(str2)
    if len1 > len2:
        str1, str2 = str2, str1
        len1, len2 = len2, len1

    prev_row = list(range(len1 + 1))
    current_row = [0] * (len1 + 1)

    for j in range(1, len2 + 1):
        current_row[0] = j
        for i in range(1, len1 + 1):
            if str1[i - 1] == str2[j - 1]:
                current_row[i] = prev_row[i - 1]
            else:
                current_row[i] = 1 + min(
                    prev_row[i], current_row[i - 1], prev_row[i - 1]
                )

        prev_row, current_row = current_row, prev_row

    return prev_row[len1]


# print(calculate_difference("YDPVPBFCCKXE", "MONSSRUHUOME", "name"))

# print(levenshtein_distance("YDPVPBFCCKXE", "MONSSRUHUOME"))


# 并行计算去重后的值列表的距离
def calculate_unique_distances(attribute, unique_values, num_processes):
    unique_values = [v for v in unique_values if pd.notnull(v)]
    distance_lookup = {}
    num_unique = len(unique_values)

    # 获取所有右上三角的坐标
    coordinate_pairs = [(i, j) for i in range(num_unique) for j in range(i, num_unique)]

    chunk_size = len(coordinate_pairs) // num_processes + (
        len(coordinate_pairs) % num_processes > 0
    )
    coordinate_chunks = [
        coordinate_pairs[i : i + chunk_size]
        for i in range(0, len(coordinate_pairs), chunk_size)
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            calculate_distances_for_coordinates,
            [(attribute, unique_values, chunk) for chunk in coordinate_chunks],
        )

    for partial_lookup in results:
        distance_lookup.update(partial_lookup)

    return distance_lookup


def calculate_distances_for_coordinates(attribute, unique_values, coordinate_chunk):
    distance_lookup = {}

    for i, j in coordinate_chunk:
        value1, value2 = unique_values[i], unique_values[j]
        if pd.isnull(value1) or pd.isnull(value2):
            continue

        if i == j:
            distance = 0
        else:
            distance = calculate_difference(value1, value2, attribute)
        distance_lookup[(value1, value2)] = distance
        # distance_lookup[(value2, value1)] = distance  # 对称存储

    return distance_lookup


# 构建所有属性的距离查找表
def build_all_distance_lookups(data_instance, sorted_thresholds, num_processes):
    distance_lookups = {}
    attributes = list(sorted_thresholds.keys())

    for attribute in attributes:
        values = preprocess_values([row[attribute] for row in data_instance])
        unique_values = list(set(values))
        start_time = time.time()
        distance_lookup = calculate_unique_distances(
            attribute, unique_values, num_processes
        )
        # end_time = time.time()
        # total_time += end_time - start_time

        distance_lookups[attribute] = distance_lookup

    return distance_lookups


# 全局变量，供子进程访问
g_data_instance = None
g_distance_lookups = None
g_attributes = None
g_sorted_thresholds = None
g_n_old = 0
g_n_total = 0


def init_child_process(data_inst, dist_lookups, attributes, sorted_thres):
    # 把大对象存到全局变量（子进程内存空间）
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
    g_n_total = len(all_data_inst)  # old_data + delta_data 的总行数


def generate_clusters_parallel(
    data_instance, sorted_thresholds, distance_lookups, num_processes
):
    n = len(data_instance)
    attributes = list(sorted_thresholds.keys())

    # 1. 切分行号，将行索引以n为梯度平均划分给各个处理器
    row_splits = [[] for _ in range(num_processes)]
    for i in range(n - 1):
        proc_id = i % num_processes
        row_splits[proc_id].append(i)

    # 2. 并行计算
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.starmap(
    #         compute_local_clusters_for_rows,
    #         [
    #             (
    #                 data_instance,
    #                 rows_for_proc,
    #                 attributes,
    #                 sorted_thresholds,
    #                 distance_lookups,
    #             )
    #             for rows_for_proc in row_splits
    #         ],
    #     )

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_child_process,
        initargs=(data_instance, distance_lookups, attributes, sorted_thresholds),
    ) as pool:
        results = pool.map(compute_clusters_for_chunk, row_splits)

    # 3. 合并结果
    clusters = {}
    # pairs_dict = {}
    for attr in attributes:
        max_threshold = sorted_thresholds[attr][0]
        if attr not in clusters:
            clusters[attr] = {}
        clusters[attr][max_threshold] = []
        # pairs_dict[attr] = []

    for partial_clusters in results:
        for attr, threshold_dict in partial_clusters.items():
            for threshold, cluster_list in threshold_dict.items():
                clusters[attr][threshold].extend(cluster_list)

        # for attr, triple_list in partial_pairs.items():
        #     pairs_dict[attr].extend(triple_list)

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


def compute_clusters_for_chunk(row_indices):
    global g_data_instance, g_distance_lookups, g_attributes, g_sorted_thresholds
    local_clusters = {}
    # local_pairs = {}
    n = len(g_data_instance)

    for attr in g_attributes:
        max_threshold = g_sorted_thresholds[attr][0]
        if attr not in local_clusters:
            local_clusters[attr] = {}
        local_clusters[attr][max_threshold] = []
        # local_pairs[attr] = []

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
                    # local_pairs[attr].append((i, j, dist))

            if cluster_l:
                local_clusters[attr][max_threshold].append((i, cluster_l))

    return local_clusters


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

    # 1. 行索引切分
    n_total = n_old + n_new
    row_splits = [[] for _ in range(num_processes)]
    for i in range(n_total - 1):
        proc_id = i % num_processes
        row_splits[proc_id].append(i)

    # 2. 并行计算
    attributes = list(sorted_thresholds.keys())

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_child_process_incremental,
        initargs=(all_data, distance_lookups, attributes, sorted_thresholds, n_old),
    ) as pool:
        results = pool.map(compute_clusters_for_chunk_incremental, row_splits)

    # 3. 合并结果: 先把每个属性最大阈值的结果拼在一起
    clusters = {}
    for attr in attributes:
        max_threshold = sorted_thresholds[attr][0]
        if attr not in clusters:
            clusters[attr] = {}
        clusters[attr][max_threshold] = []

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
        # if i >= g_n_total - 1:
        #     continue

        # 如果 i < g_n_old，说明是老数据行，只与新数据行 (g_n_old ~ g_n_total-1) 进行配对
        if i < g_n_old:
            start_j = g_n_old
            end_j = g_n_total
        else:
            # 否则 i 是新数据行，则与 [i+1, g_n_total) 做配对 (新 vs 新)
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

                if dist <= max_threshold:
                    cluster_l.append(j)

            if cluster_l:
                local_clusters[attr][max_threshold].append((i, cluster_l))

    return local_clusters


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


def find_violating_pairs_parallel(dd_constraints, clusters, num_processes, n, delta_n):
    # violating_pairs_phi1: set[tuple[int, int]] = set()
    # violating_pairs_phi2: set[tuple[int, int]] = set()
    # violating_pairs_phi3: set[tuple[int, int]] = set()
    # violating_pairs_phi4: set[tuple[int, int]] = set()
    # violating_pairs_phi5: set[tuple[int, int]] = set()
    # violating_pairs_phi6: set[tuple[int, int]] = set()
    # 记录全部DD约束违反的对变量
    all_violating_pairs = set()
    all_hyper_edges = []  # 用于存储所有超边
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
        # for attr, threshold in LeftAttributes_LE:
        #     # print(f"clusters[{attr}][{threshold}] = {clusters[attr][threshold]}")

        #     if (
        #         attr in clusters
        #         and threshold in clusters[attr]
        #         and clusters[attr][threshold]
        #     ):
        #         last_cluster = clusters[attr][threshold][-1]
        #         max_i = last_cluster[0]
        #         min_max_i = min(min_max_i, max_i)

        if min_max_i == float("inf"):
            min_max_i = n + delta_n - 2

        for processor_id in range(num_processes):
            processor_task = {
                "LeftAttributes_LE": [],
                "LeftAttributes_GT": [],
                "RightAttributes": None,
            }

            # 划分第一类属性
            # 如果有第一类属性才处理
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

                        # if filtered_clusters:
                        processor_task["LeftAttributes_LE"].append(
                            (attr, threshold, filtered_clusters)
                        )

            if not LeftAttributes_LE and delta_n == 0:
                # indices = range(n)  # 元组索引范围
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
                # intersection_result分为两部分，一部分的元组对索引组合是：i的取值是[0,n-1],j的取值是[n,n+delta_n-1]，另一部分的元组对索引组合是：i的取值是[n,n+delta_n-1],j的取值是[i+1,n+delta_n-1]
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

            # 划分第二类属性
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
                        # print(
                        #     f"当前处理器划分的左侧属性任务数量是：{len(filtered_clusters)}"
                        # )
                        # if filtered_clusters:
                        processor_task["LeftAttributes_GT"].append(
                            (attr, threshold, filtered_clusters)
                        )

            # 划分第三类属性（右边属性）
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
                # print(f"当前处理器划分的右侧属性任务数量是：{len(filtered_clusters)}")
                # if filtered_clusters:
                op_type = 0 if dd[1][1][0] == "<=" else 1
                processor_task["RightAttributes"] = (
                    RightAttribute,
                    right_threshold,
                    filtered_clusters,
                    op_type,
                )

            # 添加到对应处理器的任务列表
            tasks[processor_id].append(processor_task)

        task_info.append(
            {
                "Constraint": idx,
                "Tasks": tasks,
            }
        )
        # print(f"Constraint {idx} tasks: {len(tasks)}")

        # 使用多进程并行处理任务
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                process_dd_constraint_task,
                [(task_list, n, delta_n) for task_list in tasks],
            )
            # pool.starmap(
            #     calculate_distances_for_coordinates,
            #     [(attribute, unique_values, chunk) for chunk in coordinate_chunks],
            # )
        # print(f"Constraint {idx} results: {len(results)}")
        # 合并所有处理器的结果
        violating_pairs = set()
        # hyper_edges = []  # 当前约束的超边集合
        for result in results:
            violating_pairs.update(result["violating_pairs"])
            # hyper_edges.extend(result["hyper_edges"])
            # print(f"Violating pairs: {violating_pairs}")

        hyper_edges = []
        for pair in violating_pairs:
            # 根据本约束 dd，构造对应超边
            edge = build_hyper_edge_for_pair(pair, dd)
            hyper_edges.append(edge)
        # print(f"Constraint {idx} violating pairs: {len(violating_pairs)}")
        all_violating_pairs.update(violating_pairs)
        all_hyper_edges.extend(hyper_edges)
        # print(f"All violating pairs: {all_violating_pairs}")

        print(f"Constraint {idx} violating pairs: {len(violating_pairs)}")
        # if idx == 0:
        #     violating_pairs_phi1 = violating_pairs
        # elif idx == 1:
        #     violating_pairs_phi2 = violating_pairs
        # elif idx == 2:
        #     violating_pairs_phi3 = violating_pairs
        # elif idx == 3:
        #     violating_pairs_phi4 = violating_pairs
        # elif idx == 4:
        #     violating_pairs_phi5 = violating_pairs
        # elif idx == 5:
        #     violating_pairs_phi6 = violating_pairs
        # print(f"违反的元组对是: {violating_pairs}")

    return (
        all_violating_pairs,
        all_hyper_edges,
        # violating_pairs_phi1,
        # violating_pairs_phi2,
        # violating_pairs_phi3,
        # violating_pairs_phi4,
        # violating_pairs_phi5,
        # violating_pairs_phi6,
    )


def process_dd_constraint_task(task_list, n, delta_n):
    processor_violating_pairs = set()
    processor_hyper_edges = []
    # print(f"当前处理的任务是：{task_list}")

    for task in task_list:
        LeftAttributes_LE = task["LeftAttributes_LE"]
        LeftAttributes_GT = task["LeftAttributes_GT"]
        attr, threshold, filtered_clusters_right, op_type = task["RightAttributes"]

        # 左边聚类交集计算
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
                # del pairs
        # print(f"当前计算得到的违反元组对数量是：{len(intersection_result)}")
        # 左边聚类差集计算
        if intersection_result:
            for attr, threshold, filtered_clusters in LeftAttributes_GT:
                for i, cluster_l in filtered_clusters:
                    difference_pairs = set((i, j) for j in cluster_l)
                    intersection_result -= difference_pairs
                    # del difference_pairs
        # print(f"当前计算得到的违反元组对数量是：{len(intersection_result)}")

        # 右属性交集计算
        if intersection_result and task["RightAttributes"]:
            attr, threshold, filtered_clusters, op_type = task["RightAttributes"]
            right_pairs = set()
            for i, cluster_l in filtered_clusters:
                right_pairs.update((i, j) for j in cluster_l)

            if op_type == 0:  # <= 操作符
                intersection_result -= right_pairs
            elif op_type == 1:  # > 操作符
                intersection_result &= right_pairs
            # del right_pairs
        # print(f"当前计算得到的违反元组对数量是：{len(intersection_result)}")

        # 添加结果
        if intersection_result:
            processor_violating_pairs.update(intersection_result)
    # print("当前处理器处理的违反元组对数量是: ", len(processor_violating_pairs))

    # for pair in intersection_result:
    #     # 构造超边，包含约束中的所有属性
    #     edge = []
    #     for attr_le, _, _ in task["LeftAttributes_LE"]:
    #         if attr_le != "all_pairs":
    #             edge.append(f"t{pair[0]}.{attr_le}")
    #             edge.append(f"t{pair[1]}.{attr_le}")
    #     for attr_gt, _, _ in task["LeftAttributes_GT"]:
    #         edge.append(f"t{pair[0]}.{attr_gt}")
    #         edge.append(f"t{pair[1]}.{attr_gt}")
    #     # 明确右属性
    #     right_attr = task["RightAttributes"][
    #         0
    #     ]  # 从RightAttributes元组中获取属性名
    #     edge.append(f"t{pair[0]}.{right_attr}")
    #     edge.append(f"t{pair[1]}.{right_attr}")
    #     processor_hyper_edges.append(edge)
    # print(f"Processor violating pairs: {processor_violating_pairs}")

    # print(f"Processor violating pairs: {processor_violating_pairs}")

    return {
        "violating_pairs": processor_violating_pairs,
        # "hyper_edges": processor_hyper_edges,
    }


# def convert_income(income):
#     return 1 if income == ">50K" else 0


def incremental_calculate_unique_distances(
    attribute, original_unique_values, delta_unique_values, num_processes
):
    original_unique_values = [v for v in original_unique_values if pd.notnull(v)]
    delta_unique_values = [v for v in delta_unique_values if pd.notnull(v)]

    all_unique_values = original_unique_values + delta_unique_values
    distance_lookup = {}
    num_original = len(original_unique_values)
    num_delta = len(delta_unique_values)

    coordinate_pairs_1 = [
        (i, j)
        for i in range(num_original)
        for j in range(num_original, num_original + num_delta)
    ]
    coordinate_pairs_2 = [
        (i, j)
        for i in range(num_original, num_original + num_delta)
        for j in range(i, num_original + num_delta)
    ]

    coordinate_pairs = coordinate_pairs_1 + coordinate_pairs_2
    chunk_size = len(coordinate_pairs) // num_processes + (
        len(coordinate_pairs) % num_processes > 0
    )
    coordinate_chunks = [
        coordinate_pairs[i : i + chunk_size]
        for i in range(0, len(coordinate_pairs), chunk_size)
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            calculate_distances_for_coordinates,
            [(attribute, all_unique_values, chunk) for chunk in coordinate_chunks],
        )

    for partial_lookup in results:
        distance_lookup.update(partial_lookup)

    return distance_lookup


def incremental_build_all_distance_lookups(
    original_distance_lookups, sorted_thresholds, delta_data, num_processes
):
    attributes = list(sorted_thresholds.keys())
    if isinstance(delta_data, pd.DataFrame):
        delta_data = delta_data.to_dict(orient="records")

    for attribute in attributes:
        old_unique_values = set()
        for val1, val2 in original_distance_lookups[attribute].keys():
            old_unique_values.add(val1)
            old_unique_values.add(val2)
        old_unique_values = list(old_unique_values)

        delta_values = [
            row[attribute] for row in delta_data if pd.notnull(row[attribute])
        ]
        delta_values = set(delta_values)
        new_unique_values = list(delta_values - set(old_unique_values))

        if not new_unique_values:
            continue

        partial_lookup = incremental_calculate_unique_distances(
            attribute, old_unique_values, new_unique_values, num_processes
        )

        original_distance_lookups[attribute].update(partial_lookup)

    return original_distance_lookups


def calculate_precision_recall_f1(E_final_list, ground_truth_errors):
    E_final = set(E_final_list)
    TP = len(E_final.intersection(ground_truth_errors))
    print(f"TP: {TP}")
    FP = len(E_final.difference(ground_truth_errors))
    print(f"FP: {FP}")
    FN = len(ground_truth_errors.difference(E_final))
    print(f"FN: {FN}")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1_score


def check_all_pairs_covered(find_violating_pair, E_final):
    # 提取 E_final 中的所有元组索引
    E_final_tuples = set()
    for cell in E_final:
        # 假设 cell 的形式是 "t{index}.{attribute}"，我们提取索引部分
        index = int(cell.split(".")[0][1:])
        E_final_tuples.add(index)

    # 找出未被覆盖的元组对
    uncovered_pairs = []
    for pair in find_violating_pair:
        if pair[0] not in E_final_tuples and pair[1] not in E_final_tuples:
            uncovered_pairs.append(pair)

    # 返回结果
    if uncovered_pairs:
        return f"The following pairs are not covered: {uncovered_pairs}"
    else:
        return "All violating pairs are covered by E_final."


def run_single_dataset(
    dd_constraints,
    dirty_file,
    prob_file,
    diff_file,
    sorted_thresholds,
    attribute_thresholds,
    num_processes=16,
):
    # 1) 读取数据
    start_time = time.time()
    data_instance = pd.read_csv(dirty_file)
    data_instance = data_instance.to_dict(orient="records")
    length = len(data_instance)
    load_data_time = time.time() - start_time

    # 2) 提取 ground truth
    difference_data = pd.read_csv(diff_file)
    ground_truth_errors = set(
        f"t{row['Index']}.{row['Attribute']}" for _, row in difference_data.iterrows()
    )

    # 3) 构建距离搜索表
    start_time1 = time.time()
    distance_lookups = build_all_distance_lookups(
        data_instance, sorted_thresholds, num_processes
    )
    end_time1 = time.time()
    build_lookup_time = end_time1 - start_time1

    # 4) 生成聚类
    start_time2 = time.time()
    clusters = generate_clusters_parallel(
        data_instance, sorted_thresholds, distance_lookups, num_processes
    )
    end_time2 = time.time()
    cluster_time = end_time2 - start_time2

    # 5) 查找违反对
    start_time3 = time.time()
    (
        find_violating_pair,
        hyper_edges,
    ) = find_violating_pairs_parallel(
        dd_constraints, clusters, num_processes, length, delta_n=0
    )
    end_time3 = time.time()
    find_vio_pairs_time = end_time3 - start_time3

    # 6) EDet
    start_time4 = time.time()
    E_final = build_graph_parallel(
        hyper_edges, prob_file, num_processes, attribute_thresholds=attribute_thresholds
    )
    end_time4 = time.time()
    edet_time = end_time4 - start_time4

    # 7) 总时间
    total_time = build_lookup_time + cluster_time + find_vio_pairs_time + edet_time

    # 8) 计算指标
    precision, recall, f1_score = calculate_precision_recall_f1(
        E_final, ground_truth_errors
    )

    # 9) 检查覆盖情况
    result_message = check_all_pairs_covered(find_violating_pair, E_final)

    # 将结果以字典形式返回，便于上层汇总
    return {
        # "dataset_dirty": dirty_file,
        # "dataset_prob": prob_file,
        # "dataset_diff": diff_file,
        "data_length": length,
        "build_lookup_time": build_lookup_time,
        "cluster_time": cluster_time,
        "find_vio_pairs_time": find_vio_pairs_time,
        "edet_time": edet_time,
        "time_total": total_time,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "cover_message": result_message,
    }


if __name__ == "__main__":
    # 定义约束条件
    # dd_constraints = [
    #     (
    #         {"surname": ("<=", 0.0), "birthplace": ("<=", 0.0)},
    #         ("birthyear", ("<=", 0.0)),
    #     ),
    #     ({"name": ("<=", 0.0)}, ("birthyear", ("<=", 0.0))),
    #     (
    #         {"surname": ("<=", 0.0), "birthyear": ("<=", 0.0)},
    #         ("birthplace", ("<=", 0.0)),
    #     ),
    #     ({"name": ("<=", 0.0)}, ("birthplace", ("<=", 0.0))),
    #     (
    #         {"surname": ("<=", 0.0), "city": ("<=", 0.0), "manager": ("<=", 0.0)},
    #         ("team", ("<=", 0.0)),
    #     ),
    # ]
    dd_constraints = [
        # ({"flight": ("<=", 0.0)}, ("actArrTime", ("<=", 0.0))),
        # ({"flight": ("<=", 0.0)}, ("schedArrTime", ("<=", 0.0))),
        # ({"flight": ("<=", 0.0)}, ("actDepTime", ("<=", 0.0))),
        ({"flight": ("<=", 0.0)}, ("schedDepTime", ("<=", 0.0))),
        # ({"schedArrTime": ("<=", 0.0)}, ("actArrTime", ("<=", 0.0))),
        # ({"schedDepTime": ("<=", 0.0)}, ("actDepTime", ("<=", 0.0))),
    ]
    # 固定处理器数量
    FIXED_NUM_PROCESSES = 16

    # 1. 准备数据和其他辅助变量
    start_time = time.time()
    # 提取全部约束的阈值
    sorted_thresholds = extract_attributes_thresholds(dd_constraints)
    print("初始提取阈值所需时间:", time.time() - start_time)

    data_pairs = [
        # (20, 40),
        # (40, 60),
        # (60, 80),
        # (80, 100),
        # (100, 120),
        # (120, 140),
        # (140, 160),
        # (160, 180),
        (180, 200),
    ]

    attribute_thresholds = {
        "surname": 0.1,
        "birthplace": 0.1,
        "birthyear": 0.1,
        "name": 0.1,
        "city": 0.1,
        "manager": 0.1,
        "team": 0.1,
    }

    # 初始化结果汇总列表
    results_summary = []

    for base_size, inc_size in data_pairs:
        print(
            f"\n========== 处理原始数据: {base_size}k, 增量到: {inc_size}k =========="
        )

        # ========== 2) 构建并处理“原始数据集” ==========
        # 1) 文件路径
        base_dir = f"/zhaotingfeng/Bean/soccer/incremental/{base_size}k"
        base_dirty_file = os.path.join(base_dir, "dirty.csv")

        # 2) “原始+增量”概率文件放在 inc_size k 目录
        inc_dir = f"/zhaotingfeng/Bean/soccer/incremental/{inc_size}k"
        probabilities_file = os.path.join(
            inc_dir, "predicted_probabilities_10k_inct.csv"
        )
        diff_file = os.path.join(inc_dir, "difference.csv")
        inc_data_file = os.path.join(inc_dir, "inct_dirty_data.csv")  # 增量文件

        # 读取原始数据
        start_time_base = time.time()
        data_instance = pd.read_csv(base_dirty_file)
        data_instance = data_instance.to_dict(orient="records")
        length_base = len(data_instance)
        print(f"原始数据集({base_size}k)长度: {length_base}")

        # 构建搜索表(原始部分)
        st1 = time.time()
        distance_lookups = build_all_distance_lookups(
            data_instance, sorted_thresholds, FIXED_NUM_PROCESSES
        )
        et1 = time.time()
        build_lookup_time_base = et1 - st1

        # 生成聚类(原始部分)
        st2 = time.time()
        clusters_base = generate_clusters_parallel(
            data_instance, sorted_thresholds, distance_lookups, FIXED_NUM_PROCESSES
        )
        et2 = time.time()
        cluster_time_base = et2 - st2

        # 查找违反对(原始部分)
        st3 = time.time()
        find_violating_pair_base, hyper_edges_base = find_violating_pairs_parallel(
            dd_constraints, clusters_base, FIXED_NUM_PROCESSES, length_base, delta_n=0
        )
        et3 = time.time()
        find_vio_pairs_time_base = et3 - st3

        # 构建图(原始部分), 使用“原始+增量”概率文件
        st4 = time.time()
        E_final_base = build_graph_parallel(
            hyper_edges_base,
            probabilities_file,
            FIXED_NUM_PROCESSES,
            attribute_thresholds=attribute_thresholds,
        )
        et4 = time.time()
        edet_time_base = et4 - st4

        # 原始部分总时间
        total_time_base = (
            build_lookup_time_base
            + cluster_time_base
            + find_vio_pairs_time_base
            + edet_time_base
        )
        print(f"原始数据({base_size}k)处理完毕, 总时间: {total_time_base:.2f} 秒")

        # ========== 3) 处理“增量数据” ==========
        inc_data_df = pd.read_csv(inc_data_file)
        inc_data = inc_data_df.to_dict(orient="records")
        length_inc = len(inc_data)
        print(f"增量数据集({inc_size}k - {base_size}k): {length_inc}")

        # 读取Ground Truth
        difference_data = pd.read_csv(diff_file)
        ground_truth_errors = set(
            f"t{row['Index']}.{row['Attribute']}"
            for _, row in difference_data.iterrows()
        )
        print(f"ground_truth_errors数量：{len(ground_truth_errors)}")

        # 增量部分的计时
        st5 = time.time()
        # 1) 更新搜索表(增量)
        distance_lookups_updated = incremental_build_all_distance_lookups(
            distance_lookups, sorted_thresholds, inc_data, FIXED_NUM_PROCESSES
        )
        et5 = time.time()
        build_delta_lookup_time_inc = et5 - st5

        # 2) 生成增量聚类
        st6 = time.time()
        clusters_inc = generate_clusters_parallel_incremental(
            data_instance,  # 原始
            inc_data,  # 增量
            sorted_thresholds,
            distance_lookups_updated,
            FIXED_NUM_PROCESSES,
        )
        et6 = time.time()
        cluster_time_inc = et6 - st6

        # 3) 查找增量违反对
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

        # 4) 合并原始 E_final 与增量违反对 -> 构建增量图
        # 先合并 hyper_edges，再做 EDet
        # hyper_edges_total = hyper_edges_base.copy() + hyper_edges_inc
        # print(f"合并后的总违反对数量: {len(hyper_edges_total)}")

        st8 = time.time()
        # 构建 E_final_incremental, 注意使用同一个 probabilities_file(原始+增量)
        E_final_inc = build_graph_parallel(
            hyper_edges_inc,
            probabilities_file,
            FIXED_NUM_PROCESSES,
            attribute_thresholds=attribute_thresholds,
        )
        et8 = time.time()
        edet_time_inc = et8 - st8

        # 5) 合并 E_final_base 和 E_final_inc: 其实你表述的是把“原始部分 + 增量部分”的错误集都加起来
        E_final_total = E_final_base.union(E_final_inc)
        # 也可写: E_final_total = E_final_base | E_final_inc

        # 增量总时间
        total_time_inc = (
            build_delta_lookup_time_inc
            + cluster_time_inc
            + find_vio_pairs_time_inc
            + edet_time_inc
        )
        print(f"增量总时间({inc_size}k)为: {total_time_inc:.2f} 秒")

        # ========== 4) 计算最终Precision, Recall, F1 ==========
        # 注意：我们拿 E_final_total 和 ground_truth_errors 来计算指标
        precision, recall, f1_score_val = calculate_precision_recall_f1(
            E_final_total, ground_truth_errors
        )
        print(
            f"最终合并错误集与真实错误的三大指标: P={precision:.4f}, R={recall:.4f}, F1={f1_score_val:.4f}"
        )

        # 检查覆盖
        # coverage_msg = check_all_pairs_covered(find_violating_pair_inc, E_final_total)
        # print(f"覆盖检查: {coverage_msg}")

        # 记录结果
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
                # "CoverageCheck": coverage_msg
            }
        )

    # ================== 5) 最终保存结果 ==================
    df_results = pd.DataFrame(results_summary)
    print("\n====== 所有结果汇总 ======")
    print(df_results)

    excel_file = "incremental_final_results.xlsx"
    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
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

    print(f"\n最终增量结果已保存至 {excel_file}")
