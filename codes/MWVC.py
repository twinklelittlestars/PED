import csv
import multiprocessing
from collections import defaultdict
from itertools import combinations


def process_hyper_edges(hyper_edges_subset, probabilities, attribute_thresholds):
    local_error_set = set()

    for hyper_edge in hyper_edges_subset:
        add_error_set = False
        grouped_nodes = defaultdict(list)
        for node in hyper_edge:
            tuple_index, attribute = parse_node(node)
            grouped_nodes[attribute].append(node)

        for attribute, nodes in grouped_nodes.items():
            for u in nodes:
                tuple_index_u, attribute_u = parse_node(u)
                prob_u = probabilities[tuple_index_u][attribute_u]
                threshold_u = attribute_thresholds.get(attribute_u)
                if prob_u < threshold_u:
                    add_error_set = True
                    local_error_set.add(u)

            if not add_error_set:
                min_probability = float("inf")
                least_probable_cell = None

                for node in hyper_edge:
                    tuple_index, attribute = parse_node(node)
                    prob = probabilities[tuple_index][attribute]
                    if prob < min_probability:
                        min_probability = prob
                        least_probable_cell = node

                if least_probable_cell:
                    local_error_set.add(least_probable_cell)

    return local_error_set


def build_graph_parallel(
    hyper_edges, probabilities_file, num_processes, attribute_thresholds
):
    probabilities = []
    with open(probabilities_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            probabilities.append({headers[i]: float(row[i]) for i in range(len(row))})

    chunk_size = (len(hyper_edges) + num_processes - 1) // num_processes
    hyper_edges_chunks = [
        hyper_edges[i : i + chunk_size] for i in range(0, len(hyper_edges), chunk_size)
    ]

    print("processes number:", num_processes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_hyper_edges,
            [
                (chunk, probabilities, attribute_thresholds)
                for chunk in hyper_edges_chunks
            ],
        )

    global_error_set = set()

    for error_set in results:
        global_error_set.update(error_set)

    return global_error_set


def parse_node(node):
    parts = node.split(".")
    tuple_index = int(parts[0][1:])
    attribute = parts[1]
    return tuple_index, attribute
