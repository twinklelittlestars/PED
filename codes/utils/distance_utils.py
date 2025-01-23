import multiprocessing
import re
import time
from collections import defaultdict

import pandas as pd

# Flight Dataset
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

# # Hospital Dataset
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
# INT_COLUMNS = {""}

# # Soccer Dataset
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
        value = value.strip()
    return value


def preprocess_values(values):
    return [clean_value(value) for value in values]


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


def calculate_difference(value1, value2, col_name):
    if col_name in STRING_COLUMNS:
        value1 = str(value1)
        value2 = str(value2)
        return levenshtein_distance(value1, value2)
    elif col_name in INT_COLUMNS:
        return abs(int(float(value1)) - int(float(value2)))
    else:
        raise ValueError(f"Column '{col_name}' has an unknown data type.")


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
    return distance_lookup


def calculate_unique_distances(attribute, unique_values, num_processes):
    unique_values = [v for v in unique_values if pd.notnull(v)]
    distance_lookup = {}
    num_unique = len(unique_values)

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


def build_all_distance_lookups(data_instance, sorted_thresholds, num_processes):
    distance_lookups = {}
    attributes = list(sorted_thresholds.keys())

    for attribute in attributes:
        values = preprocess_values([row[attribute] for row in data_instance])
        unique_values = list(set(values))
        distance_lookup = calculate_unique_distances(
            attribute, unique_values, num_processes
        )
        distance_lookups[attribute] = distance_lookup

    return distance_lookups


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
