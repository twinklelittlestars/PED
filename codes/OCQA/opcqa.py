import pandas as pd
import itertools
import multiprocessing as mp
import random
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Union
import time

random.seed(42)
########################
# 1. Parallel Conflict Pair Construction Module
########################
def _check_fd_group_conflicts_count(args):
    group_list, rhs, df = args
    conflict_pairs = set()
    for group in group_list:
        indices = group.index.tolist()
        for i, j in itertools.combinations(indices, 2):
            if df.at[i, rhs] != df.at[j, rhs]:
                conflict_pairs.add((i, j))
    return conflict_pairs

def build_conflict_pairs_parallel(df: pd.DataFrame, fds: List[Tuple[Union[str, Tuple[str, ...]], str]], num_processes: int = 16) -> Set[Tuple[int, int]]:
    parsed_fds = []
    for lhs, rhs in fds:
        lhs = [lhs] if isinstance(lhs, str) else list(lhs)
        parsed_fds.append((lhs, rhs))

    total_conflict_set = set()
    fd_violated_tuples = dict()
    
    for lhs, rhs in parsed_fds:
        grouped = df.groupby(lhs)
        groups = [g for _, g in grouped]
        chunk_size = len(groups) // num_processes + 1
        chunks = [groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)]
        args = [(chunk, rhs, df) for chunk in chunks]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(_check_fd_group_conflicts_count, args)

        fd_conflicts = set()
        fd_tuples = set()
        for r in results:
            fd_conflicts.update(r)
            for i, j in r:
                fd_tuples.add(i)
                fd_tuples.add(j)

        print(f"[FD: {lhs} -> {rhs}] Number of conflict tuple pairs: {len(fd_conflicts)}")
        total_conflict_set.update(fd_conflicts)
        fd_violated_tuples[(tuple(lhs), rhs)] = fd_tuples

    return total_conflict_set, fd_violated_tuples
########################
# 2. Markov Repair Sampler
########################
def sample_repair_path(conflict_pairs: Set[Tuple[int, int]], max_steps: int = 100) -> Set[int]:
    conflict_map: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for (i, j) in conflict_pairs:
        conflict_map[i].add((i, j))
        conflict_map[j].add((i, j))

    active_conflicts = set(conflict_pairs)
    deleted = set()

    for _ in range(max_steps):
        if not active_conflicts:
            break

        conflict = random.choice(list(active_conflicts))
        delete_idx = random.choice(conflict)
        deleted.add(delete_idx)

        to_remove = conflict_map[delete_idx]
        for c in to_remove:
            if c in active_conflicts:
                active_conflicts.remove(c)
        conflict_map[delete_idx].clear()

    return deleted
########################
# 3. Hit Rate Estimation
########################
def estimate_tuple_hit_probability(conflict_pairs: Set[Tuple[int, int]], 
                                   num_tuples: int, 
                                   num_samples: int = 1000) -> Dict[int, float]:
    hit_counts = defaultdict(int)
    for _ in range(num_samples):
        deleted = sample_repair_path(set(conflict_pairs))
        for idx in deleted:
            hit_counts[idx] += 1
    return {i: hit_counts[i] / num_samples for i in range(num_tuples)}
########################
# 4. PRF Evaluation Module
########################
def evaluate_detection(predicted: Set[int], ground_truth: Set[int]) -> Dict[str, float]:
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
    
def evaluate_detection_cell_level(predicted_tuple_ids: Set[int],
                                  ground_truth_df: pd.DataFrame,
                                  clean_df: pd.DataFrame,
                                  fd_violated_tuples: Dict[Tuple[Tuple[str, ...], str], Set[int]]) -> Dict[str, float]:
    # === 1. Construct ground-truth cell set ===
    ground_truth_cells = set()
    for row_idx in range(len(ground_truth_df)):
        for col in ground_truth_df.columns:
            dirty_val = ground_truth_df.at[row_idx, col]
            clean_val = clean_df.at[row_idx, col]
            if pd.notna(dirty_val) and pd.notna(clean_val) and dirty_val != clean_val:
                ground_truth_cells.add((row_idx, col))

    # === 2. Construct predicted cell set ===
    predicted_cells = set()
    for row_idx in predicted_tuple_ids:
        for (lhs, rhs), rows in fd_violated_tuples.items():
            if row_idx in rows:
                for attr in list(lhs) + [rhs]:
                    predicted_cells.add((row_idx, attr))

    # === 3. Compute PRF ===
    tp = len(predicted_cells & ground_truth_cells)
    fp = len(predicted_cells - ground_truth_cells)
    fn = len(ground_truth_cells - predicted_cells)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"[Cell-level PRF] TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"[Cell-level PRF] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def analyze_conflict_pair_coverage(conflict_pairs: Set[Tuple[int, int]], 
                                   ground_truth_errors: Set[int]) -> Dict[str, Union[int, float]]:
    all_conflict_indices = set()
    for i, j in conflict_pairs:
        all_conflict_indices.add(i)
        all_conflict_indices.add(j)

    total_conflict_tuples = len(all_conflict_indices)
    covered_tuples = len(all_conflict_indices & ground_truth_errors)
    uncovered_tuples = len(all_conflict_indices - ground_truth_errors)
    coverage_ratio = covered_tuples / total_conflict_tuples if total_conflict_tuples > 0 else 0.0

    print(f"[Conflict Pair Analysis] Total tuples involved: {total_conflict_tuples}")
    print(f"[Conflict Pair Analysis] Tuples hitting real errors: {covered_tuples}")
    print(f"[Conflict Pair Analysis] Tuples not covered: {uncovered_tuples}")
    print(f"[Conflict Pair Analysis] Tuple coverage ratio: {coverage_ratio:.4f}")

    return {
        "total_conflict_tuples": total_conflict_tuples,
        "covered_tuples": covered_tuples,
        "uncovered_tuples": uncovered_tuples,
        "coverage_ratio": round(coverage_ratio, 4)
    }

########################
# Example Usage (Main Program)
########################
if __name__ == "__main__":
    start = time.time()
    dirty_df = pd.read_csv("path/PED/data/Flight/dirty.csv")
    clean_df = pd.read_csv("path/PED/data/Flight/clean.csv")
    # Flight
    fds = [
        (("flight",), "actArrTime"),
        (("flight",), "schedArrTime"),
        (("flight",), "actDepTime"),
        (("flight",), "schedDepTime"),
        (("schedArrTime",), "actArrTime"),
        (("schedDepTime",), "actDepTime")
    ]
    
    # # Hospital
    # fds = [
    #     (("HospitalName",), "ZipCode"),
    #     (("HospitalName",), "PhoneNumber"),
    #     (("MeasureCode",), "MeasureName"),
    #     (("MeasureCode",), "Stateavg"),
    #     (("ProviderNumber",), "HospitalName"),
    #     (("MeasureCode",), "Condition"),
    #     (("HospitalName",), "Address1"),
    #     (("HospitalName",), "HospitalOwner"),
    #     (("HospitalName",), "ProviderNumber"),
    #     (("City",), "CountyName"),
    #     (("ZipCode",), "EmergencyService"),
    #     (("HospitalName",), "City"),
    #     (("MeasureName",), "MeasureCode"),
    #     (("HospitalName", "PhoneNumber", "HospitalOwner"), "State")
    # ]
    
    # # Soccer
    # fds = [
    #     (("surname", "birthplace"), "birthyear"),
    #     (("surname", "birthyear"), "birthplace"),
    #     (("name",), "birthyear"),
    #     (("name",), "birthplace"),
    #     (("surname", "city", "manager"), "team")
    # ]
    
    # # MIMIC
    # fds = [
    #     (("subject_id", "hadm_id", "stay_id", "caregiver_id"), "itemid"),
    # ]
    
    # # Plain
    # fds = [
    #     (("composed_key",), "actArrTime"),
    #     (("composed_key",), "schedArrTime"),
    #     (("composed_key",), "actDepTime"),
    #     (("composed_key",), "schedDepTime")
    # ]
        

    print("[1] Constructing conflict pairs...")
    conflict_pairs, fd_violated_tuples = build_conflict_pairs_parallel(dirty_df, fds, num_processes=16)
    print(f"Detected {len(conflict_pairs)} conflict tuple pairs")

    print("[2] Sampling repair paths...")
    prob_map = estimate_tuple_hit_probability(conflict_pairs, len(dirty_df), num_samples=1000)

    print("[3] Threshold-based filtering to predict error tuples...")
    threshold = 0.005
    predicted_errors = {idx for idx, prob in prob_map.items() if prob > threshold}

    print("[4] Evaluating detection performance (PRF)...")
    # metrics = evaluate_detection(predicted_errors, ground_truth_errors)
    metrics = evaluate_detection_cell_level(predicted_errors, dirty_df, clean_df, fd_violated_tuples)
    # print("Precision:", metrics["precision"], "Recall:", metrics["recall"], "F1:", metrics["f1"])
    print(f"Total execution time: {time.time() - start:.2f} seconds")

    ground_truth_errors = set(dirty_df[~dirty_df.eq(clean_df)].dropna(how="all").index)
    print("Analyzing whether tuples in conflict pairs cover the true error set")
    coverage_stats = analyze_conflict_pair_coverage(conflict_pairs, ground_truth_errors)

    # print("[5] Top 10 tuples by hit rate:")
    # top_hit = sorted(prob_map.items(), key=lambda x: -x[1])[:10]
    # for idx, prob in top_hit:
    #     print(f"Row {idx}: hit rate = {prob:.3f}")