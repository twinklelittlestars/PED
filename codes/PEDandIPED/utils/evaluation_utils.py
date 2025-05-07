def calculate_precision_recall_f1(E_final_list, ground_truth_errors):
    E_final = set(E_final_list)
    TP = len(E_final.intersection(ground_truth_errors))
    FP = len(E_final.difference(ground_truth_errors))
    FN = len(ground_truth_errors.difference(E_final))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1_score


def check_all_pairs_covered(find_violating_pair, E_final):
    E_final_tuples = set()
    for cell in E_final:
        index = int(cell.split(".")[0][1:])  # t{i} -> i
        E_final_tuples.add(index)

    uncovered_pairs = []
    for pair in find_violating_pair:
        if pair[0] not in E_final_tuples and pair[1] not in E_final_tuples:
            uncovered_pairs.append(pair)

    if uncovered_pairs:
        return f"The following pairs are not covered: {uncovered_pairs}"
    else:
        return "All violating pairs are covered by E_final."
