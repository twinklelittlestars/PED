import pandas as pd
# ------------------------------------------------------------------
# 1) Read multiple predicted error files, merge and deduplicate
# ------------------------------------------------------------------
print("Reading block error files...")

block_error_files = [
    "path/PED/data/MIMIC/block_errors_row_0_9.txt",
    "path/PED/data/MIMIC/block_errors_row_10_19.txt",
    "path/PED/data/MIMIC/block_errors_row_20_29.txt",
    "path/PED/data/MIMIC/block_errors_row_30_39.txt",
    "path/PED/data/MIMIC/block_errors_row_40_49.txt",
]
difference_file = r"path/PED/data/MIMIC/difference.csv"
diff_df = pd.read_csv(difference_file)
ground_truth_errors = set(
    f"t{row['Index']}.{row['Attribute']}" for _, row in diff_df.iterrows()
)
print(f"Number of ground_truth_errors: {len(ground_truth_errors)}")

predicted_errors = set()
for path in block_error_files:
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        predicted_errors.update(lines)

print(f"Total predicted errors after merge and dedup: {len(predicted_errors)}")

# ------------------------------------------------------------------
# 2) Compute metrics using predicted_errors and ground_truth_errors
# ------------------------------------------------------------------
precision = len(predicted_errors & ground_truth_errors) / len(predicted_errors) if predicted_errors else 0.0
recall = len(predicted_errors & ground_truth_errors) / len(ground_truth_errors) if ground_truth_errors else 0.0
f1_score_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n===== Evaluation of Merged Block Errors =====")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score_val:.4f}")
