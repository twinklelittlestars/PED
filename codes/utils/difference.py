import pandas as pd


def generate_difference_file(dirty_file, clean_file, output_file):
    """
    Compare dirty_file and clean_file row by row, identify different cells,
    and save them into output_file. Returns the resulting dataframe.
    """
    data1 = pd.read_csv(dirty_file)
    data2 = pd.read_csv(clean_file)

    if len(data1) != len(data2):
        raise ValueError(
            "The number of rows in dirty_file and clean_file do not match."
        )

    differences = []
    num_dirty_value = 0

    for idx, (row1, row2) in enumerate(zip(data1.iterrows(), data2.iterrows())):
        row1, row2 = row1[1], row2[1]
        for col in data1.columns:
            dirty_value = str(row1[col]).strip() if not pd.isna(row1[col]) else ""
            clean_value = str(row2[col]).strip() if not pd.isna(row2[col]) else ""

            # If the clean_value is NaN or empty, increment a dirty counter
            if pd.isna(row2[col]):
                num_dirty_value += 1

            # If they differ, record the difference
            if dirty_value != clean_value:
                differences.append({"Index": idx, "Attribute": col})

    differences_df = pd.DataFrame(differences)
    differences_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Differences saved to: {output_file}")

    return differences_df
