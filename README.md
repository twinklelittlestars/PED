# Minimum Change ≠ Best Cleaning: Parallel and Incremental Error Detection under Integrity Constraints

This repository provides the source code, data, and supplementary materials for **Minimum Change ≠ Best Cleaning: Parallel and Incremental Error Detection under Integrity Constraints**.

---

## Environment & Dependencies

- **Python version**: 3.9.12
- Other required packages are listed in [`requirements.txt`](./requirements.txt). You can install them using:
  ```bash
  pip install -r requirements.txt
  ```

---

## File Explanations

- **[`codes`](./codes)**: Contains all source code implementations

  - **['PED.py'](./codes/PED.py)**: Main script for the algorithm PED
  - **['IPED.py'](./codes/IPED.py)**: Main script for the algorithm IPED
  - Other supporting modules and utility functions are also included

- **['data'](./data)**: Includes the datasets used in our experiments

  - Each dataset folder (e.g., ['Flight'](./data/Flight/), ['Hospital'](./data/Hospital/), ['Soccer'](./data/Soccer/)) contains:
    - **Dirty** (`dirty.csv`) and **clean** (`clean.csv`) datasets
    - **`dd_constraints.txt`**: DD constraints used for the dataset
    - **`attribute_thresholds.txt`**: Probability thresholds for each attribute
  - The **Soccer** subdirectory includes multiple data files of different sizes and error rates

- **[Incremental_data](./Incremental_data)**: Stores incremental versions of the **Soccer** dataset for evaluating the performance of IPED across various base sizes

- **[appendix.pdf](./appendix.pdf)**:

  - An overview figure illustrating the algorithm IPED
  - Proofs of complexity for both PED and IPED

- **[requirements.txt](./requirements.txt)**  
  Lists all needed Python packages

---

## Usage

### Running IPED

To run the algorithm PED, use:

```bash
python IPED.py
```

### Running PED

For example, run the algorithm IPED with the Flight dataset:

```bash
python PED.py
```

You may need edit paths and parameters in the script if necessary.

---

## Data Sources

- **Flight**: [https://github.com/BigDaMa/raha/tree/master/datasets/flights](https://github.com/BigDaMa/raha/tree/master/datasets/flights)
- **Hospital**: [https://github.com/BigDaMa/raha/tree/master/datasets/hospital](https://github.com/BigDaMa/raha/tree/master/datasets/hospital)
- **Soccer**: [https://codeocean.com/capsule/8720426/tree/v1](https://codeocean.com/capsule/8720426/tree/v1)
