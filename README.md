# Minimum Change ≠ Best Cleaning: Parallel and Incremental Error Detection under Integrity Constraints

This repository provides the source code, data, and supplementary materials for **Minimum Change ≠ Best Cleaning: Parallel and Incremental Error Detection under Integrity Constraints**.

## Environment & Dependencies

### PED & IPED

- **Python version**: 3.9.12  
- Other required packages are listed in [`requirements.txt`](./requirements.txt). You can install them using:
  ```bash
  pip install -r requirements.txt
  ```

### FastDD

Setup instructions can be found in the [FastDD GitHub repository](https://github.com/TristonK/FastDD).

### densitysrepair

Configure the environment based on the [provided code](https://github.com/densitysrepair/densitysrepair).

### OCQA

OCQA is included in the environment configured for PED & IPED.

## File Explanations

- **[`codes`](./codes)**: Contains all source code implementations

  - **[`PED&IPED`](./codes/PED&IPED/)**: Contains our implementation of the PED method and the incremental IPED method.
    - **[`PED.py`](./codes/PED&IPED/PED.py)**: Main script for the method PED
    - **[`IPED.py`](./codes/PED&IPED/IPED.py)**: Main script for the method IPED
    - **[`tune_thresholds.py`](./codes/PED&IPED/tune_thresholds.py)**: Script for selecting probability thresholds using the validation set
    - Other supporting modules and utility functions are also included

  - **[`FastDD`](./codes/FastDD/)**: Contains the modified implementation of the FastDD method.
    - **[`run_all.sh`](./codes/FastDD/src/main/java/fastdd/evaluation/run_all.sh)**: Execution script for evaluating FastDD in the error detection setting.
    - The improved logic for FastDD is placed under [`evaluation`](./codes/FastDD/src/main/java/fastdd/evaluation/).

  - **[`densitysrepair`](./codes/densitysrepair/)**: Contains adapted versions of HEURISTIC and RELAXATION methods.
    - **[`run_all.sh`](./codes/densitysrepair/src/test/run_all.sh)**: Script for evaluating the methods at the cell-level rather than the tuple-level.

  - **[`OCQA`](./codes/OCQA/)**: Implementation of the OCQA method.
    - **[`opcqa.py`](./codes/OCQA/opcqa.py)**: Main script for running the OCQA method.
  

- **[`data`](./data)**: Includes the datasets used in our experiments

  - Each dataset folder (e.g., [`Flight`](./data/Flight/), [`Hospital`](./data/Hospital/), [`Soccer`](./data/Soccer/), [`MIMIC`](./data/MIMIC/), [`Plane`](./data/Plane/)) contains:
    - **dirty** (`dirty.csv`) and **clean** (`clean.csv`) datasets
    (For **MIMIC** and **Plane**, no official clean version is provided; the `clean.csv` file is generated based on manual correction.)
    - **`dd_constraints_fd.txt`**: FD constraints used for PED, IPED, FastDD, OCQA
    - **`dd_constraints_dd.txt`**: DD constraints used for PED, IPED, FastDD, OCQA
    - **`dd_constraints_dc.txt`**: Diverse DC constraints used by methods like BigDansing, Holistic, and HoloClean
    - **`attribute_thresholds.txt`**: Probability thresholds for each attribute
    - **`difference.csv`**: Ground-truth error cells in the dirty dataset
    - **`predicted_probabilities.csv`**: Cell-level error probabilities in the dirty dataset estimated based on Bayesian analysis
  - The **Soccer** subdirectory includes multiple data files of different sizes and error rates.

  - Since **[`densitysrepair`](./codes/densitysrepair/)** uses a different data format and constraint type, its required datasets are stored in [`densitysrepair/data`](./codes/densitysrepair/data):
    - **`cfd-<dataset>.final`**: CFD constraints used for this method
    - **`<dataset>.data`**: Clean dataset
    - **`<dataset>-dirty.final`**: Dirty dataset

- **[`Incremental_data`](./Incremental_data)**: Stores incremental versions of the **Soccer** dataset for evaluating the performance of IPED across various base sizes

- **[`appendix.pdf`](./appendix.pdf)**:

  - An overview figure illustrating the algorithm IPED
  - Proofs of complexity for both PED and IPED

- **[`requirements.txt`](./requirements.txt)**  
  Lists all needed Python packages

## Usage

### Running PED

For example, run the algorithm PED with the Flight dataset:

```bash
python PED.py
```

For large-scale datasets such as MIMIC, you can run PED in a distributed manner. Specifically, you can execute the following scripts on five separate server nodes:

```bash
python PED_1-10.py
python PED_10-20.py
python PED_20-30.py
python PED_30-40.py
python PED_40-50.py
```

After all processes finish, merge the results by running:

```bash
python merge.py
```

The final result is equivalent to running the pipeline in a single-machine (non-distributed) mode, but this approach significantly reduces the overall runtime.

### Running IPED

To run the algorithm IPED, use:

```bash
python IPED.py
```

You may need edit paths and parameters in the script if necessary. If you plan to switch to a different dataset, please remember to update the [`distance_utils.py`](./codes/utils/distance_utils.py) file, where the `STRING_COLUMNS` and `INT_COLUMNS` variables define which columns are treated as strings or integers.

### Running FastDD

```bash
mvn clean package
chmod +x src/main/java/fastdd/evaluation/run_all.sh
./src/main/java/fastdd/evaluation/run_all.sh
```

### Running HEURISTIC and RELAXATION

```bash
mvn clean package
chmod +x src/test/run_all.sh
./src/test/run_all.sh
```

### Running OCQA

```bash
python opcqa.py
```


## Data Sources

- **Flight**: [https://github.com/BigDaMa/raha/tree/master/datasets/flights](https://github.com/BigDaMa/raha/tree/master/datasets/flights)
- **Hospital**: [https://github.com/BigDaMa/raha/tree/master/datasets/hospital](https://github.com/BigDaMa/raha/tree/master/datasets/hospital)
- **Soccer**: [https://codeocean.com/capsule/8720426/tree/v1](https://codeocean.com/capsule/8720426/tree/v1)
- **MIMIC**: [https://mimic.mit.edu/](https://mimic.mit.edu/)
- **Plane**:[https://gitlab.com/antoonbronselaer/swipe-reproducibility/-/blob/master/datasets/flight_numbers_dataset.csv](https://gitlab.com/antoonbronselaer/swipe-reproducibility/-/blob/master/datasets/flight_numbers_dataset.csv?ref_type=heads)
