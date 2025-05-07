#!/bin/bash

MVN_CMD="mvn exec:java -Dexec.mainClass=fastdd.evaluation.DDViolationDetector -Dexec.cleanupDaemonThreads=false"

declare -a datasets=(
    "Flight"
    # "Hospital"   
    # "Soccer_10k_5%"
    # "Soccer_10k_10%"
    # "Soccer_10k_20%"
    # "Soccer_10k_30%"
    # "Soccer_10k_40%"
    # "Soccer_30k_5%"
    # "Soccer_50k_5%"
    # "Soccer_100k_5%"
    # "Soccer_150k_5%"
    # "Soccer_200k_5%"
    # "MIMIC"
    # "Plain"
)

for ds in "${datasets[@]}"; do
    echo "Running on dataset: $ds"
    DIR="path/PED/data/${ds}"

    $MVN_CMD -Dexec.args="\
        $DIR/dirty.csv \
        path/PED/data/Flight/dd_constraints_fd.txt \
        $DIR/predicted_probabilities.csv \
        $DIR/difference.csv"

    echo "Finished: $ds"
    echo "----------------------------------------"
done
