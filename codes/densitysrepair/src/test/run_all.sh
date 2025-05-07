#!/bin/bash

GUROBI_JAR="path/densitysrepair/gurobi1201/linux64/lib/gurobi.jar"
BIN_DIR="bin"

datasets=(
  flight
  # hospital
  # soccer10
  # soccer20
  # soccer30
  # soccer40
  # soccer10k
  # soccer30k
  # soccer50k
  # soccer100k
  # soccer150k
  # soccer200k
  # mimic
  # plane
)

echo "=== Compiling Java ==="
find src -name "*.java" | xargs javac -cp "$GUROBI_JAR" -d "$BIN_DIR"
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "" > results.log

for name in "${datasets[@]}"; do
  echo "Running dataset: $name"
  echo "=== Dataset: $name ===" >> results.log
  java -Xmx12g -cp "$GUROBI_JAR:$BIN_DIR" test.CompareTest "$name" >> results.log
  echo -e "------------------------------------\n" >> results.log
done
