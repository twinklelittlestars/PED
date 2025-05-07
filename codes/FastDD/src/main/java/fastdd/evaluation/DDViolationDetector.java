package fastdd.evaluation;

import de.metanome.algorithms.dcfinder.input.RelationalInput;
import de.metanome.algorithms.dcfinder.helpers.IndexProvider;
import fastdd.differentialfunction.DifferentialFunction;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

public class DDViolationDetector {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: java DDViolationDetector <dirty.csv> <dd.txt> <prob.csv> <diff.csv>");
            return;
        }

        String dirtyFilePath = args[0];
        String ddFilePath = args[1];
        String probFilePath = args[2];
        String diffFilePath = args[3];

        long startTime = System.nanoTime();

        File dirtyFile = new File(dirtyFilePath);
        RelationalInput input = new RelationalInput(dirtyFile);

        List<String> columnNames = Arrays.stream(input.columnNames())
                .map(String::trim)
                .collect(Collectors.toList());
        System.out.println("Column Name: " + columnNames);

        List<List<String>> table = new ArrayList<>();
        int originalRowCount = 0;
        int cleanedRowCount = 0;

        while (input.hasNext()) {
            List<String> row = input.next();
            if (row == null || row.isEmpty()) continue;

            List<String> trimmedRow = row.stream()
                .map(s -> s == null ? null : s.trim())
                .collect(Collectors.toList());

            boolean hasMissing = trimmedRow.stream().anyMatch(s -> s == null || s.isEmpty());
            originalRowCount++;

            if (!hasMissing) {
                table.add(trimmedRow);
                cleanedRowCount++;
            }
        }

        System.out.printf("Load row count: %d, Valid row count after cleaning missing values: %d\n", originalRowCount, cleanedRowCount);

        IndexProvider<DifferentialFunction> provider = new IndexProvider<>();
        Map<DifferentialFunction, Set<String>> dfPool = new HashMap<>();

        List<SimpleDifferentialDependency> ddList = MyDDParser.parseWithPool(
                new File(ddFilePath), columnNames, provider, dfPool);

        Map<String, DifferentialFunction> uniqueDFMap = new LinkedHashMap<>();
        for (DifferentialFunction df : dfPool.keySet()) {
            uniqueDFMap.put(df.toString(), df);
        }
        List<DifferentialFunction> allDFs = new ArrayList<>(uniqueDFMap.values());

        System.out.println("Total number of differential functions: " + allDFs.size());
        for (int i = 0; i < allDFs.size(); i++) {
            System.out.printf("  [DF-%d] %s\n", i, allDFs.get(i));
        }

        int numThreads = 16;
        int shardSize = 500;
        File outputCsv = new File("diffset_output.csv");

        DiffSetBuilder.buildDiffSetsShardToFile(table, allDFs, columnNames, numThreads, shardSize, outputCsv);
        System.out.println("Diff-Set writing to disk completed");
        Map<SimpleDifferentialDependency, List<String>> violationMap =
            DiffSetBuilder.checkViolationsFromFile(outputCsv, allDFs, ddList);

        System.out.println("\nViolating tuple pairs per constraint:");
        for (Map.Entry<SimpleDifferentialDependency, List<String>> entry : violationMap.entrySet()) {
            System.out.printf("  %s → %d violating tuple pairs\n", entry.getKey(), entry.getValue().size());
        }

        Set<String> errorCells = HyperEdgeErrorDetector.buildGraphParallel(
            violationMap,
            16,          
            null,        
            null         
        );

        System.out.println("Number of predicted error cells: " + errorCells.size());
        HyperEdgeErrorDetector.evaluate(
            diffFilePath,
            errorCells
        );

        long endTime = System.nanoTime();
        System.out.printf("Total time elapsed: %.2f seconds\n", (endTime - startTime) / 1e9);
    }
}