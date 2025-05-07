package fastdd.evaluation;

import fastdd.differentialfunction.DifferentialFunction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class HyperEdgeErrorDetector {

    public static Set<String> buildGraphParallel(
            Map<SimpleDifferentialDependency, List<String>> violationMap,
            int numThreads,
            Map<String, Double> attributeThresholds,  
            String probabilityFilePath                
    ) throws InterruptedException, ExecutionException {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Set<String>>> futures = new ArrayList<>();

        for (Map.Entry<SimpleDifferentialDependency, List<String>> entry : violationMap.entrySet()) {
            SimpleDifferentialDependency dd = entry.getKey();
            List<String> violators = entry.getValue();
            futures.add(executor.submit(() -> detectErrorsGreedy(dd, violators)));
        }

        Set<String> finalErrors = new HashSet<>();
        for (Future<Set<String>> future : futures) {
            finalErrors.addAll(future.get());
        }
        executor.shutdown();
        return finalErrors;
    }

    private static Set<String> detectErrorsGreedy(SimpleDifferentialDependency dd, List<String> violators) {
        Set<String> errorCells = new HashSet<>();
        Map<Integer, Integer> tupleFrequency = new HashMap<>();
        Set<String> remaining = new HashSet<>(violators);

        while (!remaining.isEmpty()) {
            tupleFrequency.clear();
            for (String pair : remaining) {
                String[] parts = pair.split(",");
                int i = Integer.parseInt(parts[0]);
                int j = Integer.parseInt(parts[1]);
                tupleFrequency.put(i, tupleFrequency.getOrDefault(i, 0) + 1);
                tupleFrequency.put(j, tupleFrequency.getOrDefault(j, 0) + 1);
            }

            int maxTid = -1;
            int maxFreq = -1;
            for (Map.Entry<Integer, Integer> e : tupleFrequency.entrySet()) {
                if (e.getValue() > maxFreq) {
                    maxFreq = e.getValue();
                    maxTid = e.getKey();
                }
            }

            Iterator<String> it = remaining.iterator();
            while (it.hasNext()) {
                String[] parts = it.next().split(",");
                if (Integer.parseInt(parts[0]) == maxTid || Integer.parseInt(parts[1]) == maxTid) {
                    it.remove();
                }
            }

            for (DifferentialFunction df : dd.getLhs()) {
                String attr = df.getOperand().getColumn().toString();
                errorCells.add("t" + maxTid + "." + attr);
            }
            String rhsAttr = dd.getRhs().getOperand().getColumn().toString();
            errorCells.add("t" + maxTid + "." + rhsAttr);
        }

        return errorCells;
    }


    public static void evaluate(String differenceFilePath, Set<String> predictedErrors) throws Exception {
        Set<String> groundTruth = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(differenceFilePath))) {
            br.readLine(); 
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split(",");
                if (parts.length < 2) continue;
                groundTruth.add("t" + parts[0].trim() + "." + parts[1].trim());
            }
        }

        int TP = 0;
        for (String e : predictedErrors) if (groundTruth.contains(e)) TP++;
        int FP = predictedErrors.size() - TP;
        int FN = groundTruth.size() - TP;

        double precision = TP + FP > 0 ? (double) TP / (TP + FP) : 0.0;
        double recall = TP + FN > 0 ? (double) TP / (TP + FN) : 0.0;
        double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

        System.out.printf("\nEvaluation results:\nPrecision: %.4f\nRecall: %.4f\nF1 Score: %.4f\n", precision, recall, f1);
    }
}
