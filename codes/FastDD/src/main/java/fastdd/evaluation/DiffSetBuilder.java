package fastdd.evaluation;

import fastdd.differentialfunction.DifferentialFunction;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DiffSetBuilder {

    public static class DFIndexMapper {
        private final Map<String, Integer> dfToIndex = new HashMap<>();
        private final List<String> indexToDf = new ArrayList<>();

        public DFIndexMapper(List<DifferentialFunction> allDFs) {
            for (int i = 0; i < allDFs.size(); i++) {
                String dfStr = allDFs.get(i).toString();
                dfToIndex.put(dfStr, i);
                indexToDf.add(dfStr);
            }
        }

        public String getDF(int index) {
            return indexToDf.get(index);
        }

        public int size() {
            return indexToDf.size();
        }
    }

    public static void buildDiffSetsShardToFile(
            List<List<String>> table,
            List<DifferentialFunction> allDFs,
            List<String> columnNames,
            int numThreads,
            int shardSize,
            File outputFile
    ) throws InterruptedException, IOException {

        DFIndexMapper mapper = new DFIndexMapper(allDFs);
        int numRows = table.size();

        List<List<Integer>> shards = new ArrayList<>();
        for (int i = 0; i < numRows; i += shardSize) {
            shards.add(IntStream.range(i, Math.min(i + shardSize, numRows))
                    .boxed().collect(Collectors.toList()));
        }
        System.out.printf("Number of shards: %d\n", shards.size());

        Map<Integer, Set<String>> dfToViolatedValuePairs = new HashMap<>();
        for (int dfIdx = 0; dfIdx < allDFs.size(); dfIdx++) {
            DifferentialFunction df = allDFs.get(dfIdx);
            String attr = df.getOperand().getColumn().getColumnName();
            int colIndex = columnNames.indexOf(attr);

            Map<String, List<Integer>> pli = new HashMap<>();
            for (int i = 0; i < numRows; i++) {
                String val = table.get(i).get(colIndex).trim();
                pli.computeIfAbsent(val, k -> new ArrayList<>()).add(i);
            }

            Set<String> violatedPairs = new HashSet<>();
            List<String> keys = new ArrayList<>(pli.keySet());
            for (int i = 0; i < keys.size(); i++) {
                for (int j = i; j < keys.size(); j++) {
                    String v1 = keys.get(i), v2 = keys.get(j);
                    int dist = computeEditDistance(v1, v2);
                    if (violates(df, dist)) violatedPairs.add(encode(v1, v2));
                }
            }
            dfToViolatedValuePairs.put(dfIdx, violatedPairs);
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<int[]> allShardPairs = new ArrayList<>();
        for (int a = 0; a < shards.size(); a++) {
            for (int b = a; b < shards.size(); b++) {
                allShardPairs.add(new int[]{a, b});
            }
        }
        Collections.shuffle(allShardPairs);

        PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(outputFile)));
        List<Future<?>> monitors = new ArrayList<>();

        for (int[] pair : allShardPairs) {
            int shardIdxA = pair[0], shardIdxB = pair[1];
            final List<Integer> shardA = shards.get(shardIdxA);
            final List<Integer> shardB = shards.get(shardIdxB);

            Future<Void> future = executor.submit(() -> {
                long start = System.currentTimeMillis();
                StringBuilder localBuffer = new StringBuilder();
                for (int i : shardA) {
                    for (int j : shardB) {
                        if (i >= j) continue;
                        long bitmask = 0L;
                        for (int dfIdx = 0; dfIdx < allDFs.size(); dfIdx++) {
                            String attr = allDFs.get(dfIdx).getOperand().getColumn().getColumnName();
                            int colIndex = columnNames.indexOf(attr);
                            String vi = table.get(i).get(colIndex).trim();
                            String vj = table.get(j).get(colIndex).trim();
                            if (dfToViolatedValuePairs.get(dfIdx).contains(encode(vi, vj))) {
                                bitmask |= (1L << dfIdx);
                            }
                        }
                        if (bitmask != 0L) {
                            localBuffer.append(i).append(",").append(j).append(",").append(bitmask).append("\n");
                        }
                    }
                }
                synchronized (writer) {
                    writer.write(localBuffer.toString());
                }
                System.out.printf("Completed shard(%d,%d): %ds\n", shardIdxA, shardIdxB,
                        (System.currentTimeMillis() - start) / 1000);
                return null;
            });

            monitors.add(CompletableFuture.runAsync(() -> {
                try {
                    future.get(2, TimeUnit.MINUTES);
                } catch (TimeoutException te) {
                    future.cancel(true);
                    System.err.printf("Timeout shard(%d, %d): canceled\n", shardIdxA, shardIdxB);
                } catch (Exception e) {
                    System.err.printf("Exception in shard(%d, %d): %s\n", shardIdxA, shardIdxB, e.getMessage());
                }
            }));
        }

        for (Future<?> f : monitors) {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("Monitor thread exception: " + e.getMessage());
            }
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.MINUTES);
        writer.close();

        System.out.println("All results have been written to file: " + outputFile.getAbsolutePath());
    }

    private static String encode(String a, String b) {
        return a.compareTo(b) < 0 ? a + "||" + b : b + "||" + a;
    }

    public static boolean violates(DifferentialFunction df, int dist) {
        double threshold = df.getDistance();
        return df.getOperator().getShortString().equals("<=") ? dist > threshold : dist <= threshold;
    }

    public static int computeEditDistance(String a, String b) {
        int[][] dp = new int[a.length() + 1][b.length() + 1];
        for (int i = 0; i <= a.length(); i++) dp[i][0] = i;
        for (int j = 0; j <= b.length(); j++) dp[0][j] = j;
        for (int i = 1; i <= a.length(); i++) {
            for (int j = 1; j <= b.length(); j++) {
                dp[i][j] = (a.charAt(i - 1) == b.charAt(j - 1))
                        ? dp[i - 1][j - 1]
                        : 1 + Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1]));
            }
        }
        return dp[a.length()][b.length()];
    }

    public static Map<Set<String>, List<String>> readDiffSetFromFile(File file, List<DifferentialFunction> allDFs) {
        Map<Set<String>, List<String>> diffSetToPairs = new HashMap<>();
        DFIndexMapper mapper = new DFIndexMapper(allDFs);

        try (Scanner scanner = new Scanner(file)) {
            while (scanner.hasNextLine()) {
                String[] parts = scanner.nextLine().split(",");
                if (parts.length != 3) continue;
                int i = Integer.parseInt(parts[0]);
                int j = Integer.parseInt(parts[1]);
                long bitmask = Long.parseLong(parts[2]);

                Set<String> dfSet = new LinkedHashSet<>();
                for (int k = 0; k < mapper.size(); k++) {
                    if (((bitmask >>> k) & 1L) != 0) {
                        dfSet.add(mapper.getDF(k));
                    }
                }

                diffSetToPairs.computeIfAbsent(dfSet, k -> new ArrayList<>()).add(i + "," + j);
            }
        } catch (Exception e) {
            System.err.println("Error reading diffset file: " + e.getMessage());
            e.printStackTrace();
        }

        return diffSetToPairs;
    }

    public static Map<SimpleDifferentialDependency, List<String>> checkViolationsFromFile(
            File file,
            List<DifferentialFunction> allDFs,
            List<SimpleDifferentialDependency> ddList
    ) {
        long startTime = System.currentTimeMillis();
        Map<SimpleDifferentialDependency, List<String>> result = new LinkedHashMap<>();
        for (SimpleDifferentialDependency dd : ddList) {
            result.put(dd, new ArrayList<>());
        }

        DiffSetBuilder.DFIndexMapper mapper = new DiffSetBuilder.DFIndexMapper(allDFs);

        try (Scanner scanner = new Scanner(file)) {
            while (scanner.hasNextLine()) {
                String[] parts = scanner.nextLine().split(",");
                if (parts.length != 3) continue;
                int i = Integer.parseInt(parts[0]);
                int j = Integer.parseInt(parts[1]);
                long bitmask = Long.parseLong(parts[2]);

                Set<String> dfSet = new HashSet<>();
                for (int k = 0; k < mapper.size(); k++) {
                    if (((bitmask >>> k) & 1L) != 0) {
                        dfSet.add(mapper.getDF(k));
                    }
                }

                for (SimpleDifferentialDependency dd : ddList) {
                    String rhsStr = dd.getRhs().toString();
                    if (!dfSet.contains(rhsStr)) continue;

                    boolean covered = false;
                    for (DifferentialFunction lhsDF : dd.getLhs()) {
                        if (dfSet.contains(lhsDF.toString())) {
                            covered = true;
                            break;
                        }
                    }

                    if (!covered) {
                        result.get(dd).add(i + "," + j);
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("Error reading and validating diffset file: " + e.getMessage());
            e.printStackTrace();
        }
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        System.out.printf("Detected violating value pairs for all differential functions, took %d ms\n", duration);

        return result;
    }


}
