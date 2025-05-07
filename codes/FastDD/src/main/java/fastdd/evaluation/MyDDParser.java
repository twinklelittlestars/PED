package fastdd.evaluation;

import fastdd.differentialfunction.DifferentialFunction;
import fastdd.evaluation.MyOperandHelper;
import de.metanome.algorithms.dcfinder.helpers.IndexProvider;
import de.metanome.algorithms.dcfinder.predicates.Operator;
import de.metanome.algorithms.dcfinder.predicates.operands.ColumnOperand;

import java.io.*;
import java.util.*;
import java.util.regex.*;

public class MyDDParser {

    public static List<SimpleDifferentialDependency> parseWithPool(
            File file,
            List<String> columnNames,
            IndexProvider<DifferentialFunction> provider,
            Map<DifferentialFunction, Set<String>> dfPool) throws IOException {

        List<SimpleDifferentialDependency> ddList = new ArrayList<>();

        BufferedReader reader = new BufferedReader(new FileReader(file));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            sb.append(line.trim());
        }
        reader.close();

        String content = sb.toString().replaceAll("\\s+", "");
        if (content.startsWith("[")) content = content.substring(1);
        if (content.endsWith("]")) content = content.substring(0, content.length() - 1);

        Pattern ddPattern = Pattern.compile("\\((\\{.*?\\}),\\((.*?)\\)\\)");
        Matcher matcher = ddPattern.matcher(content);

        int ddIndex = 1;
        while (matcher.find()) {
            String lhsStr = matcher.group(1);
            String rhsStr = matcher.group(2);

            Set<DifferentialFunction> lhs = new LinkedHashSet<>(parsePredicateMap(lhsStr, columnNames, provider, dfPool));
            DifferentialFunction rhs = parseSinglePredicate(rhsStr, columnNames, provider, dfPool);


            if (rhs == null) continue;

            SimpleDifferentialDependency dd = new SimpleDifferentialDependency(lhs, rhs);
            ddList.add(dd);

            System.out.println("\n Parsing constraint #" + (ddIndex++) + ": " + dd);
            System.out.println("  LHS: " + lhs);
            System.out.println("  RHS: " + rhs);
        }

        int dfIndex = 1;
        for (DifferentialFunction df : dfPool.keySet()) {
            System.out.println("  [DF-" + (dfIndex++) + "]: " + df);
        }

        return ddList;
    }

    private static List<DifferentialFunction> parsePredicateMap(String lhsStr,
                                                                 List<String> columnNames,
                                                                 IndexProvider<DifferentialFunction> provider,
                                                                 Map<DifferentialFunction, Set<String>> dfPool) {
        List<DifferentialFunction> result = new ArrayList<>();
        Pattern p = Pattern.compile("\"?(\\w+)\"?\\s*:\\s*\\(\\s*\"(<=|>)\"\\s*,\\s*([\\d\\.eE-]+)\\s*\\)");
        Matcher matcher = p.matcher(lhsStr);
        while (matcher.find()) {
            String col = matcher.group(1);
            String op = matcher.group(2);
            double threshold = Double.parseDouble(matcher.group(3));
            DifferentialFunction df = buildDF(col, op, threshold, columnNames, provider);
            if (df != null) {
                df = registerDF(df, dfPool);
                result.add(df);
            }
        }
        return result;
    }

    private static DifferentialFunction parseSinglePredicate(String rhsStr,
                                                             List<String> columnNames,
                                                             IndexProvider<DifferentialFunction> provider,
                                                             Map<DifferentialFunction, Set<String>> dfPool) {
        try {
            int commaIndex = rhsStr.indexOf(',');
            String col = rhsStr.substring(0, commaIndex).replaceAll("[\"\\s]", "");
            String rest = rhsStr.substring(commaIndex + 1).trim();

            Pattern p = Pattern.compile("\\(?\\s*\"?(<=|>)\"?\\s*,\\s*([\\d\\.eE-]+)\\s*\\)?");
            Matcher matcher = p.matcher(rest);
            if (matcher.find()) {
                String op = matcher.group(1);
                double threshold = Double.parseDouble(matcher.group(2));
                DifferentialFunction df = buildDF(col, op, threshold, columnNames, provider);
                return df == null ? null : registerDF(df, dfPool);
            }
        } catch (Exception e) {
            System.err.println(" RHS parsing failed: " + rhsStr);
        }
        return null;
    }

    private static DifferentialFunction buildDF(String colName, String op, double threshold,
                                                List<String> columnNames,
                                                IndexProvider<?> provider) {
        if (!columnNames.contains(colName)) {
            System.out.println(" Attribute name mismatch: " + colName + " not found in " + columnNames);
            return null;
        }
        int colIndex = columnNames.indexOf(colName);

        Operator operator = op.equals("<=") ? Operator.LESS_EQUAL : Operator.GREATER;
        ColumnOperand<String> operand = MyOperandHelper.buildStringOperand(colName, colIndex);
        return new DifferentialFunction(operator, threshold, operand);
    }

    private static DifferentialFunction registerDF(DifferentialFunction df,
                                                   Map<DifferentialFunction, Set<String>> dfPool) {
        for (DifferentialFunction exist : dfPool.keySet()) {
            if (exist.equals(df)) return exist;
        }
        dfPool.put(df, new HashSet<>()); 
        return df;
    }
}
