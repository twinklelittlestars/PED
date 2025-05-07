package fastdd.evaluation;

import de.metanome.algorithms.dcfinder.input.ParsedColumn;
import de.metanome.algorithms.dcfinder.predicates.operands.ColumnOperand;
import de.metanome.algorithms.dcfinder.helpers.IndexProvider;

public class MyOperandHelper {
    public static ColumnOperand<String> buildStringOperand(String colName, int colIndex) {
        ParsedColumn<String> parsedColumn = new ParsedColumn<>(colName, String.class, colIndex, null);
        return new ColumnOperand<>(parsedColumn, colIndex);
    }
}
