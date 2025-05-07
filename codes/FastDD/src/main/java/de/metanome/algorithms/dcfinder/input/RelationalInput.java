package de.metanome.algorithms.dcfinder.input;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RelationalInput {

    private BufferedReader reader;
    public int numberOfColumns;
    public String relationName;
    public String[] columnNames;
    private String nextLine;  // ✅ 改成读取的原始行
    public String filePath;

    public RelationalInput(File file) throws IOException {
        reader = new BufferedReader(new FileReader(file));
        columnNames = reader.readLine().split(",");
        numberOfColumns = columnNames.length;
        relationName = file.getName();
        filePath = file.getPath();
    }

    public boolean hasNext() {
        try {
            nextLine = reader.readLine(); // ✅ 实际读取行
            return nextLine != null;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public List<String> next() {
        if (nextLine == null) return Collections.emptyList();
        String[] values = nextLine.split(",", -1);
        List<String> row = new ArrayList<>();
        Collections.addAll(row, values);
        nextLine = null;  // ✅ 清空，防止被多次调用重复返回
        return row;
    }

    public int numberOfColumns() {
        return numberOfColumns;
    }

    public String relationName() {
        return relationName;
    }

    public String[] columnNames() {
        return columnNames;
    }
}
