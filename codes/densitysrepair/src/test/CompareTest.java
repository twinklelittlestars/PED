package test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

import algorithm.Heuristic;
import algorithm.Relaxation;
import entity.CFD;
import entity.Tuple;
import entity.Database;
import util.FileHandler;

public class CompareTest {

	private Database db;

	public Database getDb() {
		return db;
	}

	public void setDb(Database db) {
		this.db = db;
	}

	public CompareTest() {

	}

	public static void main(String[] args) {
		// CompareTest test = new CompareTest();
		// String name = "hospital";
		String name = args[0];
		String dataFilename = name + ".data";
		String dirtyDataFilename = name + "-dirty.data";
		FileHandler fh = new FileHandler();
		CompareTest test = new CompareTest(); 
		test.setDb(fh.readDirtyData(dirtyDataFilename));
		Database db = test.getDb();
		db.setCleanTpList(fh.readCleanData(dataFilename));
		ArrayList<CFD> cfdList = fh.readRule("cfd.final");
		// ArrayList<Integer> groundTruth = getTrueErrorCells(db.getDirtyTpList(), db.getCleanTpList());
		// System.out.println("Ground Truth: " + groundTruth.size() + " errors");
		// for (int i = 0; i < groundTruth.size(); i++) {
		// 	System.out.print(groundTruth.get(i) + " ");
		// }
		// System.out.println();
		final int K = 10;

		Heuristic heuristic = new Heuristic(db);
		heuristic.setK(K);
		heuristic.setCfdList(cfdList);
		long startTime1 = System.currentTimeMillis();
		heuristic.mainHeuristic();
		long endTime1 = System.currentTimeMillis();
		// evaluate("Heuristic", heuristic.getDetectedRowIndexList(), groundTruth, startTime1, endTime1);
		evaluateCellLevel("Heuristic", heuristic.getDetectedRowIndexList(), heuristic.getCfdToViolationRows(), db.getDirtyTpList(), db.getCleanTpList(), cfdList, startTime1, endTime1);


		Relaxation relaxation = new Relaxation(db);
		relaxation.setK(K);
		relaxation.setCfdList(cfdList);
		long startTime2 = System.currentTimeMillis();
		relaxation.mainRelaxation();
		long endTime2 = System.currentTimeMillis();
		// evaluate("Relaxation", relaxation.getDetectedRowIndexList(), groundTruth, startTime2, endTime2);
		evaluateCellLevel("Relaxation", relaxation.getDetectedRowIndexList(), relaxation.getCfdToViolationRows(), db.getDirtyTpList(), db.getCleanTpList(), cfdList, startTime2, endTime2);
	}

	// public static void evaluate(String method, ArrayList<Integer> detected, ArrayList<Integer> truth, long startTime, long endTime) {
	// 	HashSet<Integer> detectedSet = new HashSet<>(detected);
	// 	HashSet<Integer> truthSet = new HashSet<>(truth);
	// 	// System.out.println("Detected Set: " + detectedSet);
	// 	// System.out.println("Truth Set:    " + truthSet);

	// 	int tp = 0;
	// 	for (Integer id : detectedSet) {
	// 		if (truthSet.contains(id)) {
	// 			tp++;
	// 		}
	// 	}
	// 	int fp = detectedSet.size() - tp;
	// 	int fn = truthSet.size() - tp;
	// 	System.out.println("TP: " + tp + ", FP: " + fp + ", FN: " + fn);

	// 	double precision = tp / (double)(tp + fp + 1e-10);
	// 	double recall = tp / (double)(tp + fn + 1e-10);
	// 	double f1 = 2 * precision * recall / (precision + recall + 1e-10);
	// 	double seconds = (endTime - startTime) / 1000.0;

	// 	System.out.println("======== " + method + " Evaluation ========");
	// 	System.out.printf("Precision: %.4f\n", precision);
	// 	System.out.printf("Recall:    %.4f\n", recall);
	// 	System.out.printf("F1 Score:  %.4f\n", f1);
	// 	System.out.printf("Time:      %.2f seconds\n", seconds);
	// 	System.out.println();
	// }

	// public static ArrayList<Integer> getTrueErrorRows(ArrayList<Tuple> dirtyList, ArrayList<Tuple> cleanList) {
	// 	ArrayList<Integer> errorRows = new ArrayList<>();
	// 	for (int i = 0; i < dirtyList.size(); i++) {
	// 		String[] dirty = dirtyList.get(i).getAllData();
	// 		String[] clean = cleanList.get(i).getAllData();
	// 		for (int j = 0; j < dirty.length; j++) {
	// 			if (!dirty[j].equals(clean[j])) {
	// 				// System.out.println("Mismatch at row " + i + ", col " + j + ": dirty='" + dirty[j] + "', clean='" + clean[j] + "'");
	// 				errorRows.add(i);
	// 				break;
	// 			}
	// 		}
	// 	}
	// 	return errorRows;
	// }
	public static void evaluateCellLevel(String method, ArrayList<Integer> detectedRows, Map<CFD, Set<Integer>> cfdToViolationRows, ArrayList<Tuple> dirtyList, ArrayList<Tuple> cleanList, ArrayList<CFD> cfdList, long startTime, long endTime) {
			Set<String> predictedCells = new HashSet<>();
			for (Integer row : detectedRows) {
					for (CFD cfd : cfdList) {
							if (cfdToViolationRows.containsKey(cfd) && cfdToViolationRows.get(cfd).contains(row)) {
									for (int attr : cfd.getAttrXs()) {
											predictedCells.add(row + "," + attr);
									}
									predictedCells.add(row + "," + cfd.getAttrY());
							}
					}
			}

			Set<String> trueCells = getTrueErrorCells(dirtyList, cleanList);

			int tp = 0;
			for (String cell : predictedCells) {
					if (trueCells.contains(cell)) tp++;
			}
			int fp = predictedCells.size() - tp;
			int fn = trueCells.size() - tp;

			double precision = tp / (double)(tp + fp + 1e-10);
			double recall = tp / (double)(tp + fn + 1e-10);
			double f1 = 2 * precision * recall / (precision + recall + 1e-10);
			double seconds = (endTime - startTime) / 1000.0;

			System.out.println("======== " + method + " Cell-Level Evaluation ========");
			System.out.printf("Precision: %.4f\n", precision);
			System.out.printf("Recall:    %.4f\n", recall);
			System.out.printf("F1 Score:  %.4f\n", f1);
			System.out.printf("Time:      %.2f seconds\n", seconds);
			System.out.println();
	}

	public static Set<String> getTrueErrorCells(ArrayList<Tuple> dirtyList, ArrayList<Tuple> cleanList) {
			Set<String> errorCells = new HashSet<>();
			for (int i = 0; i < dirtyList.size(); i++) {
					String[] dirty = dirtyList.get(i).getAllData();
					String[] clean = cleanList.get(i).getAllData();
					for (int j = 0; j < dirty.length; j++) {
							if (!dirty[j].equals(clean[j])) {
									errorCells.add(i + "," + j);
							}
					}
			}
			return errorCells;
	}


}
