package com.rae.formicapi.math.operators;

import java.util.*;

public class CSRMatrix implements Matrix {

    private final int rows;
    private final int cols;

    private final double[] values;
    private final int[] colIndex;
    private final int[] rowPtr;

    public CSRMatrix(
            int rows,
            int cols,
            double[] values,
            int[] colIndex,
            int[] rowPtr) {

        this.rows = rows;
        this.cols = cols;
        this.values = values;
        this.colIndex = colIndex;
        this.rowPtr = rowPtr;
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int cols() {
        return cols;
    }

    @Override
    public double get(int r, int c) {

        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {

            if (colIndex[k] == c)
                return values[k];
        }

        return 0;
    }

    @Override
    public void multiply(double[] x, double[] result) {

        for (int r = 0; r < rows; r++) {

            double sum = 0;

            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {

                sum += values[k] * x[colIndex[k]];
            }

            result[r] = sum;
        }
    }

    @Override
    public void transposeMultiply(double[] x, double[] result) {
        Arrays.fill(result, 0.0);
        for (int r = 0; r < rows; r++) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                result[colIndex[k]] += values[k] * x[r];
            }
        }
    }

    public static CSRMatrix fromHash(HashSparseMatrix hash) {

        int rows = hash.rows();
        int cols = hash.cols();

        List<Double> valuesList = new ArrayList<>();
        List<Integer> colIndexList = new ArrayList<>();

        int[] rowPtr = new int[rows + 1];

        int count = 0;

        for (int r = 0; r < rows; r++) {

            rowPtr[r] = count;

            Map<Integer, Double> row = hash.getRow(r);

            if (row != null) {

                List<Integer> sortedCols = new ArrayList<>(row.keySet());
                Collections.sort(sortedCols);

                for (int c : sortedCols) {

                    valuesList.add(row.get(c));
                    colIndexList.add(c);

                    count++;
                }
            }
        }

        rowPtr[rows] = count;

        double[] values = new double[count];
        int[] colIndex = new int[count];

        for (int i = 0; i < count; i++) {

            values[i] = valuesList.get(i);
            colIndex[i] = colIndexList.get(i);
        }

        return new CSRMatrix(rows, cols, values, colIndex, rowPtr);
    }
}