package com.rae.formicapi.math.operators;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


public class HashSparseMatrix implements MutableMatrix {

    private final int rows;
    private final int cols;

    private final Map<Integer, Map<Integer, Double>> data = new HashMap<>();

    public HashSparseMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
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
    public void add(int r, int c, double v) {

        data.computeIfAbsent(r, k -> new HashMap<>())
                .merge(c, v, Double::sum);
    }

    @Override
    public void set(int r, int c, double v) {

        data.computeIfAbsent(r, k -> new HashMap<>())
                .put(c, v);
    }

    @Override
    public double get(int r, int c) {

        Map<Integer, Double> row = data.get(r);

        if (row == null) return 0;

        return row.getOrDefault(c, 0.0);
    }

    @Override
    public void multiply(double[] x, double[] result) {

        Arrays.fill(result, 0);

        for (var rowEntry : data.entrySet()) {

            int r = rowEntry.getKey();

            for (var colEntry : rowEntry.getValue().entrySet()) {

                result[r] += colEntry.getValue() * x[colEntry.getKey()];
            }
        }
    }

    public CSRMatrix toCSR() {
        return CSRMatrix.fromHash(this);
    }

    public Map<Integer, Double> getRow(int r) {
        return data.get(r);
    }
}