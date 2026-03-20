package com.rae.formicapi.fondation.math.operators;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A mutable sparse matrix backed by a nested {@link HashMap}.
 *
 * <p>Entries are stored as a map of row index to a map of column index to value.
 * Only non-zero entries are stored — zero values passed to {@link #add} or
 * {@link #set} are silently ignored or removed, keeping the structure lean.
 *
 * <p>This implementation is optimised for assembly (random-access writes) rather
 * than repeated multiplication. For solve-phase performance, convert to
 * {@link CSRMatrix} via {@link #toCSR()}, or use {@link DynamicCSRMatrix} directly.
 *
 * @see CSRMatrix
 * @see DynamicCSRMatrix
 * @see MutableMatrix
 */
public class HashSparseMatrix implements MutableMatrix {

    private final int rows;
    private final int cols;

    private final Map<Integer, Map<Integer, Double>> data = new HashMap<>();

    public HashSparseMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    @Override
    public void add(int r, int c, double v) {
        if (v == 0.0) return;
        data.computeIfAbsent(r, k -> new HashMap<>())
                .merge(c, v, Double::sum);
    }

    @Override
    public void set(int r, int c, double v) {
        if (v == 0.0) {
            Map<Integer, Double> row = data.get(r);
            if (row != null) row.remove(c);
            return;
        }
        data.computeIfAbsent(r, k -> new HashMap<>())
                .put(c, v);
    }

    @Override
    public void multiply(double[] x, double[] result) {
        Arrays.fill(result, 0);
        for (var rowEntry : data.entrySet()) {
            int r = rowEntry.getKey();
            for (var colEntry : rowEntry.getValue().entrySet())
                result[r] += colEntry.getValue() * x[colEntry.getKey()];
        }
    }

    /**
     * Multiplies the transpose of this matrix by vector {@code x},
     * storing Aᵀx in {@code result}.
     *
     * <p>Iterates directly over stored non-zeros, scattering each
     * contribution into the result. O(nnz) — overrides the default
     * O(rows × cols) implementation in {@link Matrix}.
     *
     * @param x      input vector of length rows
     * @param result output vector of length cols, overwritten with Aᵀx
     */
    @Override
    public void transposeMultiply(double[] x, double[] result) {
        Arrays.fill(result, 0);
        for (var rowEntry : data.entrySet()) {
            int r = rowEntry.getKey();
            double xr = x[r];
            for (var colEntry : rowEntry.getValue().entrySet())
                result[colEntry.getKey()] += colEntry.getValue() * xr;
        }
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
        Map<Integer, Double> row = data.get(r);
        if (row == null) return 0;
        return row.getOrDefault(c, 0.0);
    }

    public CSRMatrix toCSR() {
        return CSRMatrix.fromHash(this);
    }

    public Map<Integer, Double> getRow(int r) {
        return data.get(r);
    }
}