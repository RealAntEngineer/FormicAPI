package com.rae.formicapi.fondation.math.operators;

import java.util.*;

/**
 * Represents an immutable sparse matrix in Compressed Sparse Row (CSR) format.
 *
 * <p>CSR stores only non-zero entries, giving O(nnz) time for matrix-vector
 * multiply and transpose multiply. Random access via {@link #get(int, int)}
 * requires a linear scan of the row's stored entries and should be avoided
 * in hot loops.
 *
 * <p>Instances are typically created via {@link #fromHash(HashSparseMatrix)}
 * during the transition from assembly to solve phase.
 *
 * @see Matrix
 * @see HashSparseMatrix
 * @see DynamicCSRMatrix
 */
public class CSRMatrix implements Matrix {

    private final int rows;
    private final int cols;

    private final double[] values;
    private final int[] colIndex;
    private final int[] rowPtr;

    /**
     * Constructs a CSR matrix from raw arrays.
     *
     * <p>The arrays must be consistent:
     * <ul>
     *     <li>{@code values} and {@code colIndex} must have the same length (nnz)</li>
     *     <li>{@code rowPtr} must have length {@code rows + 1}</li>
     *     <li>{@code rowPtr[rows]} must equal nnz</li>
     *     <li>Zero values in {@code values} are silently skipped to preserve sparsity</li>
     * </ul>
     *
     * <p>Prefer {@link #fromHash(HashSparseMatrix)} over this constructor
     * unless building CSR arrays manually.
     *
     * @param rows     number of rows
     * @param cols     number of columns
     * @param values   non-zero values
     * @param colIndex column indices parallel to {@code values}
     * @param rowPtr   row pointer array of length {@code rows + 1}
     */
    public CSRMatrix(int rows, int cols, double[] values, int[] colIndex, int[] rowPtr) {
        this.rows = rows;
        this.cols = cols;
        this.values = values;
        this.colIndex = colIndex;
        this.rowPtr = rowPtr;
    }

    /**
     * Builds a {@link CSRMatrix} from a {@link HashSparseMatrix}.
     *
     * <p>Rows are sorted by column index as required by the CSR format.
     * Zero entries are excluded — {@link HashSparseMatrix} already prevents
     * them from being stored, so this is a safeguard for direct callers.
     *
     * @param hash the source hash matrix
     * @return a compiled CSR representation
     */
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
                    double v = row.get(c);
                    if (v == 0.0) continue; // safeguard — should not occur
                    valuesList.add(v);
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

    /**
     * Multiplies this matrix by vector {@code x}, storing Ax in {@code result}.
     *
     * <p>Iterates over stored non-zeros only. O(nnz).
     *
     * @param x      input vector of length {@link #cols()}
     * @param result output vector of length {@link #rows()}, overwritten with Ax
     */
    @Override
    public void multiply(double[] x, double[] result) {
        for (int r = 0; r < rows; r++) {
            double sum = 0;
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++)
                sum += values[k] * x[colIndex[k]];
            result[r] = sum;
        }
    }

    /**
     * Multiplies the transpose of this matrix by vector {@code x},
     * storing Aᵀx in {@code result}.
     *
     * <p>Iterates over stored non-zeros, scattering each contribution
     * into the result. O(nnz).
     *
     * @param x      input vector of length {@link #rows()}
     * @param result output vector of length {@link #cols()}, overwritten with Aᵀx
     */
    @Override
    public void transposeMultiply(double[] x, double[] result) {
        Arrays.fill(result, 0.0);
        for (int r = 0; r < rows; r++)
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++)
                result[colIndex[k]] += values[k] * x[r];
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int cols() {
        return cols;
    }

    /**
     * Returns the value at (r, c).
     *
     * <p>Performs a linear scan of the stored entries in row {@code r}.
     * O(nnz_per_row) — avoid calling this in hot loops; use
     * {@link #multiply} or {@link #transposeMultiply} instead.
     *
     * @param r row index
     * @param c column index
     * @return the scalar value at (r, c), or {@code 0.0} if not stored
     */
    @Override
    public double get(int r, int c) {
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++)
            if (colIndex[k] == c) return values[k];
        return 0;
    }
}