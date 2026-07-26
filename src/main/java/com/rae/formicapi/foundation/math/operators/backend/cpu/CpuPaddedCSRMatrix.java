package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

public class CpuPaddedCSRMatrix implements MutableMatrix {

    CpuExecutor executor;
    private int rows;
    private int cols;
    private final int nnzPerRow;

    /**
     * CSR values (mutable)
     */
    private double[] values;

    /**
     * CSR column indices
     */
    private int[] colIndex;

    /**
     * Constructs a fully defined CSR matrix.
     *
     * <p>The structure MUST already be consistent:
     * <ul>
     *     <li>{@code values.length == colIndex.length == nnz}</li>
     *     <li>{@code rowPtr.length == rows + 1}</li>
     *     <li>{@code rowPtr[rows] == nnz}</li>
     * </ul>
     *
     * <p>This constructor builds the fast lookup table used for updates.
     *
     * @param rows number of rows
     * @param cols number of columns
     *
     */
    public CpuPaddedCSRMatrix(int rows, int cols, int nnzPerRow) {

        this.rows = rows;
        this.cols = cols;
        this.nnzPerRow = nnzPerRow;

        this.values = new double[rows * nnzPerRow];
        this.colIndex = new int[rows * nnzPerRow];
    }

    private CpuPaddedCSRMatrix(int rows, int cols, int nnzPerRow, double[] values, int[] colIndex) {
        this.rows = rows;
        this.cols = cols;
        this.nnzPerRow = nnzPerRow;
        this.values = values;
        this.colIndex = colIndex;
    }

    /**
     * Resizes this matrix in-place.
     *
     * <p>If growing, new entries are zero initialized and column indices are
     * initialized to 0. Caller must populate new rows using {@link #setRow}.
     * If shrinking, trailing rows are discarded logically.
     *
     * @param newRows new row/column count
     */
    public void resize(int newRows) {
        int requiredLength = newRows * nnzPerRow;

        if (requiredLength > values.length) {
            values = Arrays.copyOf(values, requiredLength);
            colIndex = Arrays.copyOf(colIndex, requiredLength);
        }

        rows = newRows;
        cols = newRows;
    }
    /**
     * Adds a value to an existing entry in the CSR structure.
     *
     * <p>This method only works if the (row, col) entry already exists.
     * If it does not exist, an {@link IllegalStateException} is thrown.
     *
     * <p>Complexity: O(nnz_per_row) worst case (typically ~7).
     *
     * @param row   row index
     * @param col   column index
     * @param value value to add
     */
    @Override
    public void add(int row, int col, double value) {
        if (value == 0.0) return;

        int idx = findIndex(row, col);
        values[idx] += value;
    }

    /**
     * Sets a value in an existing CSR entry.
     *
     * <p>If the value is 0, the method simply writes 0 but does not remove
     * the entry (structure is fixed).
     *
     * <p>If the (row, col) entry does not exist, an exception is thrown.
     *
     * @param row   row index
     * @param col   column index
     * @param value new value
     */
    @Override
    public void set(int row, int col, double value) {
        int idx = findIndex(row, col);
        values[idx] = value;
    }

    /**
     * Finds CSR index for a (row, col) pair.
     *
     * @throws IllegalStateException if entry does not exist
     */
    private int findIndex(int row, int col) {
        int base = row * nnzPerRow;

        for (int i = 0; i < nnzPerRow; i++) {
            if (colIndex[base + i] == col) {
                return base + i;
            }
        }

        throw new IllegalStateException(
                "CSR entry does not exist: (" + row + "," + col + ")"
        );
    }

    /**
     * Overwrites an entire row of the matrix, including both its structure
     * (column indices) and its values.
     *
     * <p>This method assumes a fixed row capacity defined by {@code nnzPerRow}.
     * The provided {@code cols} array defines the column positions for each
     * entry in the row, and {@code newValues} defines the corresponding
     * numerical values.
     *
     * <p>After this call:
     * <ul>
     *     <li>The previous contents of the row are fully replaced</li>
     *     <li>The column structure of the row is updated to match {@code cols}</li>
     *     <li>No resizing of the underlying storage occurs</li>
     * </ul>
     *
     * <p><b>Important:</b> This operation modifies the sparsity structure of the row.
     * All subsequent operations (e.g., multiplication) will use the updated
     * column layout.
     *
     * <p>Constraints:
     * <ul>
     *     <li>{@code newValues.length == nnzPerRow}</li>
     *     <li>{@code cols.length == nnzPerRow}</li>
     * </ul>
     *
     * <p>Performance: O(nnzPerRow)
     *
     * @param row       row index to modify
     * @param newValues new non-zero values for the row
     * @param newCols   column indices corresponding to each value
     * @throws IllegalArgumentException if array sizes do not match {@code nnzPerRow}
     */
    public void setRow(int row, double[] newValues, int[] newCols, int count) {

        if (newValues.length > nnzPerRow || newCols.length != newValues.length) {
            throw new IllegalArgumentException(
                    "Expected arrays of size " + nnzPerRow +
                            " but got values=" + newValues.length +
                            " cols=" + newCols.length
            );
        }

        int base = row * nnzPerRow;
        for (int i = 0; i < count; i++) {
            colIndex[base + i] = newCols[i];
            values[base + i] = newValues[i];
        }
        // zero out trailing slots — both value AND colIndex set to safe default (diagonal)
        for (int i = count; i < nnzPerRow; i++) {
            colIndex[base + i] = row; // points to diagonal — safe for both multiply paths
            values[base + i] = 0.0;
        }
    }


    @Override
    public void apply(Vector x, Vector result) {
        if (executor  == null) throw new RuntimeException("Executor wasn't setup");//TODO maybe default to serial ?
        if (x instanceof CpuVector xCpu && result instanceof CpuVector resCpu) {
            double[] xArr = xCpu.array();
            double[] resultArr = resCpu.array();

            if (rows < executor.getParallelThreshold()) {
                applyRange(0, rows, xArr, resultArr);
                return;
            }

            executor.parallelFor(rows, (start, end) -> applyRange(start, end, xArr, resultArr));
        } else {
            throw new IllegalArgumentException("For a cpu backend matrix, you need to cpu backend vectors");
        }
    }

    private void applyRange(int start, int end, double[] xArr, double[] resultArr) {
        for (int row = start; row < end; row++) {
            int    base = row * nnzPerRow;
            double sum  = 0.0;

            for (int i = 0; i < nnzPerRow; i++) {
                sum += values[base + i] * xArr[colIndex[base + i]];
            }

            resultArr[row] = sum;
        }
    }

    @Override
    public void multiply(double[] x, double[] result) {
        throw new RuntimeException("Unsuported, use vector version instead");
        /*for (int r = 0; r < rows; r++) {

            int    base = r * nnzPerRow;
            double sum  = 0.0;

            for (int i = 0; i < nnzPerRow; i++) {
                sum += values[base + i] * x[colIndex[base + i]];
            }

            result[r] = sum;
        }*/
    }


    @Override
    public void transposeApply(Vector x, Vector result) {
        if (executor == null) throw new RuntimeException("Executor wasn't setup");
        if (x instanceof CpuVector xCpu && result instanceof CpuVector resCpu) {
            double[] xArr = xCpu.array();
            double[] resultArr = resCpu.array();

            Arrays.fill(resultArr, 0.0);

            if (rows < executor.getParallelThreshold()) {
                transposeApplyRange(0, rows, xArr, resultArr);
                return;
            }

            final double[] vals = values;
            final int[] cols = colIndex;
            final int nnz = nnzPerRow;

            // Scatter-add: two different row ranges can write the same
            // output column, so a plain parallelFor would race on
            // `resultArr`. Each worker accumulates into its own private
            // buffer instead, and the buffers get summed into resultArr
            // once every row is done.
            executor.parallelForAccumulate(rows, resultArr, (start, end, local) -> {
                for (int r = start; r < end; r++) {
                    int base = r * nnz;
                    double xr = xArr[r];
                    for (int i = 0; i < nnz; i++)
                        local[cols[base + i]] += vals[base + i] * xr;
                }
            });
        } else {
            throw new IllegalArgumentException("For a cpu backend matrix, you need to cpu backend vectors");
        }
    }

    private void transposeApplyRange(int start, int end, double[] xArr, double[] resultArr) {
        for (int r = start; r < end; r++) {
            int base = r * nnzPerRow;
            double xr = xArr[r];
            for (int i = 0; i < nnzPerRow; i++) {
                resultArr[colIndex[base + i]] += values[base + i] * xr;
            }
        }
    }

    @Override
    public void transposeMultiply(double[] x, double[] result) {
        throw new RuntimeException("Unsuported, use vector version instead");
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
        int base = r * nnzPerRow;

        for (int i = 0; i < nnzPerRow; i++) {
            if (colIndex[base + i] == c) {
                return values[base + i];
            }
        }

        return 0.0;
    }

    public double[] getRowValues(int r) {
        int base = r * nnzPerRow;

        double[] out = new double[nnzPerRow];
        System.arraycopy(values, base, out, 0, nnzPerRow);

        return out;
    }

    public int[] getRowCols(int r) {
        int base = r * nnzPerRow;

        int[] out = new int[nnzPerRow];
        System.arraycopy(colIndex, base, out, 0, nnzPerRow);

        return out;
    }

    public void setExecutor(CpuExecutor cpuExecutor) {
        executor = cpuExecutor;
    }
}