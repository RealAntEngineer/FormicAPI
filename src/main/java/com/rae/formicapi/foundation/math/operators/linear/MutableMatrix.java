package com.rae.formicapi.foundation.math.operators.linear;

public interface MutableMatrix extends Matrix {

//TODO add fluent chaining
    /**
     * Resizes this matrix in-place.
     *
     * <p>If growing, new entries are zero initialized and column indices are
     * initialized to 0. Caller must populate new rows using {@link #setRow}.
     * If shrinking, trailing rows are discarded logically.
     *
     * @param newRows new row/column count
     */
    void resize(int newRows);

    void add(int row, int col, double value);

    void set(int row, int col, double value);

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
     * @param rowValues new non-zero values for the row
     * @param rowCols   column indices corresponding to each value
     * @throws IllegalArgumentException if array sizes do not match {@code nnzPerRow}
     */
    void setRow(int row, double[] rowValues, int[] rowCols, int count);

    int[] getRowCols(int r);

    double[] getRowValues(int r);
}