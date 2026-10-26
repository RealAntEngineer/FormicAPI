package com.rae.formicapi.foundation.math.operators.linear;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuIntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;

import java.util.Arrays;

/**
 * Mutable sparse matrix using padded CSR and CSC layouts.
 *
 * <p>The matrix maintains row-oriented and column-oriented storage to
 * support efficient matrix-vector multiplication in both directions:
 * {@code A * x} and {@code A^T * x}.
 *
 * <p>Each row and column has a fixed storage capacity, specified by
 * {@code nnzPerRow} and {@code nnzPerCol}, respectively. Unused slots
 * are padded with zero values.
 *
 * <p>The matrix supports numerical updates and allows entries to be set
 * or added at arbitrary coordinates. Structural changes may require
 * rebuilding the column-oriented layout to keep both storage layouts
 * consistent.
 *
 * <p>Designed for large sparse systems solved using iterative linear
 * solvers that require efficient multiplication by both the matrix and
 * its transpose.
 */
@SuppressWarnings("unused")
public class PaddedCSRCSCMatrixD implements MutableMatrix {

    private int rows;
    private int cols;
    private final int nnzPerRow;
    private final int nnzPerCol;

    /**
     * CSR values (mutable)
     */
    private double[] rowValues;
    private double[] colValues;


    /**
     * CSR column indices
     */
    private int[] colIndex;
    private int[] rowIndex;

    private boolean needColSet = false;


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
    public PaddedCSRCSCMatrixD(int rows, int cols, int nnzPerRow, int nnzPerCol) {

        this.rows = rows;
        this.cols = cols;
        this.nnzPerRow = nnzPerRow;
        this.nnzPerCol = nnzPerCol;

        this.rowValues = new double[rows * nnzPerRow];
        this.colValues = new double[cols * nnzPerCol];
        this.colIndex = new int[rows * nnzPerRow];
        this.rowIndex = new int[cols * nnzPerCol];

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
    public void resize(int newRows, int newCols) {
        int requiredRowLength = newRows * nnzPerRow;
        int requiredColLength = newCols * nnzPerCol;

        if (requiredRowLength > rowValues.length) {
            rowValues = Arrays.copyOf(rowValues, requiredRowLength);
            colIndex = Arrays.copyOf(colIndex, requiredRowLength);
        }

        if (requiredColLength > colValues.length) {
            colValues = Arrays.copyOf(colValues, requiredColLength);
            rowIndex = Arrays.copyOf(rowIndex, requiredColLength);
        }

        rows = newRows;
        cols = newCols;
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

        rowValues[findRowIndex(row, col)] += (double) value;
        colValues[findColIndex(row, col)] += (double) value;

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
        rowValues[findRowIndex(row, col)] = (double) value;
        colValues[findColIndex(row, col)] = (double) value;
    }

    /**
     * Finds CSR index for a (row, col) pair.
     *
     * @throws IllegalStateException if entry does not exist
     */
    private int findRowIndex(int row, int col) {
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
     * Finds CSC index for a (row, col) pair.
     *
     * @throws IllegalStateException if entry does not exist
     */
    private int findColIndex(int row, int col) {
        int base = col * nnzPerCol;

        for (int i = 0; i < nnzPerCol; i++) {
            if (rowIndex[base + i] == row) {
                return base + i;
            }
        }

        throw new IllegalStateException(
                "CSC entry does not exist: (" + row + "," + col + ")"
        );
    }


    @Override
    public void setRow(int row, double[] newValues, int[] newCols, int count) {

        if (newValues.length > nnzPerRow || newCols.length != newValues.length) {
            throw new IllegalArgumentException(
                    "Expected arrays of size " + nnzPerRow + " but got values=" + newValues.length +
                            " cols=" + newCols.length
            );
        }

        int base = row * nnzPerRow;
        for (int i = 0; i < count; i++) {
            colIndex[base + i] = newCols[i];
            rowValues[base + i] = (double) newValues[i];

            //colValues[findRowIndex(row, newCols[i])]

        }
        // zero out trailing slots — both value AND colIndex set to safe default (diagonal)
        for (int i = count; i < nnzPerRow; i++) {
            colIndex[base + i] = 0;//garbage index (it's zeroed anyway)
            //rowIndex[base + i] =
            rowValues[base + i] = 0.0f;
        }

        for (int i = count; i < nnzPerCol; i++) {
            rowIndex[base + i] = 0; // points to diagonal — safe for both multiply paths
            colValues[base + i] = 0.0f;
        }

        needColSet = true;
        //inefficient but \_":)"_/
    }

    @Override
    public void multiply(double[] x, double[] result) {
        for (int r = 0; r < rows; r++) {

            int    base = r * nnzPerRow;
            double sum  = 0.0;

            for (int i = 0; i < nnzPerRow; i++) {
                sum += rowValues[base + i] * x[colIndex[base + i]];
            }

            result[r] = sum;
        }
    }

    @Override
    public void apply(RealVector x, RealVector result) {
        if (x instanceof CpuDoubleVector xCpu && result instanceof CpuDoubleVector resCpu) {
            double[] resArr = resCpu.array();
            double[] xArr = xCpu.array();
            Arrays.fill(resArr, 0.0f);

            for (int r = 0; r < rows; r++) {

                int   base = r * nnzPerRow;
                double sum  = 0.0f;

                for (int i = 0; i < nnzPerRow; i++) {
                    sum += rowValues[base + i] * xArr[colIndex[base + i]];
                }

                resArr[r] = sum;
            }
        } else {
            throw new UnsupportedOperationException(
                    "Matrix.apply(...) PaddedCSRCSCMatrixF only supports CpuDoubleVector operands; got x="
                            + x.getClass().getSimpleName() + ", result=" + result.getClass().getSimpleName());
        }
    }


    private void buildColumnLayout() {
        Arrays.fill(colValues, 0.0f);
        Arrays.fill(rowIndex, 0);

        int[] next = new int[cols];

        for (int r = 0; r < rows; r++) {
            int base = r * nnzPerRow;

            for (int i = 0; i < nnzPerRow; i++) {
                int p = base + i;
                int c = colIndex[p];

                if (c < 0 || c >= cols)
                    throw new IllegalStateException("Invalid column index: " + c);

                if (next[c] >= nnzPerCol)
                    throw new IllegalStateException("Column capacity exceeded: " + c);

                int dst = c * nnzPerCol + next[c]++;
                colValues[dst] = rowValues[p];
                rowIndex[dst] = r;
            }
        }

        needColSet = false;
    }

    @Override
    public void transposeApply(RealVector x, RealVector result) {

        if (needColSet){
            buildColumnLayout();//scan the rows and build the cols
        }

        if (x instanceof CpuDoubleVector xCpu && result instanceof CpuDoubleVector resCpu) {
            double[] resArr = resCpu.array();
            double[] xArr = xCpu.array();
            Arrays.fill(resArr, 0.0f);

            for (int c = 0; c < cols; c++) {

                int base = c * nnzPerCol;
                double sum = 0.0f;

                for (int i = 0; i < nnzPerCol; i++) {
                     sum += colValues[base + i] * xArr[rowIndex[base + i]];
                }

                resArr[c] = sum;
            }
        } else {
            throw new UnsupportedOperationException(
                    "Matrix.transposeApply(...) PaddedCSRCSCMatrixF only supports CpuDoubleVector operands; got x="
                            + x.getClass().getSimpleName() + ", result=" + result.getClass().getSimpleName());
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
        int baseR = r * nnzPerRow;
        //int baseC = c * nnzPerCol;

        for (int i = 0; i < nnzPerRow; i++) {
            if (colIndex[baseR + i] == c) {
                return rowValues[baseR + i];
            }
        }

        return 0.0;
    }

    @Override
    public RealVector getValues(IntegerVector r, IntegerVector c) {
        if (r instanceof CpuIntegerVector rCpu && c instanceof CpuIntegerVector cCpu) {
            int[] rArr = rCpu.array();
            int[] cArr = cCpu.array();

            double[] values = new double[rCpu.size()];
            for (int i = 0; i < rCpu.size(); i++) {
                values[i] = get(rArr[i], cArr[i]);
            }

            return new CpuDoubleVector(values);

        } else {
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public double[] getRowValues(int r) {
        int base = r * nnzPerRow;

        double[] out = new double[nnzPerRow];
        System.arraycopy(rowValues, base, out, 0, nnzPerRow);

        return out;
    }

    @Override
    public int[] getRowCols(int r) {
        int base = r * nnzPerRow;

        int[] out = new int[nnzPerRow];
        System.arraycopy(colIndex, base, out, 0, nnzPerRow);

        return out;
    }

}