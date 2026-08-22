package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import org.jocl.cl_mem;

import java.util.Arrays;

public class GpuPaddedCSRMatrix extends GpuExecutable implements MutableMatrix {

    private final int nnzPerRow;

    private int rows;
    private int cols;

    private double[] values;
    private int[] colIndex;

    private cl_mem dValues;
    private cl_mem dColIndex;

    public GpuPaddedCSRMatrix(int rows, int cols, int nnzPerRow) {
        if (rows < 0)
            throw new IllegalArgumentException("rows < 0");

        if (cols < 0)
            throw new IllegalArgumentException("cols < 0");

        if (nnzPerRow <= 0)
            throw new IllegalArgumentException("nnzPerRow <= 0");

        this.rows = rows;
        this.cols = cols;
        this.nnzPerRow = nnzPerRow;

        int length = rows * nnzPerRow;

        this.values = new double[length];
        this.colIndex = new int[length];
    }

    @Override
    public void setExecutor(GpuExecutor executor) {
        super.setExecutor(executor);

        if (dValues != null)
            executor.release(dValues);

        if (dColIndex != null)
            executor.release(dColIndex);

        dValues = executor.allocateDoubleBuffer(Math.max(values.length, 1));
        dColIndex = executor.allocateIntBuffer(Math.max(colIndex.length, 1));

        if (values.length > 0) {
            executor.uploadDoubles(dValues, values, values.length);
            executor.uploadInts(dColIndex, colIndex, colIndex.length);
        }
    }

    @Override
    public void apply(DoubleVector x, DoubleVector result) {

        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException(
                    "Unable to execute operation with a vector of class " + x.getClass()
            );

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException(
                    "Unable to execute operation with a vector of class " + result.getClass()
            );

        if (x.size() != cols)
            throw new IllegalArgumentException("x.size() != A.cols()");

        result.resize(rows);
        //ptr what ?
        requireExecutor().launchCsrMatvec(
                dValues,
                dColIndex,
                nnzPerRow,
                gpuX.buffer(),
                gpuResult.buffer(),
                rows
        );
    }

    @Override
    public void transposeApply(DoubleVector x, DoubleVector result) {
        requireExecutor();

        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException(
                    "Unable to execute operation with a vector of class " + x.getClass()
            );

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException(
                    "Unable to execute operation with a vector of class " + result.getClass()
            );

        if (x.size() != rows)
            throw new IllegalArgumentException("x.size() != A.rows()");

        result.resize(cols);

        gpuResult.clear();

        requireExecutor().launchCsrMatvecTranspose(
                dValues,
                dColIndex,
                nnzPerRow,
                gpuX.buffer(),
                gpuResult.buffer(),
                rows
        );
    }

    @Override
    public void add(int row, int col, double value) {
        int index = findIndex(row, col);

        values[index] += value;

        requireExecutor().uploadDoubleAt(
                dValues,
                index,
                values[index]
        );
    }

    @Override
    public void set(int row, int col, double value) {
        int index = findIndex(row, col);

        values[index] = value;

        requireExecutor().uploadDoubleAt(
                dValues,
                index,
                value
        );
    }

    @Override
    public double get(int row, int col) {
        checkRow(row);

        int base = row * nnzPerRow;

        for (int i = 0; i < nnzPerRow; i++) {
            if (colIndex[base + i] == col)
                return values[base + i];
        }

        return 0.0;
    }

    private int findIndex(int row, int col) {
        checkRow(row);

        int base = row * nnzPerRow;

        for (int i = 0; i < nnzPerRow; i++) {
            if (colIndex[base + i] == col)
                return base + i;
        }

        throw new IllegalArgumentException(
                "CSR entry does not exist: (" + row + ", " + col + ")"
        );
    }

    public void setRow(int row, double[] rowValues, int[] rowCols, int count) {
        checkRow(row);

        if (count < 0 || count > nnzPerRow)
            throw new IllegalArgumentException("count out of range");

        if (rowValues.length < count)
            throw new IllegalArgumentException("rowValues too small");

        if (rowCols.length < count)
            throw new IllegalArgumentException("rowCols too small");

        int base = row * nnzPerRow;

        for (int i = 0; i < count; i++) {
            values[base + i] = rowValues[i];
            colIndex[base + i] = rowCols[i];
        }

        /*
         * Unused padded entries are zero.
         */
        for (int i = count; i < nnzPerRow; i++) {
            values[base + i] = 0.0;
            colIndex[base + i] = row < cols ? row : 0;
        }

        requireExecutor().uploadDoubles(
                dValues,
                base,
                values,
                base,
                nnzPerRow
        );

        requireExecutor().uploadInts(
                dColIndex,
                base,
                colIndex,
                base,
                nnzPerRow
        );
    }

    public void resize(int newRows) {
        if (newRows < 0)
            throw new IllegalArgumentException("newRows < 0");

        int newLength = newRows * nnzPerRow;

        values = Arrays.copyOf(values, newLength);
        colIndex = Arrays.copyOf(colIndex, newLength);

        rows = newRows;

        requireExecutor();

        cl_mem newValues =
                executor.allocateDoubleBuffer(Math.max(newLength, 1));

        cl_mem newColIndex =
                executor.allocateIntBuffer(Math.max(newLength, 1));

        if (newLength > 0) {
            executor.uploadDoubles(
                    newValues,
                    values,
                    newLength
            );

            executor.uploadInts(
                    newColIndex,
                    colIndex,
                    newLength
            );
        }

        executor.release(dValues);
        executor.release(dColIndex);

        dValues = newValues;
        dColIndex = newColIndex;
    }

    @Override
    public void multiply(double[] x, double[] result) {
        throw new UnsupportedOperationException(
                "Use GPU DoubleVector operations instead"
        );
    }

    @Override
    public void transposeMultiply(double[] x, double[] result) {
        throw new UnsupportedOperationException(
                "Use GPU DoubleVector operations instead"
        );
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int cols() {
        return cols;
    }

    public int nnzPerRow() {
        return nnzPerRow;
    }

    private void checkRow(int row) {
        if (row < 0 || row >= rows)
            throw new IndexOutOfBoundsException("row: " + row);
    }

    public cl_mem valuesBuffer() {
        return dValues;
    }

    public cl_mem colIndexBuffer() {
        return dColIndex;
    }

    public double[] getValues() {
        return values;
    }

    public int[] getColIndex() {
        return colIndex;
    }

    public void close() {
        if (executor != null) {
            if (dValues != null) {
                executor.release(dValues);
                dValues = null;
            }

            if (dColIndex != null) {
                executor.release(dColIndex);
                dColIndex = null;
            }
        }
    }
}