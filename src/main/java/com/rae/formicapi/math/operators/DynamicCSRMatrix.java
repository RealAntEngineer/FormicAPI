package com.rae.formicapi.math.operators;

/**
 * A sparse matrix that supports dynamic assembly via a hash-based buffer,
 * and compiles to CSR format for efficient multiplication.
 *
 * <p>During assembly, entries are accumulated in a {@link HashSparseMatrix}.
 * Before any read or multiply operation, the matrix is compiled into CSR
 * format. Subsequent mutations invalidate the CSR cache and trigger a
 * recompile on next access.
 *
 * <p>This allows the same matrix instance to be used for both stamping
 * (assembly phase) and solving (multiply phase) without requiring an
 * explicit conversion step.
 *
 * @see MutableMatrix
 * @see HashSparseMatrix
 */
public class DynamicCSRMatrix implements MutableMatrix {

    private final HashSparseMatrix buffer;
    private CSRMatrix csr = null;

    /**
     * Creates a dynamic CSR matrix of the given dimensions.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public DynamicCSRMatrix(int rows, int cols) {
        this.buffer = new HashSparseMatrix(rows, cols);
    }

    // ------------------------------------------------
    // MutableMatrix — invalidate CSR on write
    // ------------------------------------------------

    /**
     * Adds a value to the entry at (row, col), accumulating with any existing value.
     *
     * @param row   row index
     * @param col   column index
     * @param value value to add
     */
    @Override
    public void add(int row, int col, double value) {
        csr = null;
        buffer.add(row, col, value);
    }

    /**
     * Sets the entry at (row, col) to the given value.
     *
     * @param row   row index
     * @param col   column index
     * @param value value to set
     */
    @Override
    public void set(int row, int col, double value) {
        csr = null;
        buffer.set(row, col, value);
    }

    // ------------------------------------------------
    // Matrix reads — compile CSR lazily
    // ------------------------------------------------

    @Override
    public int rows() {
        return buffer.rows();
    }

    @Override
    public int cols() {
        return buffer.cols();
    }

    /**
     * Returns the value at (r, c). Triggers CSR compilation if needed.
     *
     * @param r row index
     * @param c column index
     * @return the scalar value at that position
     */
    @Override
    public double get(int r, int c) {
        return compiled().get(r, c);
    }

    /**
     * Multiplies this matrix by vector {@code x}, storing the result in {@code result}.
     * Triggers CSR compilation if needed.
     *
     * @param x      input vector of length cols
     * @param result output vector of length rows
     */
    @Override
    public void multiply(double[] x, double[] result) {
        compiled().multiply(x, result);
    }

    @Override
    public void transposeMultiply(double[] x, double[] result) {
        compiled().transposeMultiply(x, result);
    }

    // ------------------------------------------------
    // Internal — lazy CSR compilation
    // ------------------------------------------------

    private CSRMatrix compiled() {
        if (csr == null) csr = buffer.toCSR();
        return csr;
    }

    public CSRMatrix getCash() {
        return compiled();
    }
}