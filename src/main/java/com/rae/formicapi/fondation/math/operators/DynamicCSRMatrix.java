package com.rae.formicapi.fondation.math.operators;

/**
 * A sparse matrix that supports dynamic assembly via a hash-based buffer,
 * and compiles to CSR format for efficient multiplication.
 *
 * <p>During assembly, entries are accumulated in a {@link HashSparseMatrix}.
 * Before any read or multiply operation, the matrix is compiled into CSR
 * format. Subsequent mutations invalidate the CSR cache and trigger a
 * recompile on next access.
 *
 * <p>Zero values passed to {@link #add} or {@link #set} are ignored or
 * removed, preserving sparsity and avoiding unnecessary CSR recompilation.
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
     * Adds {@code value} to the entry at (row, col), accumulating with any existing value.
     *
     * <p>If {@code value} is zero, this call is a no-op and the CSR cache
     * is not invalidated.
     *
     * @param row   row index
     * @param col   column index
     * @param value value to accumulate
     */
    @Override
    public void add(int row, int col, double value) {
        if (value == 0.0) return;
        csr = null;
        buffer.add(row, col, value);
    }

    /**
     * Sets the entry at (row, col) to {@code value}.
     *
     * <p>If the current value at (row, col) is already equal to {@code value},
     * this call is a no-op and the CSR cache is not invalidated.
     *
     * <p>If {@code value} is zero, the entry is removed from the buffer
     * rather than stored, preserving sparsity.
     *
     * @param row   row index
     * @param col   column index
     * @param value value to set
     */
    @Override
    public void set(int row, int col, double value) {
        if (buffer.get(row, col) == value) return;
        csr = null;
        buffer.set(row, col, value);
    }

    // ------------------------------------------------
    // Matrix reads — compile CSR lazily
    // ------------------------------------------------

    /**
     * Multiplies this matrix by vector {@code x}, storing Ax in {@code result}.
     *
     * <p>Triggers CSR compilation if the cache is invalid.
     *
     * @param x      input vector of length {@link #cols()}
     * @param result output vector of length {@link #rows()}, overwritten with Ax
     */
    @Override
    public void multiply(double[] x, double[] result) {
        compiled().multiply(x, result);
    }

    /**
     * Multiplies the transpose of this matrix by vector {@code x},
     * storing Aᵀx in {@code result}.
     *
     * <p>Triggers CSR compilation if the cache is invalid.
     *
     * @param x      input vector of length {@link #rows()}
     * @param result output vector of length {@link #cols()}, overwritten with Aᵀx
     */
    @Override
    public void transposeMultiply(double[] x, double[] result) {
        compiled().transposeMultiply(x, result);
    }

    @Override
    public int rows() {
        return buffer.rows();
    }

    @Override
    public int cols() {
        return buffer.cols();
    }

    /**
     * Returns the value at (r, c).
     *
     * <p>Reads directly from the hash buffer rather than the compiled CSR,
     * so it never triggers a recompile. This is intentional — element reads
     * during assembly should not pay the full CSR compilation cost.
     *
     * <p>If no value has been set at (r, c), returns {@code 0.0}.
     *
     * @param r row index
     * @param c column index
     * @return the scalar value at (r, c), or {@code 0.0} if not stored
     */
    @Override
    public double get(int r, int c) {
        return buffer.get(r, c);
    }

    // ------------------------------------------------
    // Internal — lazy CSR compilation
    // ------------------------------------------------

    private CSRMatrix compiled() {
        if (csr == null) csr = buffer.toCSR();
        return csr;
    }

    /**
     * Returns the compiled {@link CSRMatrix}, triggering compilation if needed.
     *
     * <p>Intended for use by benchmarks and tests that need direct CSR access.
     *
     * @return the compiled CSR representation of this matrix
     */
    public CSRMatrix toCSR() {
        return compiled();
    }
}