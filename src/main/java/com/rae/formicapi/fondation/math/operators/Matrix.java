package com.rae.formicapi.fondation.math.operators;

import java.util.Arrays;

/**
 * Represents a matrix used in nodal network simulations.
 *
 * <p>Implementations may be dense or sparse, mutable or immutable.
 * The interface exposes the minimal operations required by iterative
 * solvers: element access, matrix-vector multiply, and transpose
 * matrix-vector multiply.
 *
 * <p>Mutable variants should implement {@link MutableMatrix}, which
 * extends this interface with assembly operations.
 *
 * @see MutableMatrix
 * @see CSRMatrix
 * @see HashSparseMatrix
 */
public interface Matrix {

    /**
     * Multiplies this matrix by vector {@code x}, storing Ax in {@code result}.
     *
     * @param x      input vector of length {@link #cols()}
     * @param result output vector of length {@link #rows()}, overwritten with Ax
     */
    void multiply(double[] x, double[] result);

    /**
     * Multiplies the transpose of this matrix by vector {@code x},
     * storing Aᵀx in {@code result}.
     *
     * <p>The default implementation iterates over all (r, c) pairs via
     * {@link #get(int, int)} and is correct but O(rows × cols). Sparse
     * implementations should override this to achieve O(nnz) performance
     * by iterating directly over stored non-zeros.
     *
     * @param x      input vector of length {@link #rows()}
     * @param result output vector of length {@link #cols()}, overwritten with Aᵀx
     */
    default void transposeMultiply(double[] x, double[] result) {
        Arrays.fill(result, 0.0);
        for (int r = 0; r < rows(); r++) {
            for (int c = 0; c < cols(); c++) {
                result[c] += get(r, c) * x[r];
            }
        }
    }

    /**
     * Returns the number of rows in this matrix.
     *
     * @return row count
     */
    int rows();

    /**
     * Returns the number of columns in this matrix.
     *
     * @return column count
     */
    int cols();

    /**
     * Returns the value at position (r, c).
     *
     * <p>For sparse implementations, returns {@code 0.0} for entries
     * not explicitly stored.
     *
     * @param r row index
     * @param c column index
     * @return the scalar value at (r, c)
     */
    double get(int r, int c);
}