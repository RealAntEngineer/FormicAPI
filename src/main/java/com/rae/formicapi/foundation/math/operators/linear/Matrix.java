package com.rae.formicapi.foundation.math.operators.linear;

import com.rae.formicapi.foundation.math.operators.DifferentiableOperator;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;

import java.util.Arrays;

/**
 * Represents a matrix used in simulations.
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
public interface Matrix extends DifferentiableOperator {

    @Override
    default void apply(RealVector x, RealVector result) {
        if (x instanceof CpuDoubleVector xCpu && result instanceof CpuDoubleVector resCpu) {
            multiply(xCpu.array(), resCpu.array());
            return;
        }

        throw new UnsupportedOperationException(
                "Matrix.apply(...) default implementation only supports CpuDoubleVector operands; got x="
                        + x.getClass().getSimpleName() + ", result=" + result.getClass().getSimpleName()
                        + ". A backend-specific Matrix implementation should override apply() directly instead of relying on this default.");
    }

    /**
     * When the operator is linear the Jacobian can be directly expressed by apply.
     */
    @Override
    default void multiplyJacobian(RealVector x, RealVector direction, RealVector result){
        apply(direction, result);//TODO is it x or direction ?
    }

    /**
     * Multiplies this matrix by vector {@code x}, storing Ax in {@code result}.
     *
     * @param x      input vector of length {@link #cols()}
     * @param result output vector of length {@link #rows()}, overwritten with Ax
     */
    @Deprecated
    void multiply(double[] x, double[] result);

    @Override
    default int inputSize() {
        return cols();
    }

    @Override
    default int outputSize() {
        return rows();
    }
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
    default void transposeApply(RealVector x, RealVector result) {
        if (x instanceof CpuDoubleVector xCpu && result instanceof CpuDoubleVector resCpu) {
            double[] result1 = resCpu.array();
            Arrays.fill(result1, 0.0);
            for (int r = 0; r < rows(); r++) {
                for (int c = 0; c < cols(); c++) {
                    result1[c] += get(r, c) * xCpu.array()[r];
                }
            }
            return;
        }

        throw new UnsupportedOperationException(
                "Matrix.transposeApply(...) default implementation only supports CpuDoubleVector operands; got x="
                        + x.getClass().getSimpleName() + ", result=" + result.getClass().getSimpleName()
                        + ". A backend-specific Matrix implementation should override transposeApply() directly instead of relying on this default.");
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

    RealVector getValues(IntegerVector r, IntegerVector c);
}