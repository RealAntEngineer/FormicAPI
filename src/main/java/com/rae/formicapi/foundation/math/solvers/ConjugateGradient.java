package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuIntegerVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;

/**
 * Conjugate Gradient solvers for symmetric positive-definite linear systems.
 *
 * <p>Ports the original array-based implementation onto the {@link DoubleVector}
 * / {@link IntegerVector} backend abstraction (see {@link LeastSquare} for the
 * same pattern), so the solve can run on whichever backend the caller's vectors
 * belong to instead of being hard-wired to {@code double[]}.
 *
 * <p>The standard solver handles a regular SPD system:
 *
 * <pre>
 *     A x = b
 * </pre>
 *
 * where {@code A} is square, symmetric, and positive definite.
 *
 * <p>The constrained solver extends this by supporting fixed variables in
 * {@code x}. Fixed values remain stored directly in the solution vector and
 * are injected into the matrix-vector product through the working direction
 * buffer. The free variables form the actual iterative system:
 *
 * <pre>
 *     A_free x_free = b_free
 * </pre>
 *
 * <p>This allows solving systems where the physical matrix storage still
 * contains boundary/fixed variables without allocating a reduced matrix.
 *
 * <p>Both solvers use caller-provided working buffers to avoid allocations
 * during iteration. Buffers are overwritten on every call and their previous
 * contents are undefined.
 *
 * <p>Plain Conjugate Gradient works directly on {@code A} and converges in at
 * most {@code n} iterations for an exact SPD system.
 */
@SuppressWarnings("unused")
public class ConjugateGradient {

    // indices into the n-sized DoubleVector WorkingBuffer used by solve()
    private static final int R  = 0;
    private static final int P  = 1;
    private static final int AP = 2;

    // indices into the constrained solver's buffers (used by solvedConstrained())
    private static final int CR  = 0; // equation space (size n)
    private static final int CAP = 1; // equation space (size n)
    private static final int CP  = 0; // variable space (size m)

    /**
     * Solve {@code Ax = b} with a zero initial guess, allocating working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static double[] solve(Matrix A, double[] b, int maxIter, double tol) {
        int n = A.rows();
        return solve(A, new double[n], b, maxIter, tol);
    }

    /**
     * Solve {@code Ax = b} from an initial guess, allocating working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static double[] solve(Matrix A, double[] x_init, double[] b, int maxIter, double tol) {
        DoubleVector xv = new CpuDoubleVector(x_init);
        solve(A, xv, new CpuDoubleVector(b), maxIter, tol);
        return ((CpuDoubleVector) xv).array();
    }

    /**
     * Solve {@code Ax = b}, allocating its working buffer internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static int solve(Matrix A, DoubleVector x, DoubleVector b, int maxIter, double tol) {
        int n = A.rows();
        return solve(A, x, b, maxIter, tol, new WorkingBuffer<>(3, () -> new CpuDoubleVector(n)));
    }

    //TODO pattern iteration on fixed :
    // the X vector can be fixed with a regular pattern like [false, true, true, false, true, true,....] if that's
    //   the case [false, true, true] is enough to define it and could be used to jump around instead of checking
    //   every value

    /**
     * Solve {@code A * x = b} using Conjugate Gradient, with a warm start and
     * a caller-supplied working buffer.
     *
     * <p>{@code x} is used as both the initial guess and the output. Pass the
     * previous solution (for example {@code T_next}) to warm-start the iteration.
     * The solution is written in-place; no additional solution vector is allocated.
     *
     * <p>The working buffer is overwritten during the solve. Its contents
     * between calls are undefined and must not be read by the caller.
     *
     * <p>This method requires {@code A} to be square, symmetric, and positive
     * definite: {@code rows(A) == cols(A)}.
     *
     * <p>The matrix structure and multiplication operation must represent the
     * complete unknown vector. For constrained systems where some variables are
     * fixed, use {@link #solveConstrained} instead.
     *
     * @param A       SPD matrix representing the linear system. Must be square.
     * @param x       initial guess on entry and solution on exit.
     * @param b       right-hand side vector. Size must equal {@code n = A.rows()}.
     * @param maxIter maximum number of CG iterations before returning the current best estimate.
     * @param tol     convergence threshold on the residual norm: {@code ||r||₂ < tol}
     * @param buffer  working buffer providing 3 vectors of size {@code n}: residual {@code r},
     *                search direction {@code p}, and matrix-vector product {@code Ap}.
     * @return the number of iterations performed (capped at {@code maxIter}).
     * @throws IllegalArgumentException if the matrix is not square or {@code b} has the wrong size.
     */
    public static int solve(Matrix A, DoubleVector x, DoubleVector b, int maxIter, double tol,
                            WorkingBuffer<DoubleVector> buffer) {
        int n = A.rows(); // number of equations
        int m = A.cols(); // number of unknowns
        if (n != m)
            throw new IllegalArgumentException(
                    "CG requires a square matrix: rows=" + n + ", cols=" + m);
        if (b.size() != n)
            throw new IllegalArgumentException(
                    "b size (" + b.size() + ") != matrix size (" + n + ")");

        x.resize(n);

        DoubleVector r  = buffer.get(R);
        DoubleVector p  = buffer.get(P);
        DoubleVector Ap = buffer.get(AP);

        r.resize(n);
        p.resize(n);
        Ap.resize(n);

        // r = b - A*x
        A.apply(x, Ap); // use Ap as temp for the initial residual
        r.copy(b);
        r.axpy(-1.0, Ap);
        p.copy(r);

        double rsold = r.dot(r);

        for (int k = 0; k < maxIter; k++) {
            A.apply(p, Ap);

            double dotPAp = p.dot(Ap);
            if (dotPAp == 0) return k; // already at solution or breakdown

            double alpha = rsold / dotPAp;

            x.axpy(alpha, p);
            r.axpy(-alpha, Ap);

            double rsnew = r.dot(r);
            if (Math.sqrt(rsnew) < tol) return k + 1;

            double beta = rsnew / rsold;
            // p = r + beta*p
            p.scale(beta);
            p.add(r);
            rsold = rsnew;
        }
        return maxIter;
    }

    /**
     * Solve a constrained system {@code A * x = b} with a zero-length free-variable
     * warm start, allocating working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static double[] solveConstrained(Matrix A, double[] x, boolean[] fixedVariables, double[] b,
                                            int maxIter, double tol) {
        int n = A.rows();
        DoubleVector xv = new CpuDoubleVector(x);
        DoubleVector bv = new CpuDoubleVector(b);
        solveConstrained(A, xv, fixedVariables, bv, maxIter, tol, new CpuIntegerVector(n));
        return ((CpuDoubleVector) xv).array();
    }

    /**
     * Solve a constrained system, allocating its working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static int solveConstrained(Matrix A, DoubleVector x, boolean[] fixedVariables, DoubleVector b,
                                       int maxIter, double tol, IntegerVector unknownIdx) {
        int n = A.rows();
        int m = A.cols();
        return solveConstrained(A, x, fixedVariables, b, maxIter, tol, unknownIdx,
                new WorkingBuffer<>(2, () -> new CpuDoubleVector(n)),
                new WorkingBuffer<>(1, () -> new CpuDoubleVector(m)));
    }

    /**
     * Solve a constrained linear system using Conjugate Gradient:
     *
     * <pre>
     *     A * x = b
     * </pre>
     *
     * <p>{@code x} contains both free and fixed variables. Variables marked in
     * {@code fixedVariables} are prescribed values and are never modified during
     * the solve. Only free variables participate in the CG iterations.
     *
     * <p>The solver avoids constructing an explicit reduced matrix. Instead, it
     * keeps all vectors in their natural spaces:
     *
     * <ul>
     *     <li>{@code x}, {@code p}: full variable space (matrix columns)</li>
     *     <li>{@code r}, {@code Ap}: equation space (matrix rows)</li>
     * </ul>
     *
     * <p>Fixed variables contribute to the initial residual through the full
     * matrix-vector product: {@code r = b - A*x}.
     *
     * <p>After initialization, the CG search direction only contains free-variable
     * components. Entries of {@code p} corresponding to fixed variables are kept
     * at zero because fixed variables have no search direction: {@code p[fixedIndex] = 0}.
     *
     * <p>This allows the regular {@link Matrix#apply(DoubleVector, DoubleVector)}
     * operation to be used directly. The multiplication of {@code A*p} naturally
     * ignores fixed variables because their direction is zero.
     *
     * <p>The internally solved system is equivalent to the reduced free-variable
     * system {@code A_free * x_free = b_adjusted}, where the effect of fixed
     * variables is already included in the residual. No reduced matrix is allocated.
     *
     * <p><b>Backend-agnostic by construction:</b> {@code p} is kept zero at every
     * fixed index for the whole solve (cleared once, then only ever written through
     * {@code scatterAxpy} at free indices). That invariant means the {@code x}
     * update can run as a plain full-vector {@code x.axpy(alpha, p)} — fixed
     * entries just add zero — instead of needing an indexed scatter of its own.
     * The only genuinely indexed operation left is placing the compact,
     * equation-space residual {@code r} into the free positions of the
     * variable-space direction {@code p}; that's exactly what
     * {@link DoubleVector#skippedAxpy} is for, the scatter counterpart of
     * {@link DoubleVector#skippedDot}. No vector's backing array is ever touched
     * directly, so this runs on whatever backend {@code x}/{@code b}/the buffers
     * belong to.
     *
     * <p>{@code x} is used as both the initial guess and the output. Pass the
     * previous solution to obtain a warm start. It is modified in-place; no
     * solution vector is allocated.
     *
     * <p>All working buffers are overwritten during the solve. Their contents
     * between calls are undefined and must not be reused.
     *
     * <p>The matrix itself does not need to be square. However, the number of free
     * variables must equal the number of equations, and the reduced system seen by
     * CG must be symmetric positive definite.
     *
     * @param A               system matrix with {@code n} rows (equations) and {@code m}
     *                        columns (variables). {@code A.apply} must accept the full
     *                        variable vector.
     * @param x               full variable vector, size {@code m}. Free variables are
     *                        updated in-place; fixed variables remain unchanged.
     * @param fixedVariables  boolean mask defining constrained variables:
     *                        {@code fixedVariables[i] == true} means {@code x[i]} is
     *                        prescribed and excluded from the CG iteration. The number
     *                        of {@code false} entries must equal the number of equations
     *                        ({@code n}).
     * @param b               right-hand side vector, size {@code n}.
     * @param maxIter         maximum number of CG iterations.
     * @param tol             convergence threshold on the residual norm {@code ||r||₂}.
     * @param unknownIdx      working index buffer, size {@code n}. After initialization,
     *                        {@code unknownIdx[equationIndex] = variableIndex}, mapping each
     *                        reduced equation entry to the corresponding free variable.
     * @param nBuffer         working buffer providing 2 vectors of size {@code n}:
     *                        residual {@code r} and matrix-vector product {@code Ap}.
     * @param mBuffer         working buffer providing 1 vector of size {@code m}:
     *                        the CG search direction {@code p} (full variable space,
     *                        zero at fixed indices).
     * @return the number of iterations performed (capped at {@code maxIter}).
     * @throws IllegalStateException if the number of free variables does not equal
     *         the number of equations.
     */
    public static int solveConstrained(Matrix A, DoubleVector x, boolean[] fixedVariables, DoubleVector b,
                                       int maxIter, double tol, IntegerVector unknownIdx,
                                       WorkingBuffer<DoubleVector> nBuffer, WorkingBuffer<DoubleVector> mBuffer) {
        int n = A.rows(); // number of equations
        int m = A.cols(); // number of unknowns

        x.resize(m);
        unknownIdx.resize(n);

        DoubleVector r  = nBuffer.get(CR);
        DoubleVector Ap = nBuffer.get(CAP);
        DoubleVector p  = mBuffer.get(CP);

        r.resize(n);
        Ap.resize(n);
        p.resize(m);

        // Build the free-variable index map: unknownIdx[equationIndex] = variableIndex.
        // fixedVariables is a plain boolean mask, not a Vector, so this stays a normal loop.
        int idx = 0;
        for (int i = 0; i < m; i++) {
            if (!fixedVariables[i]) {
                unknownIdx.set(i, idx); // set(value, idx) -> unknownIdx[idx] = i
                idx += 1;
            }
        }
        if (idx != n) {
            throw new IllegalStateException(
                    "CG requires number of free variables == equations, got " + idx);
        }

        // r = b - A*x : a plain full-vector op in equation space. Fixed variables'
        // contribution is already baked in since A.apply used the full x.
        A.apply(x, Ap); // use Ap as temp for the initial full residual
        r.copy(b);
        r.axpy(-1.0, Ap);

        // p = scatter(r) into variable space; stays zero at every fixed index.
        p.clear();
        p.skippedAxpy(1.0, r, unknownIdx, true, false);

        double rsold = r.dot(r);

        for (int k = 0; k < maxIter; k++) {
            A.apply(p, Ap);

            double dotPAp = p.skippedDot(Ap, unknownIdx, true, false);
            if (dotPAp == 0) return k; // already at solution or breakdown

            double alpha = rsold / dotPAp;

            // Safe as a full-vector axpy: p is zero at fixed indices, so those
            // entries of x are left unchanged.
            x.axpy(alpha, p);
            r.axpy(-alpha, Ap);

            double rsnew = r.dot(r);
            if (Math.sqrt(rsnew) < tol) return k + 1;

            double beta = rsnew / rsold;
            // p[unknownIdx[i]] = r[i] + beta*p[unknownIdx[i]], for all free i;
            // fixed indices stay at beta*0 = 0.
            p.scale(beta);
            p.skippedAxpy(1.0, r, unknownIdx, false, true);
            rsold = rsnew;
        }
        return maxIter;
    }
}