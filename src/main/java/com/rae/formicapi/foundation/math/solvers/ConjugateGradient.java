package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.Matrix;

/**
 * Conjugate Gradient solvers for symmetric positive-definite linear systems.
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

    /**
     * Solve {@code Ax = b} with a zero initial guess, allocating working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated if you will solve repeatedly.
     */
    public static double[] solve(Matrix A, double[] b, int maxIter, double tol) {
        int n = A.rows();
        return solve(A, new double[n], b, maxIter, tol,
                new double[n], new double[n], new double[n]);
    }

    /**
     * Solve {@code Ax = b} from an initial guess, allocating working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated if you will solve repeatedly.
     */
    public static double[] solve(Matrix A, double[] x_init, double[] b, int maxIter, double tol) {
        int n = A.rows();
        return solve(A, x_init, b, maxIter, tol,
                new double[n], new double[n], new double[n]);
    }

    private static double[] solve(Matrix A, double[] x, double[] b, int maxIter, double tol,
                                  double[] r, double[] p, double[] Ap) {
        return solve(A, x, b, maxIter, tol, r, p, Ap, false);
    }

    //TODO pattern iteration on fixed :
    // the X array can be fixed with a regular patter like [false, true, true, false, true, true,....] if that the case
    //   [false, true, true] is enough to define it and could be used to jump around in the array instead of checking on
    //   every value

    /**
     * Solve {@code A * x = b} using Conjugate Gradient, with a warm start and
     * caller-supplied working buffers.
     *
     * <p>{@code x} is used as both the initial guess and the output. Pass the
     * previous solution (for example {@code T_next}) to warm-start the iteration.
     * The solution is written in-place; no additional solution array is allocated.
     *
     * <p>All working arrays are overwritten during the solve. Their contents
     * between calls are undefined and must not be read by the caller.
     *
     * <p>This method requires {@code A} to be square, symmetric, and positive
     * definite:
     *
     * <pre>
     *     rows(A) == cols(A)
     * </pre>
     *
     * <p>The matrix structure and multiplication operation must represent the
     * complete unknown vector. For constrained systems where some variables are
     * fixed, use {@link #solveConstrained} instead.
     *
     * @param A
     *        SPD matrix representing the linear system. Must be square:
     *        {@code rows == cols}.
     *
     * @param x
     *        initial guess on entry and solution on exit. Length must be at least
     *        {@code n = A.rows()}.
     *
     * @param b
     *        right-hand side vector. Length must be at least
     *        {@code n = A.rows()}.
     *
     * @param maxIter
     *        maximum number of CG iterations before returning the current best
     *        estimate.
     *
     * @param tol
     *        convergence threshold on the residual norm:
     *
     *        <pre>
     *        ||r||₂ < tol
     *        </pre>
     *
     * @param r
     *        pre-allocated residual buffer. Length must be at least {@code n}.
     *
     * @param p
     *        pre-allocated CG search direction buffer. Length must be at least
     *        {@code n}.
     *
     * @param Ap
     *        pre-allocated matrix-vector product buffer storing {@code A * p}.
     *        Length must be at least {@code n}.
     *
     * @param allowBufferedEntries
     *        if true, {@code x}, {@code b}, and working arrays may contain trailing
     *        unused entries beyond the matrix dimensions. Only the first required
     *        entries are accessed; remaining values are ignored.
     *
     * @return {@code x} (the same array instance passed as input), containing the
     *         computed solution.
     *
     * @throws IllegalArgumentException
     *         if the matrix is not square or if provided buffers are too small.
     */
    public static double[] solve(Matrix A, double[] x, double[] b, int maxIter, double tol,
                                 double[] r, double[] p, double[] Ap, boolean allowBufferedEntries) {
        int n = A.rows();//number of equations
        int m = A.cols();//number of unknows
        if (n != m)
            throw new IllegalArgumentException(
                    "CG requires a square matrix: rows=" + n + ", cols=" + m);
        if ((x.length != m && !allowBufferedEntries) || x.length < n)
            throw new IllegalArgumentException(
                    "x length (" + x.length + ") != matrix size (" + m+ ")");
        if (b.length != n && !allowBufferedEntries || b.length < n)
            throw new IllegalArgumentException(
                    "b length (" + b.length + ") != matrix size (" + n + ")");
        if (r.length < n || p.length < n || Ap.length < n)
            throw new IllegalArgumentException(
                    "Working buffers r/p/Ap must have length >= " + n);

        // r = b - A*x
        A.multiply(x, Ap); // use Ap as temp for the initial residual
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - Ap[i];
            p[i] = r[i];
        }

        double rsold = dot(r, r, n);

        for (int k = 0; k < maxIter; k++) {
            A.multiply(p, Ap);

            double dotPAp = dot(p, Ap, n);
            if (dotPAp == 0) break; // already at solution or breakdown

            double alpha = rsold / dotPAp;

            for (int i = 0; i < n; i++) x[i] += alpha * p[i];
            for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];

            double rsnew = dot(r, r, n);
            if (Math.sqrt(rsnew) < tol) break;

            double beta = rsnew / rsold;
            for (int i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
            rsold = rsnew;
        }

        return x;
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
     * matrix-vector product:
     *
     * <pre>
     *     r = b - A*x
     * </pre>
     *
     * <p>After initialization, the CG search direction only contains free-variable
     * components. Entries of {@code p} corresponding to fixed variables are kept
     * at zero because fixed variables have no search direction:
     *
     * <pre>
     *     p[fixedIndex] = 0
     * </pre>
     *
     * <p>This allows the regular {@link Matrix#multiply(double[], double[])}
     * operation to be used directly. The multiplication of {@code A*p} naturally
     * ignores fixed variables because their direction is zero.
     *
     * <p>The internally solved system is equivalent to the reduced free-variable
     * system:
     *
     * <pre>
     *     A_free * x_free = b_adjusted
     * </pre>
     *
     * <p>where the effect of fixed variables is already included in the residual.
     * No reduced matrix or temporary vectors are allocated.
     *
     * <p>{@code x} is used as both the initial guess and the output. Pass the
     * previous solution to obtain a warm start. The array is modified in-place;
     * no solution array is allocated.
     *
     * <p>All working arrays are overwritten during the solve. Their contents
     * between calls are undefined and must not be reused.
     *
     * <p>The matrix itself does not need to be square. However, the number of free
     * variables must equal the number of equations, and the reduced system seen by
     * CG must be symmetric positive definite.
     *
     * @param A                    system matrix with {@code n} rows (equations) and
     *                             {@code m} columns (variables). Matrix multiplication
     *                             must accept the full variable vector.
     * @param x                    full variable vector. Length must be at least
     *                             {@code m}. Free variables are updated in-place;
     *                             fixed variables remain unchanged.
     * @param fixedVariables       boolean mask defining constrained variables.
     *
     *                             <pre>
     *                             fixedVariables[i] == true
     *                             </pre>
     *
     *                             means {@code x[i]} is prescribed and excluded
     *                             from the CG iteration.
     *
     *                             <p>The number of {@code false} entries must
     *                             equal the number of equations ({@code n}).
     * @param b                    right-hand side vector. Length must be at least
     *                             {@code n}.
     * @param maxIter              maximum number of CG iterations.
     * @param tol                  convergence threshold on the residual norm
     *                             {@code ||r||₂}.
     * @param allowBufferedEntries if {@code true}, {@code x}, {@code b}, and
     *                             {@code fixedVariables} may contain unused
     *                             trailing entries beyond the required size.
     *                             These entries are ignored.
     * @param r                    pre-allocated residual buffer in equation space.
     *                             Length must be at least {@code n}.
     * @param p                    pre-allocated CG direction buffer in variable
     *                             space. Length must be at least {@code m}.
     *
     *                             <p>Free-variable entries contain the CG search
     *                             direction. Fixed-variable entries remain zero.
     * @param Ap                   pre-allocated matrix-vector product buffer.
     *                             Stores {@code A*p}. Length must be at least
     *                             {@code n}.
     * @param unknownIdx           pre-allocated mapping buffer. After initialization:
     *
     *                             <pre>
     *                             unknownIdx[equationIndex] = variableIndex
     *                             </pre>
     *
     *                             maps each reduced equation entry to the
     *                             corresponding free variable in the full vectors.
     *                             Length must be at least {@code n}.
     *
     * @return {@code x} (same instance as input), containing the solved variables.
     *
     * @throws IllegalArgumentException if supplied buffers are too small or
     *         incompatible with matrix dimensions.
     * @throws IllegalStateException if the number of free variables does not equal
     *         the number of equations.
     */
    public static double[] solveConstrained(Matrix A, double[] x, boolean[] fixedVariables, double[] b, int maxIter, double tol,
                                            boolean allowBufferedEntries, double[] r, double[] p, double[] Ap, int[] unknownIdx) {
        int n = A.rows();//number of equations
        int m = A.cols();//number of unknows

        if ((x.length != m && !allowBufferedEntries) || x.length < m)
            throw new IllegalArgumentException(
                    "x length (" + x.length + ") != matrix columns (" + m+ ")");
        if (b.length != n && !allowBufferedEntries || b.length < n)
            throw new IllegalArgumentException(
                    "b length (" + b.length + ") != matrix rows (" + n + ")");
        if (r.length < n || p.length < m || Ap.length < n || unknownIdx.length < n)
            throw new IllegalArgumentException(
                    "Working buffers Ap/r/unknownIdx must have length >= " + n + "and p must have length >=" + m);

        // r = b - A*x
        A.multiply(x, Ap); // use Ap as temp for the initial residual

        int idx = 0;
        for (int i = 0; i < m; i++) {
            if (!fixedVariables[i]) {
                if (idx < n) {
                    r[idx] = b[idx] - Ap[idx];
                    p[i] = r[idx];
                    unknownIdx[idx] = i;
                    idx += 1;
                }
            } else {
                p[i] = 0;//the value is constrained so we don't search in that direction
            }
        }

        if (idx != n){//valid because we add + 1 at the end
            throw new IllegalStateException(
                    "CG requires number of free variables == equations, got "+ idx);
        }

        double rsold = dot(r, r, n);

        for (int k = 0; k < maxIter; k++) {
            A.multiply(p, Ap);//the ei

            double dotPAp = skippedDot(p, Ap, unknownIdx,n);
            if (dotPAp == 0) break; // already at solution or breakdown

            double alpha = rsold / dotPAp;

            for (int i = 0; i < n; i++) {
                x[unknownIdx[i]] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double rsnew = dot(r, r, n);
            if (Math.sqrt(rsnew) < tol) break;

            double beta = rsnew / rsold;
            for (int i = 0; i < n; i++){
                p[unknownIdx[i]] = r[i] + beta * p[unknownIdx[i]];
            }
            rsold = rsnew;
        }

        return x;
    }

    private static double dot(double[] a, double[] b, int n) {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    private static double skippedDot(double[] a, double[] b, int[] unknowIdx, int n){
        double sum = 0;
        for (int i = 0; i < n; i++){
            sum += a[unknowIdx[i]] * b[i];
        }
        return sum;
    }
}