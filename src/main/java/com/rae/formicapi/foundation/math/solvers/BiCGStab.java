package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.Matrix;

/**
 * Biconjugate Gradient Stabilized (BiCGSTAB) solver for general (non-symmetric)
 * systems {@code Ax = b}.
 *
 * <p>Prefer this over {@link ConjugateGradient} when {@code A} is not known to be
 * symmetric positive definite. BiCGSTAB works directly on {@code A} and, unlike
 * plain BiCG, has smoothed, more stable convergence at roughly the same cost per
 * iteration (two matrix-vector products instead of one).
 *
 * <p>BiCGSTAB can still break down (division by ~0) on pathological matrices;
 * callers that need a guaranteed-convergent fallback should be prepared to
 * catch a stall (residual stops decreasing) and switch solvers.
 */
@SuppressWarnings("unused")
public class BiCGStab {

    /**
     * Solve {@code Ax = b} with a zero initial guess, allocating working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static double[] solve(Matrix A, double[] b, int maxIter, double tol) {
        int n = A.rows();
        return solve(A, new double[n], b, maxIter, tol,
                new double[n], new double[n], new double[n],
                new double[n], new double[n], new double[n]);
    }

    /**
     * Solve {@code Ax = b} from an initial guess, allocating working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static double[] solve(Matrix A, double[] x_init, double[] b, int maxIter, double tol) {
        int n = A.rows();
        return solve(A, x_init, b, maxIter, tol,
                new double[n], new double[n], new double[n],
                new double[n], new double[n], new double[n]);
    }

    /**
     * Solve {@code Ax = b} using BiCGSTAB, with a warm start and caller-supplied
     * working buffers.
     *
     * <p>{@code x} is used as both the initial guess and the output — pass
     * {@code x_next} from the previous tick for warm-starting. The array is
     * overwritten in-place; no solution array is allocated.
     *
     * <p>All working arrays are overwritten on every call. Their values between
     * calls are undefined and must not be read by the caller.
     *
     * <p>Requires {@code A} to be square. Unlike {@link ConjugateGradient}, {@code A}
     * need not be symmetric or positive definite.
     *
     * @param A       matrix, must be square ({@code rows == cols})
     * @param x       initial guess on entry, solution on exit — length {@code n}
     * @param b       right-hand side — length {@code n}
     * @param maxIter maximum iterations before returning current best estimate
     * @param tol     convergence threshold on {@code ||r||₂}
     * @param r       pre-allocated residual buffer      — length ≥ {@code n}
     * @param rHat0   pre-allocated shadow residual buffer (fixed for the whole solve) — length ≥ {@code n}
     * @param p       pre-allocated search direction buffer — length ≥ {@code n}
     * @param v       pre-allocated {@code A·p} buffer     — length ≥ {@code n}
     * @param s       pre-allocated stabilizer buffer      — length ≥ {@code n}
     * @param t       pre-allocated {@code A·s} buffer     — length ≥ {@code n}
     * @return {@code x} (same array as input, for chaining)
     * @throws IllegalArgumentException if matrix is not square or buffer sizes mismatch
     */
    public static double[] solve(Matrix A, double[] x, double[] b, int maxIter, double tol,
                                 double[] r, double[] rHat0, double[] p, double[] v,
                                 double[] s, double[] t) {
        int n = A.rows();
        int m = A.cols();

        if (n != m)
            throw new IllegalArgumentException(
                    "BiCGSTAB requires a square matrix: rows=" + n + ", cols=" + m);
        if (x.length != n)
            throw new IllegalArgumentException(
                    "x length (" + x.length + ") != matrix size (" + n + ")");
        if (b.length != n)
            throw new IllegalArgumentException(
                    "b length (" + b.length + ") != matrix size (" + n + ")");
        if (r.length < n || rHat0.length < n || p.length < n
                || v.length < n || s.length < n || t.length < n)
            throw new IllegalArgumentException(
                    "Working buffers r/rHat0/p/v/s/t must have length >= " + n);

        // r = b - A*x ; rHat0 = r (fixed shadow residual, arbitrary choice)
        A.multiply(x, v); // use v as temp for the initial residual
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - v[i];
            rHat0[i] = r[i];
            p[i] = 0;
            v[i] = 0;
        }

        double rho = 1, alpha = 1, omega = 1;

        if (Math.sqrt(dot(r, r, n)) < tol) return x;

        for (int k = 0; k < maxIter; k++) {
            double rhoNew = dot(rHat0, r, n);
            if (rhoNew == 0) break; // breakdown: rHat0 orthogonal to r

            double beta = (rhoNew / rho) * (alpha / omega);
            for (int i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * v[i]);

            A.multiply(p, v);

            double rHat0v = dot(rHat0, v, n);
            if (rHat0v == 0) break; // breakdown
            alpha = rhoNew / rHat0v;

            for (int i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];

            double sNorm = Math.sqrt(dot(s, s, n));
            if (sNorm < tol) {
                for (int i = 0; i < n; i++) x[i] += alpha * p[i];
                break;
            }

            A.multiply(s, t);

            double tDotT = dot(t, t, n);
            omega = (tDotT == 0) ? 0 : dot(t, s, n) / tDotT;

            for (int i = 0; i < n; i++) x[i] += alpha * p[i] + omega * s[i];
            for (int i = 0; i < n; i++) r[i] = s[i] - omega * t[i];

            if (Math.sqrt(dot(r, r, n)) < tol) break;
            if (omega == 0) break; // breakdown

            rho = rhoNew;
        }

        return x;
    }

    /**
     * Solve a constrained linear system {@code Ax = b} using BiCGSTAB while keeping
     * fixed variables embedded in the original solution vector.
     *
     * <p>The solver operates directly on the full matrix without constructing an
     * explicit reduced system. Variables marked in {@code fixedVariables} are
     * treated as prescribed values and are never modified. Only free variables are
     * updated by the BiCGSTAB iterations.
     *
     * <p>The solution vector {@code x} remains in the original variable space.
     * Fixed entries contain their prescribed values and are kept unchanged.
     * Free entries are the actual unknowns solved by BiCGSTAB.
     *
     * <p>The search vectors {@code p} and {@code s} are stored in the full variable
     * space (one entry per matrix column). Entries corresponding to fixed variables
     * are always zero because fixed variables have no degrees of freedom. Entries
     * corresponding to free variables contain the BiCGSTAB search direction and
     * stabilizer vectors. Because fixed entries are zero, the regular
     * {@link Matrix#multiply(double[], double[])} operation can be used directly
     * without constructing a reduced matrix.
     *
     * <p>The vectors {@code r}, {@code rHat0}, {@code v}, and {@code t} live in
     * equation space (one entry per matrix row). The {@code unknownIdx} array maps
     * each equation index to the corresponding free variable index in the full
     * variable space.
     *
     * <p>The initial residual includes the contribution of the fixed variables:
     *
     * <pre>
     * r = b - A*x
     * </pre>
     *
     * <p>After initialization, BiCGSTAB only modifies the free variables. Fixed
     * variables are never restored or projected because they are never changed.
     *
     * <p>This method requires the number of free variables to equal the number of
     * equations. Unlike {@link ConjugateGradient}, the matrix does not need to be
     * symmetric or positive definite.
     *
     * @param A matrix representing the constrained system
     * @param x initial guess on entry, solution on exit; fixed entries remain unchanged
     * @param fixedVariables boolean mask indicating fixed variables. A {@code true}
     *        entry means that the corresponding entry of {@code x} is prescribed
     *        and excluded from the solve
     * @param b right-hand side vector; one entry per equation
     * @param maxIter maximum number of BiCGSTAB iterations
     * @param tol convergence threshold on the residual norm {@code ||r||₂}
     * @param allowBufferedEntries if {@code true}, input and working arrays may be
     *        larger than required; excess entries are ignored
     * @param r residual buffer in equation space; length ≥ number of equations
     * @param rHat0 fixed shadow residual buffer used by BiCGSTAB; length ≥ number
     *        of equations
     * @param p full variable-space search direction buffer; length ≥ number of
     *        variables. Fixed-variable entries remain zero. Free-variable entries
     *        contain the current BiCGSTAB direction
     * @param v equation-space buffer storing {@code A*p}; length ≥ number of
     *        equations
     * @param s full variable-space stabilizer buffer; length ≥ number of variables.
     *        Fixed-variable entries remain zero. Free-variable entries contain the
     *        current BiCGSTAB stabilizer vector
     * @param t equation-space buffer storing {@code A*s}; length ≥ number of
     *        equations
     * @param unknownIdx mapping from equation index to the corresponding free
     *        variable index in {@code x}, {@code p}, and {@code s}; length ≥ number
     *        of equations
     *
     * @return {@code x} (the same array passed as input)
     *
     * @throws IllegalArgumentException if supplied buffers are too small or
     *         incompatible with matrix dimensions
     * @throws IllegalStateException if the number of free variables does not equal
     *         the number of equations
     */
    public static double[] solveConstrained(Matrix A, double[] x, boolean[] fixedVariables, double[] b, int maxIter, double tol,
                                            boolean allowBufferedEntries, double[] r, double[] rHat0, double[] p, double[] v,
                                            double[] s, double[] t, int[] unknownIdx) {
        int n = A.rows();
        int m = A.cols();


        if (n > m)
            throw new IllegalArgumentException(
                    "BiCGSTAB constrained requires an under constrained matrix: rows=" + n + ", cols=" + m);

        if ((x.length != m && !allowBufferedEntries) || x.length < m)
            throw new IllegalArgumentException( "x length (" + x.length + ") != matrix columns (" + m+ ")");
        if (b.length != n && !allowBufferedEntries || b.length < n)
            throw new IllegalArgumentException("b length (" + b.length + ") != matrix rows (" + n + ")");
        if (r.length < n || rHat0.length < n || p.length < m || v.length < n || s.length < m || t.length < n)
            throw new IllegalArgumentException(
                    "Working buffers r/rHat0/v/s/t must have length >= " + n + "and p/s must have length >=" + m);

        // r = b - A*x ; rHat0 = r (fixed shadow residual, arbitrary choice)
        A.multiply(x, v); // use v as temp for the initial residual

        int idx = 0;
        for (int i = 0; i < m; i++) {
            p[i] = 0;
            s[i] = 0;
            if (!fixedVariables[i]) {
                if (idx < n) {
                    r[idx] = b[idx] - v[idx];
                    rHat0[idx] = r[idx];
                    unknownIdx[idx] = i;
                    idx++;
                }
            }
        }

        if (idx != n)
            throw new IllegalStateException(
                    "BiCGStab requires number of free variables == equations"
            );

        double rho = 1, alpha = 1, omega = 1;

        if (Math.sqrt(dot(r, r, n)) < tol) return x;

        for (int k = 0; k < maxIter; k++) {
            double rhoNew = dot(rHat0, r, n);
            if (rhoNew == 0) break; // breakdown: rHat0 orthogonal to r

            double beta = (rhoNew / rho) * (alpha / omega);
            for (int i = 0; i < n; i++){
                p[unknownIdx[i]] = r[i] + beta * (p[unknownIdx[i]] - omega * v[i]);
            }


            A.multiply(p, v);

            double rHat0v = dot(v, rHat0, n);
            if (rHat0v == 0) break; // breakdown
            alpha = rhoNew / rHat0v;

            for (int i = 0; i < n; i++){
                s[unknownIdx[i]] = r[i] - alpha * v[i];
            }

            double sNorm = Math.sqrt(skippedDotBoth(s, s, unknownIdx,n));
            //we only evaluate the unknows -> should be faster

            if (sNorm < tol) {
                for (int i = 0; i < n; i++){
                    x[unknownIdx[i]] += alpha * p[unknownIdx[i]];
                }
                break;
            }

            A.multiply(s, t);

            double tDotT = dot(t, t, n);
            omega = (tDotT == 0) ? 0 : skippedDot(s,t,unknownIdx, n) / tDotT;

            for (int i = 0; i < n; i++){
                x[unknownIdx[i]] += alpha * p[unknownIdx[i]] + omega * s[unknownIdx[i]];
                r[i] = s[unknownIdx[i]] - omega * t[i];
            }

            if (Math.sqrt(dot(r, r, n)) < tol) break;
            if (omega == 0) break; // breakdown

            rho = rhoNew;
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

    private static double skippedDotBoth(double[] a, double[] b, int[] unknowIdx, int n){
        double sum = 0;
        for (int i = 0; i < n; i++){
            sum += a[unknowIdx[i]] * b[unknowIdx[i]];
        }
        return sum;
    }
}