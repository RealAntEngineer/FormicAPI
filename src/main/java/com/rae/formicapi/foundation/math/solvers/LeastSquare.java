package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.Matrix;

@Deprecated
public class LeastSquare {

    /**
     * Convenience overload: zero initial guess, allocates working buffers internally.
     * Use only when the caller has no long-lived matrix to cache buffers on — prefer
     * the pre-allocated overload for any hot path.
     */
    public static double[] solve(Matrix A, double[] b, int maxIter, float tol) {
        int m = A.cols();
        int n = A.rows();
        return solve(A, b, maxIter, tol,
                new double[m], new double[m], new double[m], new double[m], new double[m], new double[n]);
    }

    public static double[] solve(Matrix A, double[] b,double[] x0,int maxIter, float tol) {
        int m = A.cols();
        int n = A.rows();
        return solve(A, b, maxIter, tol,
                x0, new double[m], new double[m], new double[m], new double[m], new double[n]);
    }

    /**
     * Solve {@code Ax = b} in the least-squares sense using CG on the normal equations
     * {@code AᵀA x = Aᵀb}, with caller-supplied working buffers.
     *
     * <p>Pass pre-allocated arrays from a long-lived object (e.g. {@code PhysicsMatrix})
     * to avoid allocating ~4 × n doubles on every call. At 376 832 voxels and 20 ticks/s
     * the naive version allocates ~240 MB/s; this overload allocates nothing after warmup.
     *
     * <p>The contents of all four working arrays are overwritten on every call.
     * Their values between calls are undefined and must not be read by the caller.
     *
     * @param A      input matrix (square or rectangular)
     * @param b      right-hand side, length {@code A.rows()}
     * @param maxIter maximum CG iterations
     * @param tol    convergence tolerance on the residual norm
     * @param r      pre-allocated residual buffer,          length ≥ {@code A.cols()}
     * @param p      pre-allocated search-direction buffer,  length ≥ {@code A.cols()}
     * @param Ap     pre-allocated AᵀA·p buffer,            length ≥ {@code A.cols()}
     * @param temp   pre-allocated A·p intermediate buffer,  length ≥ {@code A.rows()}
     * @return solution vector x (a new array of length {@code A.cols()})
     */
    public static double[] solve(Matrix A, double[] b, int maxIter, float tol,
                                 double[] initialX, double[] r, double[] p, double[] Atb, double[] Ap, double[] temp) {
        int n = A.rows();
        int m = A.cols();

        if (b.length != n)
            throw new IllegalArgumentException(
                    "RHS length (" + b.length + ") != matrix rows (" + n + ")");
        if (r.length < m || p.length < m || Ap.length < m)
            throw new IllegalArgumentException(
                    "Working buffers r/p/Ap must have length >= A.cols() = " + m);
        if (temp.length < n)
            throw new IllegalArgumentException(
                    "Working buffer temp must have length >= A.rows() = " + n);
        if (initialX.length != m)
            throw new IllegalArgumentException(
                    "Initial guess length (" + initialX.length + ") does not match matrix columns (" + m + ")"
            );

        // Aᵀb — written into r temporarily, then copied to Atb slot
        A.transposeMultiply(b, Atb);

        return conjugateGradientNormalEq(A, initialX, Atb, maxIter, tol, r, p, Ap, temp);
    }

    /**
     * CG on AᵀA x = Aᵀb without forming AᵀA explicitly.
     * All working arrays are passed in and reused across calls.
     */
    private static double[] conjugateGradientNormalEq(
            Matrix A, double[] x, double[] Atb, int maxIter, double tol,
            double[] r, double[] p, double[] Ap, double[] temp) {

        int n = A.rows();
        int m = A.cols();

        // r = Atb - AᵀA·x  (x is zero, so r = Atb on first call)
        multiplyAtA(A, x, temp, Ap);
        for (int i = 0; i < m; i++) {
            r[i] = Atb[i] - Ap[i];
            p[i] = r[i];
        }

        double rsold = dot(r, r, m);

        for (int k = 0; k < maxIter; k++) {
            multiplyAtA(A, p, temp, Ap);

            double dotPAp = dot(p, Ap, m);
            if (dotPAp == 0) break;
            double alpha = rsold / dotPAp;

            for (int i = 0; i < m; i++) x[i] += alpha * p[i];
            for (int i = 0; i < m; i++) r[i] -= alpha * Ap[i];

            double rsnew = dot(r, r, m);
            if (Math.sqrt(rsnew) < tol) break;

            double beta = rsnew / rsold;
            for (int i = 0; i < m; i++) p[i] = r[i] + beta * p[i];
            rsold = rsnew;
        }
        return x;
    }

    private static void multiplyAtA(Matrix A, double[] p, double[] temp, double[] result) {
        A.multiply(p, temp);
        A.transposeMultiply(temp, result);
    }

    private static double dot(double[] a, double[] b, int len) {
        double sum = 0;
        for (int i = 0; i < len; i++) sum += a[i] * b[i];
        return sum;
    }
}