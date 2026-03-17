package com.rae.formicapi.math.solvers;

import com.rae.formicapi.math.operators.Matrix;
import com.rae.formicapi.math.operators.MutableMatrix;

public class LeastSquare {

    /**
     * Solve Ax = b in least-squares sense.
     * Works for square, rectangular, symmetric or non-symmetric matrices.
     *
     * @param A       input matrix
     * @param x_init  initial guess (length must match A.cols())
     * @param b       right-hand side vector (length must match A.rows())
     * @param maxIter maximum iterations
     * @param tol     tolerance for residual
     * @return solution vector x
     */
    public static double[] solve(Matrix A, double[] x_init, double[] b, int maxIter, double tol) {
        int n = A.rows();
        int m = A.cols();

        if (b.length != n)
            throw new IllegalArgumentException(
                    "RHS vector length (" + b.length + ") does not match matrix rows (" + n + ")"
            );

        if (x_init.length != m)
            throw new IllegalArgumentException(
                    "Initial guess length (" + x_init.length + ") does not match matrix columns (" + m + ")"
            );

        // Compute Aᵀ * b
        double[] Atb = new double[m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                Atb[j] += A.get(i, j) * b[i];

        // Use CG on normal equations AᵀA x = Aᵀb
        return conjugateGradientNormalEq(A, x_init.clone(), Atb, maxIter, tol);
    }

    /** CG on AᵀA x = Aᵀb without forming AᵀA explicitly */
    private static double[] conjugateGradientNormalEq(Matrix A, double[] x, double[] Atb, int maxIter, double tol) {
        int n = A.rows();
        int m = A.cols();

        double[] r = new double[m]; // residual
        double[] p = new double[m]; // search direction
        double[] Ap = new double[m]; // AᵀA * p
        double[] temp = new double[n]; // temp = A*p

        // r = Atb - Aᵀ(A*x)
        multiplyAtA(A, x, temp, Ap);
        for (int i = 0; i < m; i++) {
            r[i] = Atb[i] - Ap[i];
            p[i] = r[i];
        }

        double rsold = dot(r, r);

        for (int k = 0; k < maxIter; k++) {
            multiplyAtA(A, p, temp, Ap);

            double dotPAp = dot(p, Ap);
            if (dotPAp == 0) break; // breakdown
            double alpha = rsold / dotPAp;

            for (int i = 0; i < m; i++) x[i] += alpha * p[i];
            for (int i = 0; i < m; i++) r[i] -= alpha * Ap[i];

            double rsnew = dot(r, r);
            if (Math.sqrt(rsnew) < tol) break;

            double beta = rsnew / rsold;
            for (int i = 0; i < m; i++) p[i] = r[i] + beta * p[i];
            rsold = rsnew;
        }

        return x;
    }

    /** Multiply by AᵀA efficiently: temp = A*p, result = Aᵀ*temp */
    private static void multiplyAtA(Matrix A, double[] p, double[] temp, double[] result) {
        A.multiply(p, temp);
        A.transposeMultiply(temp, result);
    }

    private static double dot(double[] a, double[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Dot product length mismatch: " + a.length + " vs " + b.length);
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    /** Convenience overload: zero initial guess */
    public static double[] solve(MutableMatrix A, double[] b, int maxIter, float tol) {
        return solve(A, new double[A.cols()], b, maxIter, tol);
    }
}
