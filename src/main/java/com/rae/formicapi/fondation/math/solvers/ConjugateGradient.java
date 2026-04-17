package com.rae.formicapi.fondation.math.solvers;

import com.rae.formicapi.fondation.math.operators.Matrix;
import com.rae.formicapi.fondation.math.operators.MutableMatrix;

public class ConjugateGradient {

    // Convenience overload
    public static double[] solve(MutableMatrix matrix, double[] rhs, int maxIter, float tol) {
        return solve(matrix, new double[matrix.cols()], rhs, maxIter, tol);
    }

    /**
     * Solve Ax = b using Conjugate Gradient.
     *
     * @param A       square matrix (must be rows == cols)
     * @param x_init  initial guess (length must match A.rows())
     * @param b       right-hand side vector (length must match A.rows())
     * @param maxIter maximum iterations
     * @param tol     tolerance for residual
     * @return solution vector x
     */
    public static double[] solve(Matrix A, double[] x_init, double[] b, int maxIter, double tol) {

        int n = A.rows();
        int m = A.cols();

        // Shape checks
        if (n != m) {
            throw new IllegalArgumentException(
                    "Conjugate Gradient requires a square matrix: rows = " + n + ", cols = " + m
            );
        }

        if (x_init.length != n) {
            throw new IllegalArgumentException(
                    "Initial guess vector length (" + x_init.length +
                            ") does not match matrix size (" + n + ")"
            );
        }

        if (b.length != n) {
            throw new IllegalArgumentException(
                    "RHS vector length (" + b.length +
                            ") does not match matrix size (" + n + ")"
            );
        }

        double[] x  = x_init.clone();
        double[] r  = new double[n];
        double[] p  = new double[n];
        double[] Ap = new double[n];

        // r = b - A * x
        A.multiply(x, Ap);
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - Ap[i];
            p[i] = r[i]; // initial search direction
        }

        double rsold = dot(r, r);

        for (int i = 0; i < maxIter; i++) {

            A.multiply(p, Ap);

            double dotPAp = dot(p, Ap);
            if (dotPAp == 0) {
                throw new ArithmeticException(
                        "Breakdown in Conjugate Gradient: division by zero in iteration " + i
                );
            }

            double alpha = rsold / dotPAp;

            for (int j = 0; j < n; j++)
                x[j] += alpha * p[j];

            for (int j = 0; j < n; j++)
                r[j] -= alpha * Ap[j];

            double rsnew = dot(r, r);

            if (Math.sqrt(rsnew) < tol)
                break;

            double beta = rsnew / rsold;

            for (int j = 0; j < n; j++)
                p[j] = r[j] + beta * p[j];

            rsold = rsnew;
        }

        return x;
    }

    private static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Cannot compute dot product: vector lengths differ (" + a.length + " vs " + b.length + ")"
            );
        }
        double sum = 0;
        for (int i = 0; i < a.length; i++)
            sum += a[i] * b[i];
        return sum;
    }
}
