package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.Matrix;
import com.rae.formicapi.foundation.math.operators.PaddedCSR2Tensor;

/**
 * Newton-Krylov solver for nonlinear systems of the form
 *
 * <pre>
 *     C:x:x + A*x - b = 0
 * </pre>
 *
 * where {@code C} is represented by a {@link PaddedCSR2Tensor} and
 * {@code A} is a linear matrix.
 *
 * <p>The nonlinear system is solved using Newton iterations. Each Newton step
 * requires solving
 *
 * <pre>
 *     J(x) Δx = -F(x)
 * </pre>
 *
 * where
 *
 * <pre>
 *     F(x) = C:x:x + A*x - b.
 * </pre>
 *
 * <p>The Jacobian is never assembled explicitly. Instead, the linear solve is
 * performed in matrix-free form using BiCGSTAB and the Jacobian-vector product
 *
 * <pre>
 *     J(x)v = d(C:x:x)/dx · v + A*v
 * </pre>
 *
 * computed by {@link PaddedCSR2Tensor#multiplyJacobian(double[], double[], double[])}.
 *
 * <p>This avoids constructing the Jacobian matrix while retaining quadratic
 * convergence whenever Newton's method converges.
 */
public class NewtonKrylov {

    /**
     * Solve the nonlinear system
     *
     * <pre>
     *     C:x:x + A*x - b = 0
     * </pre>
     *
     * using Newton iterations with matrix-free BiCGSTAB linear solves.
     *
     * <p>{@code x} is both the initial guess and the solution. Supplying a warm
     * start from a previous solve can significantly reduce the number of Newton
     * iterations.
     *
     * <p>No temporary arrays are allocated. Every working buffer is supplied by
     * the caller and overwritten during the solve.
     *
     * @param C quadratic tensor term
     * @param A linear matrix term
     * @param x initial guess on entry, solution on exit
     * @param b right-hand side vector
     * @param maxNewtonIter maximum Newton iterations
     * @param maxLinearIter maximum BiCGSTAB iterations per Newton step
     * @param newtonTol convergence tolerance on {@code ||F(x)||₂}
     * @param linearTol convergence tolerance for each linear solve
     * @param F nonlinear residual buffer; length ≥ number of equations
     * @param dx Newton correction buffer; length ≥ number of variables
     * @param r BiCGSTAB residual buffer
     * @param rHat0 BiCGSTAB shadow residual buffer
     * @param p BiCGSTAB search direction buffer
     * @param v working buffer storing {@code J(x)p}
     * @param s BiCGSTAB stabilizer buffer
     * @param t working buffer storing {@code J(x)s}
     *
     * @return {@code x} (same array passed as input)
     *
     * @throws IllegalArgumentException if the supplied buffers are too small.
     */
    public static double[] solve(PaddedCSR2Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                                 double newtonTol, double linearTol, double[] F, double[] dx, double[] r, double[] rHat0,
                                 double[] p, double[] v, double[] s, double[] t) {

        int n = b.length;

        double[] Ax = new double[n];

        for (int iteration = 0; iteration < maxNewtonIter; iteration++) {

            C.multiply(x, F);
            A.multiply(x, Ax);

            double norm = 0;

            for (int i = 0; i < n; i++) {
                F[i] += Ax[i] - b[i];
                norm += F[i] * F[i];
            }

            if (Math.sqrt(norm) < newtonTol)
                return x;


            for (int i = 0; i < n; i++)
                dx[i] = 0;


            solveNewtonStep(C, A, x, dx, F, maxLinearIter, linearTol, r, rHat0, p, v, s, t);


            for (int i = 0; i < n; i++)
                x[i] += dx[i];
        }

        return x;
    }

    //return value is redundant.
    /**
     * Solve one Newton linearization
     *
     * <pre>
     *     J(x)Δx = -F(x)
     * </pre>
     *
     * using a matrix-free BiCGSTAB iteration.
     *
     * <p>The Jacobian is evaluated only through Jacobian-vector products. No
     * Jacobian matrix is assembled.
     *
     * <p>{@code dx} is both the initial guess and the solution. It is typically
     * initialized to zero by the caller before each Newton iteration.
     *
     * <p>All working buffers are overwritten.
     *
     * @param C quadratic tensor term
     * @param A linear matrix term
     * @param x current Newton iterate
     * @param dx initial guess on entry, Newton correction on exit
     * @param b right-hand side of the linearized system (typically {@code -F(x)})
     * @param maxIter maximum BiCGSTAB iterations
     * @param tol convergence tolerance on the linear residual
     * @param r residual buffer
     * @param rHat0 shadow residual buffer
     * @param p search direction buffer
     * @param v working buffer storing {@code J(x)p}
     * @param s stabilizer buffer
     * @param t working buffer storing {@code J(x)s}
     *
     * @return {@code dx}
     */
    private static double[] solveNewtonStep(PaddedCSR2Tensor C, Matrix A, double[] x, double[] dx, double[] b, int maxIter,
                                            double tol, double[] r, double[] rHat0, double[] p, double[] v, double[] s, double[] t) {

        int n = b.length;

        // r = b - J(x)*dx
        C.multiplyJacobian(x, dx, v);
        A.multiply(dx, t);

        for (int i = 0; i < n; i++) {
            r[i] = b[i] - v[i] - t[i];
            rHat0[i] = r[i];
            p[i] = 0;
            v[i] = 0;
        }

        double rho = 1;
        double alpha = 1;
        double omega = 1;

        for (int k = 0; k < maxIter; k++) {

            double rhoNew = dot(rHat0, r, n);
            if (rhoNew == 0)
                break;

            double beta = (rhoNew / rho) * (alpha / omega);

            for (int i = 0; i < n; i++)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);



            //TODO code duplicate, should go into a private function
            // v = J(x)*p
            C.multiplyJacobian(x, p, v);
            A.multiply(p, t);

            for (int i = 0; i < n; i++)
                v[i] += t[i];


            double rHatV = dot(rHat0, v, n);
            if (rHatV == 0)
                break;

            alpha = rhoNew / rHatV;


            for (int i = 0; i < n; i++)
                s[i] = r[i] - alpha * v[i];


            if (norm(s) < tol) {
                for (int i = 0; i < n; i++)
                    dx[i] += alpha * p[i];

                break;
            }


            // t = J(x)*s
            C.multiplyJacobian(x, s, t);
            A.multiply(s, v);

            for (int i = 0; i < n; i++)
                t[i] += v[i];


            double tt = dot(t, t, n);
            if (tt == 0)
                break;

            omega = dot(t, s, n) / tt;


            for (int i = 0; i < n; i++) {
                dx[i] += alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * t[i];
            }


            if (norm(r) < tol)
                break;

            if (omega == 0)
                break;

            rho = rhoNew;
        }

        return dx;
    }


    private static double dot(double[] a, double[] b, int n) {
        double sum = 0;

        for (int i = 0; i < n; i++)
            sum += a[i] * b[i];

        return sum;
    }


    private static double norm(double[] a) {
        return Math.sqrt(dot(a, a, a.length));
    }
}