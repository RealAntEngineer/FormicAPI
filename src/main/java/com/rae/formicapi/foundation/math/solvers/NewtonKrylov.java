package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR2Tensor;
import org.jetbrains.annotations.Nullable;

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
     * Optional convergence diagnostics for a {@link #solve} call. Pass
     * {@code null} anywhere a {@code Stats} parameter is accepted to skip
     * tracking entirely (zero overhead, same as before this existed).
     *
     * <p>Not thread-safe; use one instance per solve.
     */
    public static final class Stats {

        /** ||F(x)||₂ at the start of each Newton iteration, in order. */
        public final java.util.List<Double> newtonResiduals = new java.util.ArrayList<>();

        /** BiCGSTAB iterations actually used per Newton step (parallel to {@link #newtonResiduals}). */
        public final java.util.List<Integer> linearIterations = new java.util.ArrayList<>();

        /** Total Newton iterations performed. Equal to {@code newtonResiduals.size()}. */
        public int newtonIterationCount;

        /** Whether the solve converged (residual < newtonTol) before exhausting maxNewtonIter. */
        public boolean converged;

        /** Clears all recorded data so the same Stats instance can be reused across solves. */
        public void reset() {
            newtonResiduals.clear();
            linearIterations.clear();
            newtonIterationCount = 0;
            converged = false;
        }
    }


    public static void solve(PaddedCSR2Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                             double newtonTol, double linearTol) {
        solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, null);
    }

    /**
     * Same as {@link #solve(PaddedCSR2Tensor, Matrix, double[], double[], int, int, double, double)}
     * but recording per-iteration diagnostics into {@code stats} if non-null.
     */
    public static void solve(PaddedCSR2Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                             double newtonTol, double linearTol, @Nullable Stats stats) {
        int n = C.equations();

        solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol,
                new double[n], new double[n], new double[n], new double[n], new double[n], new double[n],
                new double[n], new double[n], new double[n], stats);
    }

    public static double[] solve(PaddedCSR2Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                                 double newtonTol, double linearTol, double[] Ax, double[] F, double[] dx, double[] r, double[] rHat0,
                                 double[] p, double[] v, double[] s, double[] t){

        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, Ax, F, dx,
                r, rHat0, p, v, s, t, null);

    }
    /**
     * Solve the nonlinear system
     *
     * <pre>
     *     C:x:x + A*x - b = 0
     * </pre>
     *
     * using Newton iterations with matrix-free BiCGSTAB linear solves.
     *
     * <p>At each Newton iteration, the Jacobian system is solved without
     * explicitly assembling the Jacobian:
     *
     * <pre>
     *     J(x) * dx = F(x)
     * </pre>
     *
     * where {@code dx} is the correction direction applied as:
     *
     * <pre>
     *     x = x - dx
     * </pre>
     *
     * <p>The Jacobian-vector products are evaluated directly from the nonlinear
     * tensor and linear matrix terms:
     *
     * <pre>
     *     J(x)v = C'(x)v + A*v
     * </pre>
     *
     * <p>{@code x} is both the initial guess and the solution. Supplying a warm
     * start from a previous solve can significantly reduce the number of Newton
     * iterations.
     *
     * <p>No temporary arrays are allocated. Every working buffer is supplied by
     * the caller and overwritten during the solve.
     *
     * @param C quadratic tensor term defining the nonlinear contribution
     * @param A linear matrix term
     * @param x initial guess on entry, solution on exit
     * @param b right-hand side vector
     * @param maxNewtonIter maximum Newton iterations
     * @param maxLinearIter maximum BiCGSTAB iterations per Newton step
     * @param newtonTol convergence tolerance on {@code ||F(x)||₂}
     * @param linearTol convergence tolerance for each linear solve
     * @param Ax buffer for the linear contribution {@code A*x}
     * @param F nonlinear residual buffer; stores {@code C:x:x + A*x - b}
     * @param dx Newton correction direction buffer; applied as {@code x -= dx}
     * @param r BiCGSTAB residual buffer
     * @param rHat0 BiCGSTAB shadow residual buffer
     * @param p BiCGSTAB search direction buffer
     * @param v working buffer for Jacobian-vector products
     * @param s BiCGSTAB stabilizer buffer
     * @param t working buffer for Jacobian-vector products
     *
     * @return {@code x} (same array passed as input)
     *
     * @throws IllegalArgumentException if the supplied buffers are too small.
     */
    public static double[] solve(PaddedCSR2Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                                 double newtonTol, double linearTol, double[] Ax, double[] F, double[] dx, double[] r, double[] rHat0,
                                 double[] p, double[] v, double[] s, double[] t, @Nullable Stats stats) {

        int n = b.length;

        for (int iteration = 0; iteration < maxNewtonIter; iteration++) {

            C.multiply(x, F);
            A.multiply(x, Ax);

            double norm = 0;

            for (int i = 0; i < n; i++) {
                F[i] += Ax[i] - b[i];
                norm += F[i] * F[i];
            }

            if (!Double.isFinite(norm))
                throw new ArithmeticException(
                        "NewtonKrylov: residual became non-finite at Newton iteration " + iteration
                                + " (norm=" + norm + "). The linear solve likely diverged/broke down"
                                + " on the previous step -- check conditioning rather than continuing.");

            double residual = Math.sqrt(norm);

            if (stats != null) {
                stats.newtonResiduals.add(residual);
                stats.newtonIterationCount = iteration + 1;
            }

            if (residual < newtonTol) {
                if (stats != null) stats.converged = true;
                return x;
            }

            for (int i = 0; i < n; i++)
                dx[i] = 0;

            int linearIterations = solveNewtonStep(C, A, x, dx, F, maxLinearIter, linearTol, r, rHat0, p, v, s, t);

            if (stats != null)
                stats.linearIterations.add(linearIterations);

            for (int i = 0; i < n; i++)
                x[i] += dx[i];
        }
        return x;
    }

    //return value is redundant.
    /**
     * Solve one Newton linearization:
     *
     * <pre>
     *     J(x) * dx = -F(x)
     * </pre>
     *
     * using a matrix-free BiCGSTAB iteration.
     *
     * <p>The Jacobian is never assembled explicitly. Matrix-vector products are
     * evaluated directly through the quadratic tensor Jacobian contribution and
     * the linear matrix term:
     *
     * <pre>
     *     J(x)v = C'(x)v + A*v
     * </pre>
     *
     * <p>{@code dx} is both the initial guess and the solution. It is typically
     * initialized to zero by the caller before each Newton iteration. The
     * resulting correction is applied by the caller using:
     *
     * <pre>
     *     x += dx
     * </pre>
     *
     * <p>All working buffers are overwritten during the solve and their contents
     * are undefined afterward.
     *
     * @param C       quadratic tensor term defining the nonlinear Jacobian contribution
     * @param A       linear matrix term
     * @param x       current Newton iterate where the Jacobian is evaluated
     * @param dx      initial guess on entry, Newton correction on exit
     * @param F       nonlinear residual {@code F(x)}; the solver internally solves
     *                the system with right-hand side {@code -F(x)}
     * @param maxIter maximum BiCGSTAB iterations
     * @param tol     convergence tolerance on the linear residual norm
     * @param r       BiCGSTAB residual buffer
     * @param rHat0   BiCGSTAB shadow residual buffer
     * @param p       BiCGSTAB search direction buffer
     * @param v       temporary buffer for Jacobian-vector products
     * @param s       BiCGSTAB stabilizer buffer
     * @param t       temporary buffer for Jacobian-vector products
     */
    private static int solveNewtonStep(PaddedCSR2Tensor C, Matrix A, double[] x, double[] dx, double[] F, int maxIter,
                                        double tol, double[] r, double[] rHat0, double[] p, double[] v, double[] s, double[] t) {

        int n = F.length;

        // r = F - J(x)*dx, evaluated matrix-free
        C.multiplyJacobian(x, dx, v);
        A.multiply(dx, t);

        for (int i = 0; i < n; i++) {
            r[i] = -F[i] - v[i] - t[i];
            rHat0[i] = r[i];
            p[i] = 0;
            v[i] = 0;
        }

        double rho = 1;
        double alpha = 1;
        double omega = 1;

        // Breakdown threshold: BiCGSTAB divides by rho, rHatV, and tt each
        // iteration. On an ill-conditioned/near-singular Jacobian (e.g. a
        // saddle-point system with a very small pressure penalty) these can
        // shrink toward zero without ever hitting it exactly, so alpha/beta
        // blow up and silently poison dx with Inf/NaN. Treat "small" the
        // same as "zero" and restart the Krylov subspace from the current
        // dx instead of dividing by near-nothing.
        double breakdownTol = 1e-30;

        int iterationsUsed = 0;
        
        for (int k = 0; k < maxIter; k++) {

            iterationsUsed = k + 1;

            double rhoNew = dot(rHat0, r, n);

            if (Math.abs(rhoNew) < breakdownTol) {
                // Restart: recompute the true residual at the current dx and
                // reseed the shadow residual, rather than aborting with
                // whatever dx we happened to have.
                C.multiplyJacobian(x, dx, v);
                A.multiply(dx, t);
                for (int i = 0; i < n; i++) {
                    r[i] = -F[i] - v[i] - t[i];
                    rHat0[i] = r[i];
                    p[i] = 0;
                    v[i] = 0;
                }
                rho = 1;
                alpha = 1;
                omega = 1;
                continue;
            }

            double beta = (rhoNew / rho) * (alpha / omega);

            for (int i = 0; i < n; i++)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);

            // v = J(x)*p
            C.multiplyJacobian(x, p, v);
            A.multiply(p, t);

            for (int i = 0; i < n; i++)
                v[i] += t[i];


            double rHatV = dot(rHat0, v, n);
            if (Math.abs(rHatV) < breakdownTol)
                break;

            alpha = rhoNew / rHatV;

            if (!Double.isFinite(alpha))
                throw new ArithmeticException(
                        "NewtonKrylov: BiCGSTAB coefficient alpha became non-finite at linear iteration "
                                + k + " (rhoNew=" + rhoNew + ", rHatV=" + rHatV + "). Indicates a Krylov"
                                + " breakdown on an ill-conditioned/near-singular Jacobian.");


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

            if (Math.abs(tt) < breakdownTol)
                break;

            omega = dot(t, s, n) / tt;

            if (!Double.isFinite(omega))
                throw new ArithmeticException(
                        "NewtonKrylov: BiCGSTAB coefficient omega became non-finite at linear iteration "
                                + k + " (tt=" + tt + "). Indicates a Krylov breakdown on an"
                                + " ill-conditioned/near-singular Jacobian.");

            for (int i = 0; i < n; i++) {
                dx[i] += alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * t[i];
            }

            if (norm(r) < tol || omega == 0)
                break;

            rho = rhoNew;
        }

        return iterationsUsed;
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