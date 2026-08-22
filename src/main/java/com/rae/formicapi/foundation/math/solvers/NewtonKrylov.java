package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import org.jetbrains.annotations.Nullable;

/**
 * Newton-Krylov solver for nonlinear systems of the form
 *
 * <pre>
 *     C:x:x + A*x - b = 0
 * </pre>
 *
 * where {@code C} is represented by a {@link PaddedCSR3Tensor} and
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
 * <p>This avoids constructing the Jacobian matrix while retaining quadratic
 * convergence whenever Newton's method converges.
 *
 * <p><b>Backend note:</b> every vector quantity here (residuals, search
 * directions, corrections) goes through {@link DoubleVector}, so this runs on
 * whichever backend {@code x}/{@code b}/the working buffers belong to — same
 * pattern as {@link LeastSquare} and {@link ConjugateGradient}. This requires
 * {@link PaddedCSR3Tensor} to expose {@code DoubleVector}-based overloads of
 * its matrix-free products, mirroring {@link Matrix#apply(DoubleVector, DoubleVector)}:
 *
 * <pre>
 *     void multiply(DoubleVector x, DoubleVector out);                          // out = C:x:x
 *     void multiplyJacobian(DoubleVector x, DoubleVector v, DoubleVector out);  // out = d(C:x:x)/dx · v
 * </pre>
 *
 * <p>No step in this solver needs per-element indexed access (no fixed/free
 * variable split like {@link ConjugateGradient#solveConstrained}), so
 * {@code dot}/{@code axpy}/{@code scale}/{@code add}/{@code copy}/{@code clear}
 * are all that's needed on top of those two tensor overloads.
 */
public class NewtonKrylov {

    // indices into the 9-vector WorkingBuffer used by the allocating overloads
    private static final int AX     = 0;
    private static final int RES    = 1; // F, the nonlinear residual
    private static final int DX     = 2;
    private static final int R      = 3;
    private static final int RHAT0  = 4;
    private static final int P      = 5;
    private static final int V      = 6;
    private static final int S      = 7;
    private static final int T      = 8;

    /**
     * Optional convergence diagnostics for a {@link #solve} call. Pass
     * {@code null} anywhere a {@code Stats} parameter is accepted to skip
     * tracking entirely (zero overhead, same as before this existed).
     *
     * <p>Not thread-safe; use one instance per solve.
     */
    public static final class Stats {//maybe make that more abstract ?

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

    // ---------------------------------------------------------------------
    // double[] convenience wrappers (allocate internally). Use only outside
    // hot paths -- prefer the DoubleVector/WorkingBuffer overloads if you
    // will solve repeatedly.
    // ---------------------------------------------------------------------

    public static void solve(PaddedCSR3Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                             double newtonTol, double linearTol) {
        solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, (Stats) null);
    }

    /**
     * Same as {@link #solve(PaddedCSR3Tensor, Matrix, double[], double[], int, int, double, double)}
     * but recording per-iteration diagnostics into {@code stats} if non-null.
     */
    public static void solve(PaddedCSR3Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                             double newtonTol, double linearTol, @Nullable Stats stats) {
        // CpuDoubleVector wraps the array by reference (no copy), and this solver
        // never calls resize() on x or b, so x is mutated in place exactly like
        // the original array-based version.
        solve(C, A, new CpuDoubleVector(x), new CpuDoubleVector(b), maxNewtonIter, maxLinearIter, newtonTol, linearTol, stats);
    }

    /**
     * double[]-buffer convenience overload for callers migrating from the old
     * array-based API. Wraps each array by reference (no copy) and delegates to
     * the DoubleVector core solve, so buffers are still reused with zero
     * per-call allocation.
     */
    public static double[] solve(PaddedCSR3Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                                 double newtonTol, double linearTol, double[] Ax, double[] F, double[] dx, double[] r, double[] rHat0,
                                 double[] p, double[] v, double[] s, double[] t) {
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, Ax, F, dx, r, rHat0, p, v, s, t, null);
    }

    public static double[] solve(PaddedCSR3Tensor C, Matrix A, double[] x, double[] b, int maxNewtonIter, int maxLinearIter,
                                 double newtonTol, double linearTol, double[] Ax, double[] F, double[] dx, double[] r, double[] rHat0,
                                 double[] p, double[] v, double[] s, double[] t, @Nullable Stats stats) {
        solve(C, A, new CpuDoubleVector(x), new CpuDoubleVector(b), maxNewtonIter, maxLinearIter, newtonTol, linearTol,
                new CpuDoubleVector(Ax), new CpuDoubleVector(F), new CpuDoubleVector(dx), new CpuDoubleVector(r), new CpuDoubleVector(rHat0),
                new CpuDoubleVector(p), new CpuDoubleVector(v), new CpuDoubleVector(s), new CpuDoubleVector(t), stats);
        return x;
    }

    // ---------------------------------------------------------------------
    // DoubleVector API
    // ---------------------------------------------------------------------

    /**
     * Solve, allocating a working buffer internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static DoubleVector solve(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b, int maxNewtonIter, int maxLinearIter,
                                     double newtonTol, double linearTol) {
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, (Stats) null);
    }

    /**
     * Same as above but recording per-iteration diagnostics into {@code stats} if non-null.
     * Allocates a working buffer internally — prefer the pre-allocated overload for hot paths.
     */
    public static DoubleVector solve(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b, int maxNewtonIter, int maxLinearIter,
                                     double newtonTol, double linearTol, @Nullable Stats stats) {
        int n = C.equations();
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol,
                new WorkingBuffer<>(9, () -> new CpuDoubleVector(n)), stats);
    }

    /**
     * Solve using a caller-supplied working buffer providing 9 vectors of size
     * {@code n = b.size()}: {@code Ax}, {@code F}, {@code dx}, {@code r},
     * {@code rHat0}, {@code p}, {@code v}, {@code s}, {@code t} (in that order).
     */
    public static DoubleVector solve(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b, int maxNewtonIter, int maxLinearIter,
                                     double newtonTol, double linearTol, WorkingBuffer<DoubleVector> buffer, @Nullable Stats stats) {
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol,
                buffer.get(AX), buffer.get(RES), buffer.get(DX), buffer.get(R), buffer.get(RHAT0),
                buffer.get(P), buffer.get(V), buffer.get(S), buffer.get(T), stats);
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
     * where {@code dx} is the correction direction applied as {@code x += dx}.
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
     * <p>No temporary vectors are allocated by this overload. Every working
     * buffer is supplied by the caller and overwritten during the solve.
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
     * @param dx Newton correction direction buffer; applied as {@code x += dx}
     * @param r BiCGSTAB residual buffer
     * @param rHat0 BiCGSTAB shadow residual buffer
     * @param p BiCGSTAB search direction buffer
     * @param v working buffer for Jacobian-vector products
     * @param s BiCGSTAB stabilizer buffer
     * @param t working buffer for Jacobian-vector products
     * @return {@code x} (same instance passed as input)
     */
    public static DoubleVector solve(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b, int maxNewtonIter, int maxLinearIter,
                                     double newtonTol, double linearTol, DoubleVector Ax, DoubleVector F, DoubleVector dx,
                                     DoubleVector r, DoubleVector rHat0, DoubleVector p, DoubleVector v, DoubleVector s, DoubleVector t,
                                     @Nullable Stats stats) {

        int n = b.size();

        Ax.resize(n);
        F.resize(n);
        dx.resize(n);
        r.resize(n);
        rHat0.resize(n);
        p.resize(n);
        v.resize(n);
        s.resize(n);
        t.resize(n);

        for (int iteration = 0; iteration < maxNewtonIter; iteration++) {

            C.apply(x, F);
            A.apply(x, Ax);
            F.add(Ax);
            F.axpy(-1.0, b);

            double norm = F.dot(F);

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

            dx.clear();

            int linearIterations = solveNewtonStep(C, A, x, dx, F, maxLinearIter, linearTol, r, rHat0, p, v, s, t);

            if (stats != null)
                stats.linearIterations.add(linearIterations);

            x.add(dx);
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
     * resulting correction is applied by the caller using {@code x += dx}.
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
    private static int solveNewtonStep(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector dx, DoubleVector F, int maxIter,
                                       double tol, DoubleVector r, DoubleVector rHat0, DoubleVector p, DoubleVector v, DoubleVector s,
                                       DoubleVector t) {

        // r = -F - J(x)*dx, evaluated matrix-free
        C.multiplyJacobian(x, dx, v);
        A.apply(dx, t);

        r.clear();
        r.axpy(-1.0, F);
        r.axpy(-1.0, v);
        r.axpy(-1.0, t);

        rHat0.copy(r);
        p.clear();
        v.clear();

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

            double rhoNew = rHat0.dot(r);

            if (Math.abs(rhoNew) < breakdownTol) {
                // Restart: recompute the true residual at the current dx and
                // reseed the shadow residual, rather than aborting with
                // whatever dx we happened to have.
                C.multiplyJacobian(x, dx, v);
                A.apply(dx, t);
                r.clear();
                r.axpy(-1.0, F);
                r.axpy(-1.0, v);
                r.axpy(-1.0, t);
                rHat0.copy(r);
                p.clear();
                v.clear();
                rho = 1;
                alpha = 1;
                omega = 1;
                continue;
            }

            double beta = (rhoNew / rho) * (alpha / omega);

            // p = r + beta*(p - omega*v)
            p.axpy(-omega, v);
            p.scale(beta);
            p.add(r);

            // v = J(x)*p
            C.multiplyJacobian(x, p, v);
            A.apply(p, t);
            v.add(t);

            double rHatV = rHat0.dot(v);
            if (Math.abs(rHatV) < breakdownTol)
                break;

            alpha = rhoNew / rHatV;

            if (!Double.isFinite(alpha))
                throw new ArithmeticException(
                        "NewtonKrylov: BiCGSTAB coefficient alpha became non-finite at linear iteration "
                                + k + " (rhoNew=" + rhoNew + ", rHatV=" + rHatV + "). Indicates a Krylov"
                                + " breakdown on an ill-conditioned/near-singular Jacobian.");

            // s = r - alpha*v
            s.copy(r);
            s.axpy(-alpha, v);

            if (norm(s) < tol) {
                dx.axpy(alpha, p);
                break;
            }

            // t = J(x)*s
            C.multiplyJacobian(x, s, t);
            A.apply(s, v);
            t.add(v);

            double tt = t.dot(t);

            if (Math.abs(tt) < breakdownTol)
                break;

            omega = t.dot(s) / tt;

            if (!Double.isFinite(omega))
                throw new ArithmeticException(
                        "NewtonKrylov: BiCGSTAB coefficient omega became non-finite at linear iteration "
                                + k + " (tt=" + tt + "). Indicates a Krylov breakdown on an"
                                + " ill-conditioned/near-singular Jacobian.");

            dx.axpy(alpha, p);
            dx.axpy(omega, s);
            r.copy(s);
            r.axpy(-omega, t);

            if (norm(r) < tol || omega == 0)
                break;

            rho = rhoNew;
        }

        return iterationsUsed;
    }

    private static double norm(DoubleVector a) {
        return Math.sqrt(a.dot(a));
    }
}