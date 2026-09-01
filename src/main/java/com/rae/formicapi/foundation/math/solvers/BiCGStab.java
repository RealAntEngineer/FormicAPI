package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuIntegerVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import org.jetbrains.annotations.Nullable;

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

    // Constrained equation-space buffers.
    private static final int R  = 0;
    private static final int R0 = 1;
    private static final int V = 2;
    private static final int T = 3;

    // Constrained variable-space buffers.
    private static final int P = 0;
    private static final int S = 1;


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

    public static int solve(Matrix A, DoubleVector x, DoubleVector b, int maxIter, double tol) {
        int n = A.rows();
        int m = A.cols();

        return solve(A, x, b, maxIter, tol, null, null,
                new WorkingBuffer<>(4, () -> new CpuDoubleVector(n)),
                new WorkingBuffer<>(2, () -> new CpuDoubleVector(m)));
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

        IntegerVector unknownIdx = new CpuIntegerVector(n);
        Util.fillUnknowIdx(fixedVariables, unknownIdx, A.inputSize(), A.outputSize());
        solveConstrained(A, xv, bv, maxIter, tol, unknownIdx);
        return ((CpuDoubleVector) xv).array();
    }

    /**
     * Solve a constrained system, allocating its working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static int solveConstrained(Matrix A, DoubleVector x, DoubleVector b,
                                       int maxIter, double tol, IntegerVector unknownIdx) {
        int n = A.rows();
        int m = A.cols();


        return solve(A, x, b, maxIter, tol, unknownIdx,null,
                new WorkingBuffer<>(4, () -> new CpuDoubleVector(n)),
                new WorkingBuffer<>(3, () -> new CpuDoubleVector(m)));
    }

    /**
     * Solve a constrained linear system {@code Ax = b} using BiCGSTAB while keeping fixed variables embedded
     * in the original solution vector.
     *
     * <p>The solver operates directly on the full matrix without constructing an explicit reduced system.
     * Variables marked in {@code fixedVariables} are treated as prescribed values and are never modified.
     * Only free variables are updated by the BiCGSTAB iterations.
     *
     * <p>The solution vector {@code x} remains in the original variable space. Fixed entries contain their
     * prescribed values and are kept unchanged. Free entries are the actual unknowns solved by BiCGSTAB.
     *
     * <p>The search vectors {@code p} and {@code s} are stored in the full variable space. Entries corresponding
     * to fixed variables are always zero. Entries corresponding to free variables contain the BiCGSTAB search
     * direction and stabilizer vectors.
     *
     * <p>The vectors {@code r}, {@code rHat0}, {@code v}, and {@code t} live in equation space. The {@code unknownIdx}
     * vector maps each equation index to the corresponding free variable index in the full variable space.
     *
     * <p>The initial residual includes the contribution of the fixed variables:
     *
     * <pre>
     * r = b - A*x
     * </pre>
     *
     * <p>After initialization, BiCGSTAB only modifies the free variables. Fixed variables are never restored or
     * projected because they are never changed.
     *
     * <p>This method requires the number of free variables to equal the number of equations. Unlike {@link ConjugateGradient},
     * the matrix does not need to be symmetric or positive definite.
     *
     * @param A matrix representing the constrained system
     * @param x initial guess on entry, solution on exit; fixed entries remain unchanged
     * @param b right-hand side vector; one entry per equation
     * @param maxIter maximum number of BiCGSTAB iterations
     * @param tol convergence threshold on the residual norm {@code ||r||₂}
     * @param unknownIdx working index vector of size {@code n}; after initialization,
     *        {@code unknownIdx[equationIndex] = variableIndex}
     * @param scaling equation space scaling for normalising the residual to avoid
     *
     * @param nBuffer working buffer providing 4 vectors of size {@code n}:
     *        residual {@code r}, shadow residual {@code rHat0}, {@code A*p} {@code v},
     *        and {@code A*s} {@code t}
     * @param mBuffer working buffer providing 2 vectors of size {@code m}:
     *        search direction {@code p} and stabilizer {@code s}
     *
     * @return the number of iterations performed
     *
     * @throws IllegalArgumentException if supplied buffers are too small or
     *         incompatible with matrix dimensions
     * @throws IllegalStateException if the number of free variables does not equal
     *         the number of equations
     */
    public static int solve(Matrix A, DoubleVector x, DoubleVector b, int maxIter, double tol,
                            @Nullable IntegerVector unknownIdx, @Nullable DoubleVector scaling,
                            WorkingBuffer<DoubleVector> nBuffer, WorkingBuffer<DoubleVector> mBuffer) {
        int n = A.rows();
        int m = A.cols();

        boolean constrained = unknownIdx != null;
        boolean scalePrecondition = scaling != null;
        //TODO this is scaling on input, should test scaling on output
        // equivalent to solve si * (Aij * xj) = si * bi


        if (!constrained && n != m)
            throw new IllegalArgumentException("BiCGSTAB unconstrained requires a square matrix: " +
                    "rows=" + n + ", cols=" + m);
        if (constrained && n > m)
            throw new IllegalArgumentException("BiCGSTAB constrained requires an under constrained matrix: " +
                    "rows=" + n + ", cols=" + m);

        if (x.size() != m)
            throw new IllegalArgumentException("x size (" + x.size() + ") != matrix columns (" + m + ")");

        if (b.size() != n)
            throw new IllegalArgumentException("b size (" + b.size() + ") != matrix rows (" + n + ")");

        if (nBuffer.vectorNumber() < 4 || nBuffer.get(R).size() < n || nBuffer.get(R0).size() < n
                || nBuffer.get(V).size() < n || nBuffer.get(T).size() < n)
            throw new IllegalArgumentException("Equation working buffers r/r0/v/t must have length >= " + n);

        if (mBuffer.vectorNumber() < 2|| mBuffer.get(P).size() < m || mBuffer.get(S).size() < m)
            throw new IllegalArgumentException("Variable working buffers p/s must have length >= " + m);

        if (constrained && unknownIdx.size() < n)
            throw new IllegalArgumentException("unknownIdx size (" + unknownIdx.size() + ") < matrix rows (" + n + ")");

        if (scalePrecondition && scaling.size() < n)
            throw new IllegalArgumentException("scaling size (" + scaling.size() + ") < matrix rows (" + n + ")");

        DoubleVector r = nBuffer.get(R);
        DoubleVector r0 = nBuffer.get(R0);
        DoubleVector v = nBuffer.get(V);
        DoubleVector t = nBuffer.get(T);

        DoubleVector p = mBuffer.get(P);
        DoubleVector s = mBuffer.get(S);

        p.clear();
        s.clear();

        // r = b - A*x ; r0 = r (fixed shadow residual, arbitrary choice)
        A.apply(x, v); // use v as temp for the initial residual and z as conditioned space

        r.clear();
        r.add(b);
        r.subtract(v);

        if (scalePrecondition) r.scale(scaling);

        r0.clear();
        r0.add(r);

        p.clear();
        s.clear();
        v.clear();
        t.clear();

        double rho = 1, alpha = 1, omega = 1;

        if (r.norm() < tol) return 0;

        int k = 0;
        for (; k < maxIter; k++) {
            double rhoNew = r0.dot(r);
            if (rhoNew == 0) break; // breakdown: r0 orthogonal to r

            double beta = (rhoNew / rho) * (alpha / omega);

            if (constrained) {
                p.skippedAxpy(-omega, v, unknownIdx, true, false);
                p.scale(beta); // fine for fixed direction since 0 * beta = 0
                p.skippedAdd(r, unknownIdx, true, false);
            } else {
                p.axpy(-omega, v);
                p.scale(beta);
                p.add(r);
            }

            A.apply(p, v);
            if (scalePrecondition) v.scale(scaling);

            double r0v = r0.dot(v);
            if (r0v == 0) break; // breakdown
            alpha = rhoNew / r0v;

            s.clear();
            if (constrained) {
                s.skippedAxpy(-alpha, v, unknownIdx, true, false);
                s.skippedAdd(r, unknownIdx, true, false);
            } else {
                s.axpy(-alpha, v);
                s.add(r);
            }

            double sNorm = constrained ? Math.sqrt(s.skippedDot(s, unknownIdx, true, true)) : s.norm();
            if (sNorm < tol) {
                if (constrained) x.skippedAxpy(alpha, p, unknownIdx, true, true);
                else  x.axpy(alpha, p);
                return k + 1;
            }

            A.apply(s, t);
            if (scalePrecondition) t.scale(scaling);

            double tDotT = t.dot(t);
            if (tDotT == 0) break;

            omega = (constrained ? s.skippedDot(t, unknownIdx, true, false) : t.dot(s)) / tDotT;
            if (omega == 0) break; // breakdown

            r.clear();

            if (constrained) {
                x.skippedAxpy(alpha, p, unknownIdx, true, true);
                x.skippedAxpy(omega, s, unknownIdx, true, true);
                r.skippedAdd(s, unknownIdx, false, true);

            } else {
                x.axpy(alpha, p);
                x.axpy(omega, s);
                r.add(s);
            }

            r.axpy(-omega,t);

            if (r.norm() < tol)
                return k + 1;

            rho = rhoNew;
        }

        return k;
    }
}