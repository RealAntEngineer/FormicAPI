package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuIntegerVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;

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

    // Equation-space buffers.
    private static final int R  = 0;
    private static final int R0 = 1;
    private static final int V  = 2;
    private static final int T     = 3;

    // Variable-space buffers.
    private static final int P = 4;
    private static final int S = 5;

    // Constrained equation-space buffers.
    private static final int CR  = 0;
    private static final int CR0 = 1;
    private static final int CV  = 2;
    private static final int CT  = 3;

    // Constrained variable-space buffers.
    private static final int CP = 0;
    private static final int CS = 1;


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
        return solve(A, x, b, maxIter, tol, new WorkingBuffer<>(6, () -> new CpuDoubleVector(n)));
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
     * @return {@code k} (same array as input, for chaining)
     * @throws IllegalArgumentException if matrix is not square or buffer sizes mismatch
     */
    public static int solve(Matrix A, DoubleVector x, DoubleVector b, int maxIter, double tol, WorkingBuffer<DoubleVector> buffers) {
        int n = A.outputSize();
        int m = A.inputSize();

        if (n != m)
            throw new IllegalArgumentException(
                    "Non constrained BiCGSTAB requires a square matrix: rows=" + n + ", cols=" + m);
        if (x.size() != n)
            throw new IllegalArgumentException(
                    "x length (" + x.size() + ") != matrix size (" + n + ")");
        if (b.size() != n)
            throw new IllegalArgumentException(
                    "b length (" + b.size() + ") != matrix size (" + n + ")");
        if (buffers.vectorNumber() < 6 || buffers.get(R).size() < n || buffers.get(R0).size() < n || buffers.get(P).size() < n
                || buffers.get(V).size() < n || buffers.get(S).size() < n || buffers.get(T).size() < n)
            throw new IllegalArgumentException(
                    "Working buffers r/rHat0/p/v/s/t must have length >= " + n);

        // r = b - A*x ; rHat0 = r (fixed shadow residual, arbitrary choice)
        A.apply(x, buffers.get(V)); // use v as temp for the initial residual


        /*for (int i = 0; i < n; i++) {
            r[i] = b[i] - v[i];
            rHat0[i] = r[i];
            p[i] = 0;
            v[i] = 0;
        }*/
        buffers.get(R).clear();
        buffers.get(R).add(b);
        buffers.get(R).subtract(buffers.get(V));

        buffers.get(R0).clear();
        buffers.get(R0).add(buffers.get(R));

        buffers.get(P).clear();
        buffers.get(V).clear();

        double rho = 1, alpha = 1, omega = 1;

        if (buffers.get(R).norm() < tol) return 0;
        int k = 0;
        for (;k < maxIter; k++) {
            //

            double rhoNew = buffers.get(R0).dot(buffers.get(R));
            if (rhoNew == 0) break; // breakdown: rHat0 orthogonal to r

            double beta = (rhoNew / rho) * (alpha / omega);
            //for (int i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * v[i]);

            buffers.get(P).axpy(-omega, buffers.get(V));
            buffers.get(P).scale(beta);//to eliminate the previous P
            buffers.get(P).add(buffers.get(R));

            A.apply(buffers.get(P), buffers.get(V));

            double rHat0v = buffers.get(R0).dot(buffers.get(V));//dot(rHat0, v, n);
            if (rHat0v == 0) break; // breakdown
            alpha = rhoNew / rHat0v;

            //for (int i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];
            buffers.get(S).clear();
            buffers.get(S).add(buffers.get(R));
            buffers.get(S).axpy( -alpha, buffers.get(V));

            double sNorm = buffers.get(S).norm();//Math.sqrt(dot(s, s, n));
            if (sNorm < tol) {
                //for (int i = 0; i < n; i++) x[i] += alpha * p[i];
                x.axpy(alpha, buffers.get(P));
                return k + 1;
            }

            A.apply(buffers.get(S), buffers.get(T));

            double tDotT = buffers.get(T).dot(buffers.get(T));//dot(t, t, n);
            omega = (tDotT == 0) ? 0 : buffers.get(T).dot(buffers.get(S)) / tDotT;// dot(t, s, n) / tDotT;
            if (omega == 0) break; // breakdown

            //for (int i = 0; i < n; i++) x[i] += alpha * p[i] + omega * s[i];
            x.axpy(alpha, buffers.get(P));
            x.axpy(omega, buffers.get(S));

            //for (int i = 0; i < n; i++) r[i] = s[i] - omega * t[i];
            buffers.get(R).clear();
            buffers.get(R).add(buffers.get(S));
            buffers.get(R).axpy(-omega, buffers.get(T));

            if (buffers.get(R).norm() < tol)
                return k + 1;

            rho = rhoNew;
        }

        return k;
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
     * @param fixedVariables boolean mask indicating fixed variables. A {@code true}
     *        entry means that the corresponding entry of {@code x} is prescribed
     *        and excluded from the solve
     * @param b right-hand side vector; one entry per equation
     * @param maxIter maximum number of BiCGSTAB iterations
     * @param tol convergence threshold on the residual norm {@code ||r||₂}
     * @param unknownIdx working index vector of size {@code n}; after initialization,
     *        {@code unknownIdx[equationIndex] = variableIndex}
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
    public static int solveConstrained(Matrix A, DoubleVector x, boolean[] fixedVariables, DoubleVector b, int maxIter, double tol,
                                       IntegerVector unknownIdx, WorkingBuffer<DoubleVector> nBuffer, WorkingBuffer<DoubleVector> mBuffer) {
        int n = A.rows();
        int m = A.cols();


        if (n > m)
            throw new IllegalArgumentException("BiCGSTAB constrained requires an under constrained matrix: " +
                    "rows=" + n + ", cols=" + m);

        if (x.size() != m)
            throw new IllegalArgumentException("x size (" + x.size() + ") != matrix columns (" + m + ")");

        if (b.size() != n)
            throw new IllegalArgumentException("b size (" + b.size() + ") != matrix rows (" + n + ")");

        if (nBuffer.vectorNumber() < 4 || nBuffer.get(CR).size() < n || nBuffer.get(CR0).size() < n
                || nBuffer.get(CV).size() < n || nBuffer.get(CT).size() < n)
            throw new IllegalArgumentException("Equation working buffers r/rHat0/v/t must have length >= " + n);

        if (mBuffer.vectorNumber() < 2 || mBuffer.get(CP).size() < m || mBuffer.get(CS).size() < m)
            throw new IllegalArgumentException("Variable working buffers p/s must have length >= " + m);

        if (unknownIdx.size() < n)
            throw new IllegalArgumentException("unknownIdx size (" + unknownIdx.size() + ") < matrix rows (" + n + ")");

        DoubleVector r = nBuffer.get(CR);
        DoubleVector rHat0 = nBuffer.get(CR0);
        DoubleVector v = nBuffer.get(CV);
        DoubleVector t = nBuffer.get(CT);

        DoubleVector p = mBuffer.get(CP);
        DoubleVector s = mBuffer.get(CS);

        int idx = 0;//todo THIS WILL NOT WORK ON GPU !!!
        for (int i = 0; i < m; i++) {
            if (!fixedVariables[i]) {
                //overload the method to choose bwn fixedVariables[] and unknownIdx. ->
                // given the memory access of such a methode you need to set on the cpu and send to the gpu in batch.

                //also add patern recognition for the fixedVariables : if the fixedVariables array is a multiple of m it should repeat itself.
                if (idx >= n)
                    throw new IllegalStateException("BiCGSTAB requires number of free variables == equations");
                unknownIdx.set(i, idx);
                idx++;
            }
        }

        p.clear();
        s.clear();

        // r = b - A*x ; rHat0 = r (fixed shadow residual, arbitrary choice)
        A.apply(x, v); // use v as temp for the initial residual

        r.clear();
        r.add(b);
        r.subtract(v);

        rHat0.clear();
        rHat0.add(r);

        p.clear();
        s.clear();
        v.clear();
        t.clear();

        double rho = 1, alpha = 1, omega = 1;

        if (r.norm() < tol) return 0;

        int k = 0;
        for (; k < maxIter; k++) {
            double rhoNew = rHat0.dot(r);
            if (rhoNew == 0) break; // breakdown: rHat0 orthogonal to r

            double beta = (rhoNew / rho) * (alpha / omega);

            p.skippedAxpy(-omega,v, unknownIdx, true, false);
            p.scale(beta); // fine for fixed direction since 0 * beta = 0
            p.skippedAdd(r, unknownIdx, true, false);


            A.apply(p, v);

            double rHat0v = v.dot(rHat0);
            if (rHat0v == 0) break; // breakdown
            alpha = rhoNew / rHat0v;

            s.skippedAxpy(-alpha, v, unknownIdx, true, false);
            s.skippedAdd(r, unknownIdx, true, false);

            double sNorm = Math.sqrt(s.skippedDot(s, unknownIdx, true, false));

            if (sNorm < tol) {
                x.skippedAxpy(alpha, p, unknownIdx, true, true);
                return k + 1;
            }

            A.apply(s, t);

            double tDotT = t.dot(t);
            omega = (tDotT == 0) ? 0 : s.skippedDot(t, unknownIdx, true, false) / tDotT;

            if (omega == 0) break; // breakdown

            if (r.norm() < tol) break;

            x.skippedAxpy(alpha, p, unknownIdx, true, true);
            x.skippedAxpy(omega, s, unknownIdx, true, true);

            r.skippedAdd(s, unknownIdx, false, true);
            r.axpy(-omega,t);

            if (r.norm() < tol)
                return k + 1;

            rho = rhoNew;
        }

        return k;
    }
}