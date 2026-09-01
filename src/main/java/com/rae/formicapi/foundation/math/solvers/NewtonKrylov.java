package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuIntegerVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;

/**
 * Newton-Krylov solver for nonlinear systems of the form
 *
 * <pre>
 *     C:x:x + A*x - b = 0
 * </pre>
 * <p>
 * where {@code C} is represented by a {@link PaddedCSR3Tensor} and {@code A}
 * is a linear {@link Matrix}.
 *
 * <p>Each Newton step solves {@code J(x) dx = -F(x)} where
 * {@code F(x) = C:x:x + A*x - b}, via matrix-free BiCGSTAB using
 *
 * <pre>
 *     J(x)v = d(C:x:x)/dx * v + A*v = C.multiplyJacobian(x, v) + A.apply(v)
 * </pre>
 *
 * <p>{@code dx} is always the correction, not the state — it starts at
 * {@code 0} on every Newton iteration and {@code x += dx} is applied by the
 * caller-visible {@link #solve}/{@link #solveConstrained} methods, never by
 * {@code newtonStep} itself.
 *
 * <h2>Rectangular systems</h2>
 * <p>Like {@link BiCGStab}, this solver allows {@code A}/{@code C} to be
 * genuinely rectangular: {@code n = outputSize (A.outputSize(), C.equations())}
 * equations over {@code m = inputSize (A.inputSize(), C.inputSize())}
 * variables, with {@code n <= m}. {@link #solveConstrained} builds
 * {@code unknownIdx} — size {@code n}, mapping <em>equation index -> free
 * variable index</em> — via {@link Util#fillUnknowIdx}, exactly as
 * {@link BiCGStab#solveConstrained} does. This is real row reduction (rows
 * for fixed/prescribed DOFs are never assembled), not a size-{@code m} mask.
 * {@link #solve} is the {@code n == m}, {@code unknownIdx == null} special
 * case. Every {@code skippedAxpy}/{@code skippedAdd}/{@code skippedDot} call
 * below uses the same flag convention as {@link BiCGStab}: a vector flagged
 * {@code true} is variable-space (size {@code m}, accessed through
 * {@code unknownIdx[j]}), flagged {@code false} is equation-space (size
 * {@code n}, accessed directly at the loop index {@code j}). {@code p},
 * {@code s}, and {@code dx} are variable-space; {@code r}, {@code r0},
 * {@code v}, and {@code t} are equation-space and therefore never need
 * masking against each other (e.g. {@code r.axpy(-omega, t)} and
 * {@code r.norm()} are always plain, unskipped calls).
 *
 * <h2>Scaling</h2>
 * <p>{@code scaling}, when supplied, lives in <b>equation (output) space</b>
 * — size {@code n} — and is applied as a diagonal left preconditioner on the
 * residual: the linear step actually solves
 * {@code diag(scaling)*J(x)*dx = -diag(scaling)*F(x)}. Concretely, every
 * equation-space quantity gets {@code .scale(scaling)} applied once, right
 * after it's produced: {@link #residual} scales {@code F} in place (so
 * {@code F} is the <em>scaled</em> residual from that point on — both the
 * outer convergence check and the {@code r = -F} seed for the inner solve
 * use this same scaled value, no second scaling), and {@code v}/{@code t}
 * are scaled immediately after each {@link #jacobian} call. Variable-space
 * quantities ({@code p}, {@code s}, {@code dx}) are never scaled — scaling
 * only ever touches equation-space buffers. Pass {@code scaling = null} to
 * skip preconditioning entirely.
 *
 * <p>TODO: replace the separate {@code A}/{@code C} parameters with a single
 * compound operator once one exists, so this stops needing to know it's
 * summing exactly a linear and a quadratic term.
 */
public class NewtonKrylov {

    // Equation-space buffers (size n = A.outputSize() = C.equations()).
    private static final int RES            = 0; // F(x), the nonlinear residual -- scaled in place once scaling is applied
    private static final int TMP            = 1; // scratch for residual assembly (A*x before combining with C:x:x)
    private static final int CTERM          = 2; // scratch: C-part of a Jacobian-vector product
    private static final int R              = 3;
    private static final int R0             = 4;
    private static final int V              = 5;
    private static final int T              = 6;
    private static final int BUFFER_COUNT_N = 7;
    // Variable-space buffers (size m = A.inputSize() = C.inputSize()).
    private static final int P              = 0;
    private static final int S              = 1;
    private static final int DX             = 2; // Newton correction accumulator
    private static final int BUFFER_COUNT_M = 3;

    /**
     * Solve with a zero initial guess is not offered on purpose — a cold
     * start (x = 0) is rarely a useful Newton iterate for these systems.
     * Callers warm-start from the previous tick's solution.
     */
    public static double[] solve(PaddedCSR3Tensor C, Matrix A, double[] x, double[] b,
                                 int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol) {
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, null, null);
    }

    // =====================================================================
    // double[] convenience wrappers
    // =====================================================================

    public static double[] solve(PaddedCSR3Tensor C, Matrix A, double[] x, double[] b,
                                 int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol,
                                 double @Nullable [] scaling, @Nullable Stats stats) {
        DoubleVector xv       = new CpuDoubleVector(x);
        DoubleVector scalingV = scaling == null ? null : new CpuDoubleVector(scaling);
        solve(C, A, xv, new CpuDoubleVector(b), maxNewtonIter, maxLinearIter, newtonTol, linearTol, scalingV, stats);
        return ((CpuDoubleVector) xv).array();
    }

    /**
     * Unconstrained solve, allocating its working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static DoubleVector solve(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b,
                                     int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol,
                                     @Nullable DoubleVector scaling, @Nullable Stats stats) {
        int n = A.outputSize();
        int m = A.inputSize();
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, scaling, stats, null,
                new WorkingBuffer<>(BUFFER_COUNT_N, () -> new CpuDoubleVector(n)),
                new WorkingBuffer<>(BUFFER_COUNT_M, () -> new CpuDoubleVector(m)));
    }

    /**
     * Core solve. Unconstrained when {@code unknownIdx == null} (requires
     * {@code A.outputSize() == A.inputSize()}); constrained/rectangular
     * otherwise (requires {@code A.outputSize() <= A.inputSize()} and
     * {@code unknownIdx.size() >= A.outputSize()}).
     *
     * <p>{@code x} is both the initial guess and the solution — mutated in
     * place, same instance returned. Entries of {@code x} not reachable
     * through {@code unknownIdx} (fixed/prescribed DOFs) are never modified.
     *
     * @param bufferN working buffer providing {@value #BUFFER_COUNT_N} vectors of size {@code n = A.outputSize()}
     * @param bufferM working buffer providing {@value #BUFFER_COUNT_M} vectors of size {@code m = A.inputSize()}
     */
    public static DoubleVector solve(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b,
                                     int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol,
                                     @Nullable DoubleVector scaling, @Nullable Stats stats,
                                     @Nullable IntegerVector unknownIdx, WorkingBuffer<DoubleVector> bufferN, WorkingBuffer<DoubleVector> bufferM) {
        int     n           = A.outputSize();
        int     m           = A.inputSize();
        boolean constrained = unknownIdx != null;

        if (C.equations() != n)
            throw new IllegalArgumentException("C.equations() (" + C.equations() + ") != A.outputSize() (" + n + ")");
        if (C.inputSize() != m)
            throw new IllegalArgumentException("C.inputSize() (" + C.inputSize() + ") != A.inputSize() (" + m + ")");
        if (!constrained && n != m)
            throw new IllegalArgumentException("NewtonKrylov unconstrained requires a square system: outputSize=" + n + ", inputSize=" + m);
        if (n > m)
            throw new IllegalArgumentException("NewtonKrylov requires outputSize <= inputSize: outputSize=" + n + ", inputSize=" + m);
        if (x.size() != m)
            throw new IllegalArgumentException("x size (" + x.size() + ") != inputSize (" + m + ")");
        if (b.size() != n)
            throw new IllegalArgumentException("b size (" + b.size() + ") != outputSize (" + n + ")");
        if (scaling != null && scaling.size() != n)
            throw new IllegalArgumentException("scaling size (" + scaling.size() + ") != outputSize (" + n + ") -- scaling lives in equation space");
        if (constrained && unknownIdx.size() < n)
            throw new IllegalArgumentException("unknownIdx size (" + unknownIdx.size() + ") < outputSize (" + n + ")");
        if (bufferN.vectorNumber() < BUFFER_COUNT_N)
            throw new IllegalArgumentException("NewtonKrylov requires an equation-space working buffer with >= " + BUFFER_COUNT_N + " vectors");
        if (bufferM.vectorNumber() < BUFFER_COUNT_M)
            throw new IllegalArgumentException("NewtonKrylov requires a variable-space working buffer with >= " + BUFFER_COUNT_M + " vectors");
        for (int i = 0; i < BUFFER_COUNT_N; i++)
            if (bufferN.get(i).size() < n)
                throw new IllegalArgumentException("Equation-space working buffer vector " + i + " has size < " + n);
        for (int i = 0; i < BUFFER_COUNT_M; i++)
            if (bufferM.get(i).size() < m)
                throw new IllegalArgumentException("Variable-space working buffer vector " + i + " has size < " + m);

        DoubleVector F = bufferN.get(RES), tmp = bufferN.get(TMP), dx = bufferM.get(DX);

        for (int iteration = 0; iteration < maxNewtonIter; iteration++) {
            double residualNorm = residual(C, A, x, b, F, tmp, scaling);

            if (!Double.isFinite(residualNorm))
                throw new ArithmeticException(
                        "NewtonKrylov: residual became non-finite at Newton iteration " + iteration
                                + " (norm=" + residualNorm + "). The linear solve likely diverged/broke down"
                                + " on the previous step -- check conditioning rather than continuing.");

            if (stats != null) {
                stats.newtonResiduals.add(residualNorm);
                stats.newtonIterationCount = iteration + 1;
            }

            if (residualNorm < newtonTol) {
                if (stats != null) stats.converged = true;
                return x;
            }

            int linearIterations = newtonStep(C, A, x, F, unknownIdx, scaling, dx, maxLinearIter, linearTol, bufferN, bufferM);

            if (stats != null) stats.linearIterations.add(linearIterations);

            x.add(dx);
        }
        return x;
    }

    // =====================================================================
    // DoubleVector API
    // =====================================================================

    /**
     * {@code F = C:x:x + A*x - b}, written into {@code F} (using {@code tmp}
     * as scratch). When {@code scaling != null}, {@code F} is left holding
     * {@code scaling ⊙ F} — see the scaling section of the class javadoc.
     * Returns {@code ||F||2} (of the possibly-scaled {@code F}).
     */
    private static double residual(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b,
                                   DoubleVector F, DoubleVector tmp, @Nullable DoubleVector scaling) {
        C.apply(x, F);
        A.apply(x, tmp);
        F.add(tmp);
        F.subtract(b);
        if (scaling != null) F.scale(scaling);
        return F.norm();
    }

    /**
     * One Newton linearization: solve {@code J(x)*dx = -F(x)} via matrix-free
     * BiCGSTAB. {@code F} is assumed already scaled (see {@link #residual}).
     * Structure mirrors {@link BiCGStab#solve} exactly, with
     * {@code A.apply(direction, out)} replaced by {@link #jacobian} (plus an
     * explicit {@code .scale(scaling)} on the result, since {@code jacobian}
     * itself is scaling-agnostic), and {@code dx} standing in for
     * {@code BiCGStab}'s {@code x} — it starts at {@code 0} instead of a
     * warm start, so no jacobian call is needed to seed {@code r}.
     */
    private static int newtonStep(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector F,
                                  @Nullable IntegerVector unknownIdx, @Nullable DoubleVector scaling, DoubleVector dx,
                                  int maxIter, double tol, WorkingBuffer<DoubleVector> bufferN, WorkingBuffer<DoubleVector> bufferM) {
        boolean constrained       = unknownIdx != null;
        boolean scalePrecondition = scaling != null;

        DoubleVector r     = bufferN.get(R), r0 = bufferN.get(R0), v = bufferN.get(V), t = bufferN.get(T);
        DoubleVector cTerm = bufferN.get(CTERM);
        DoubleVector p     = bufferM.get(P), s = bufferM.get(S);

        dx.clear();
        p.clear();
        s.clear();

        // r = -F - J(x)*dx ; dx == 0 initially, so r = -F. F is already
        // scaled (see residual()), so no further scaling here.
        r.clear();
        r.subtract(F);

        r0.clear();
        r0.add(r);

        v.clear();

        double rho = 1, alpha = 1, omega = 1;

        if (r.norm() < tol) return 0;

        int k = 0;
        for (; k < maxIter; k++) {
            double rhoNew = r0.dot(r);
            if (rhoNew == 0) break; // breakdown: rHat0 orthogonal to r

            double beta = (rhoNew / rho) * (alpha / omega);

            if (constrained) {
                p.skippedAxpy(-omega, v, unknownIdx, true, false);
                p.scale(beta);
                p.skippedAdd(r, unknownIdx, true, false);
            } else {
                p.axpy(-omega, v);
                p.scale(beta);
                p.add(r);
            }

            jacobian(C, A, x, p, cTerm, v);
            if (scalePrecondition) v.scale(scaling);

            double r0v = r0.dot(v);
            if (r0v == 0) break; // breakdown
            alpha = rhoNew / r0v;

            s.clear();
            if (constrained) {
                s.skippedAxpy(-alpha, v, unknownIdx, true, false);
                s.skippedAdd(r, unknownIdx, true, false);
            } else {
                s.add(r);
                s.axpy(-alpha, v);
            }

            double sNorm = constrained ? Math.sqrt(s.skippedDot(s, unknownIdx, true, true)) : s.norm();
            if (sNorm < tol) {
                if (constrained) dx.skippedAxpy(alpha, p, unknownIdx, true, true);
                else dx.axpy(alpha, p);
                return k + 1;
            }

            jacobian(C, A, x, s, cTerm, t);
            if (scalePrecondition) t.scale(scaling);

            double tDotT = t.dot(t);
            if (tDotT == 0) break;

            omega = (constrained ? s.skippedDot(t, unknownIdx, true, false) : t.dot(s)) / tDotT;
            if (omega == 0) break; // breakdown

            r.clear();

            if (constrained) {
                dx.skippedAxpy(alpha, p, unknownIdx, true, true);
                dx.skippedAxpy(omega, s, unknownIdx, true, true);
                r.skippedAdd(s, unknownIdx, false, true);
            } else {
                dx.axpy(alpha, p);
                dx.axpy(omega, s);
                r.add(s);
            }

            r.axpy(-omega, t);

            if (r.norm() < tol) return k + 1;

            rho = rhoNew;
        }

        return k;
    }

    /**
     * {@code J(x)*direction = C.multiplyJacobian(x, direction) + A.apply(direction)},
     * written into {@code out}. Scaling is applied by the caller, not here —
     * see the scaling section of the class javadoc.
     */
    private static void jacobian(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector direction,
                                 DoubleVector cTerm, DoubleVector out) {
        C.multiplyJacobian(x, direction, cTerm);
        A.apply(direction, out);
        out.add(cTerm);
    }

    public static double[] solveConstrained(PaddedCSR3Tensor C, Matrix A, double[] x, boolean[] fixedVariables, double[] b,
                                            int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol,
                                            @Nullable double[] scaling) {
        return solveConstrained(C, A, x, fixedVariables, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, scaling, null);
    }

    // =====================================================================
    // Internals
    // =====================================================================

    public static double[] solveConstrained(PaddedCSR3Tensor C, Matrix A, double[] x, boolean[] fixedVariables, double[] b,
                                            int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol,
                                            @Nullable double[] scaling, @Nullable Stats stats) {
        DoubleVector xv       = new CpuDoubleVector(x);
        DoubleVector scalingV = scaling == null ? null : new CpuDoubleVector(scaling);

        IntegerVector unknownIdx = new CpuIntegerVector(A.outputSize());
        Util.fillUnknowIdx(fixedVariables, unknownIdx, A.inputSize(), A.outputSize());

        solveConstrained(C, A, xv, new CpuDoubleVector(b), maxNewtonIter, maxLinearIter,
                newtonTol, linearTol, scalingV, stats, unknownIdx);
        return ((CpuDoubleVector) xv).array();
    }

    /**
     * Constrained solve from a pre-built equation -> free-variable map,
     * allocating its working buffers internally.
     * Use only outside hot paths — prefer the pre-allocated overload if you will solve repeatedly.
     */
    public static DoubleVector solveConstrained(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b,
                                                int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol,
                                                @Nullable DoubleVector scaling, @Nullable Stats stats, IntegerVector unknownIdx) {
        int n = C.equations();
        int m = C.inputSize();
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, scaling, stats, unknownIdx,
                new WorkingBuffer<>(BUFFER_COUNT_N, () -> new CpuDoubleVector(n)),
                new WorkingBuffer<>(BUFFER_COUNT_M, () -> new CpuDoubleVector(m)));
    }

    public static DoubleVector solve(PaddedCSR3Tensor C, Matrix A, DoubleVector x, DoubleVector b,
                                     int maxNewtonIter, int maxLinearIter, double newtonTol, double linearTol) {
        return solve(C, A, x, b, maxNewtonIter, maxLinearIter, newtonTol, linearTol, null, null);
    }

    /**
     * Optional convergence diagnostics for a {@link #solve}/{@link #solveConstrained}
     * call. Pass {@code null} anywhere a {@code Stats} parameter is accepted to skip
     * tracking entirely (zero overhead).
     *
     * <p>Not thread-safe; use one instance per solve. When {@code scaling} is
     * supplied, {@link #newtonResiduals} are reported in scaled units (see
     * the scaling section of the class javadoc).
     */
    public static final class Stats {

        /**
         * ||F(x)||2 at the start of each Newton iteration, in order.
         */
        public final List<Double> newtonResiduals = new ArrayList<>();

        /**
         * BiCGSTAB iterations actually used per Newton step (parallel to {@link #newtonResiduals}).
         */
        public final List<Integer> linearIterations = new ArrayList<>();

        /**
         * Total Newton iterations performed. Equal to {@code newtonResiduals.size()}.
         */
        public int newtonIterationCount;

        /**
         * Whether the solve converged (residual < newtonTol) before exhausting maxNewtonIter.
         */
        public boolean converged;

        /**
         * Clears all recorded data so the same Stats instance can be reused across solves.
         */
        public void reset() {
            newtonResiduals.clear();
            linearIterations.clear();
            newtonIterationCount = 0;
            converged = false;
        }
    }
}