package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.linear.DenseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.DynamicCSRMatrix;
import com.rae.formicapi.foundation.math.operators.linear.HashSparseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR2Tensor;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Benchmarks {@link NewtonKrylov#solve} scaling with matrix backend (Dense /
 * Hash / CSR) for the linear term {@code A}, at varying problem size {@code n}
 * and sparsity {@code nnzPerRow}.
 *
 * <p>The nonlinear system solved is
 *
 * <pre>
 *     x_i^2 + (A*x)_i - b_i = 0    for i = 0..n-1
 * </pre>
 *
 * with {@code b} constructed from a known {@code expected} solution so every
 * backend converges to the same fixed point regardless of {@code n} or
 * {@code nnzPerRow}. The tensor {@code C} (diagonal x_i^2 term) is identical
 * across all three benchmarks so the comparison isolates {@code A}'s backend,
 * not the tensor contraction.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(2)
public class NewtonKrylovBenchmark {

    private static final int MAX_NEWTON_ITER = 50;
    private static final int MAX_LINEAR_ITER = 200;
    private static final double NEWTON_TOL = 1e-8;
    private static final double LINEAR_TOL = 1e-10;

    @Param({"50", "100", "200"})
    private int n;

    @Param({"5", "15", "30"})
    private int nnzPerRow;

    private PaddedCSR2Tensor C;
    private double[] b;
    private double[] x0;
    private DenseMatrix dense;
    private HashSparseMatrix hash;
    private DynamicCSRMatrix csr;

    @Setup(Level.Trial)
    public void setup() {
        // Diagonal quadratic term x_i^2, identical across all three backends —
        // isolates the benchmark to A's multiply implementation.
        PaddedCSR2Tensor tensor = new PaddedCSR2Tensor(n, 1);
        for (int i = 0; i < n; i++)
            tensor.setRow(i, new double[]{1.0}, new int[]{i}, new int[]{i}, 1);
        C = tensor;

        double[][] raw = randomDiagonallyDominantSparse(n, n, nnzPerRow);
        dense = toDense(raw, n, n);
        hash = toHash(raw, n, n);
        csr = toCSR(raw, n, n);

        double[] expected = randomVector(n);
        for (int i = 0; i < n; i++)
            expected[i] *= 0.5; // keep magnitudes modest so Newton converges reliably

        double[] cxx = new double[n];
        C.multiply(expected, cxx);
        double[] ax = new double[n];
        dense.multiply(expected, ax);

        b = new double[n];
        for (int i = 0; i < n; i++)
            b[i] = cxx[i] + ax[i];

        x0 = new double[n]; // zero initial guess
    }

    @Benchmark
    public void solveDense(Blackhole bh) {
        double[] x = x0.clone();
        NewtonKrylov.solve(C, dense, x, b, MAX_NEWTON_ITER, MAX_LINEAR_ITER, NEWTON_TOL, LINEAR_TOL);
        bh.consume(x);
    }

    @Benchmark
    public void solveHash(Blackhole bh) {
        double[] x = x0.clone();
        NewtonKrylov.solve(C, hash, x, b, MAX_NEWTON_ITER, MAX_LINEAR_ITER, NEWTON_TOL, LINEAR_TOL);
        bh.consume(x);
    }

    @Benchmark
    public void solveCSR(Blackhole bh) {
        double[] x = x0.clone();
        NewtonKrylov.solve(C, csr, x, b, MAX_NEWTON_ITER, MAX_LINEAR_ITER, NEWTON_TOL, LINEAR_TOL);
        bh.consume(x);
    }

    // ------------------------------------------------
    // Helpers — matrix builders
    // ------------------------------------------------

    private static double[][] randomDiagonallyDominantSparse(int rows, int cols, int nnzPerRow) {
        Random rng = new Random(42);
        double[][] data = new double[rows][cols];

        for (int r = 0; r < rows; r++) {
            data[r][r] = 10.0;
            Set<Integer> chosen = new HashSet<>();
            chosen.add(r);
            while (chosen.size() < Math.min(nnzPerRow, cols))
                chosen.add(rng.nextInt(cols));
            for (int c : chosen)
                if (c != r) data[r][c] = rng.nextDouble();
        }
        return data;
    }

    private static DenseMatrix toDense(double[][] raw, int rows, int cols) {
        DenseMatrix m = new DenseMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (raw[r][c] != 0.0) m.set(r, c, raw[r][c]);
        return m;
    }

    private static HashSparseMatrix toHash(double[][] raw, int rows, int cols) {
        HashSparseMatrix m = new HashSparseMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (raw[r][c] != 0.0) m.set(r, c, raw[r][c]);
        return m;
    }

    private static DynamicCSRMatrix toCSR(double[][] raw, int rows, int cols) {
        DynamicCSRMatrix m = new DynamicCSRMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (raw[r][c] != 0.0) m.set(r, c, raw[r][c]);
        m.multiply(new double[cols], new double[rows]); // force CSR compile
        return m;
    }

    private static double[] randomVector(int n) {
        Random rng = new Random(7);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble();
        return v;
    }
}