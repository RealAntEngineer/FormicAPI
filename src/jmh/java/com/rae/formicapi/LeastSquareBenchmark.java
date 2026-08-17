package com.rae.formicapi;


import com.rae.formicapi.foundation.math.operators.linear.DenseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.DynamicCSRMatrix;
import com.rae.formicapi.foundation.math.operators.linear.HashSparseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.solvers.LeastSquare;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.*;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(2)
public class LeastSquareBenchmark {

    private static final int MAX_ITER = 1_000;
    private static final float TOL = 1e-6f;

    @Param({"100", "500", "1000"})
    private int n;

    @Param({"5", "25", "50"})
    private int nnzPerRow;

    private double[]         b;
    private double[]         x0;
    private DenseMatrix      dense;
    private HashSparseMatrix hash;
    private DynamicCSRMatrix csr;

    @Setup(Level.Trial)
    public void setup() {
        b = randomVector(n);
        x0 = new double[n];
        double[][] raw = randomSparse(n, n, nnzPerRow);

        dense = toDense(raw, n, n);
        hash = toHash(raw, n, n);
        csr = toCSR(raw, n, n);
    }

    @Benchmark
    public void solveDense(Blackhole bh) {
        double[] x = x0.clone();
        LeastSquare.solve(dense, x, b, MAX_ITER, TOL);
        bh.consume(x);
    }

    @Benchmark
    public void solveHash(Blackhole bh) {
        double[] x = x0.clone();
        LeastSquare.solve(hash, x, b, MAX_ITER, TOL);
        bh.consume(x);
    }

    @Benchmark
    public void solveCSR(Blackhole bh) {
        double[] x = x0.clone();
        LeastSquare.solve(csr, x, b, MAX_ITER, TOL);
        bh.consume(x);
    }

    // ------------------------------------------------
    // Helpers — matrix builders
    // ------------------------------------------------

    private static double[][] randomSparse(int rows, int cols, int nnzPerRow) {
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
        m.multiply(new double[cols], new double[rows]);
        return m;
    }

    private static double[] randomVector(int n) {
        Random rng = new Random(7);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble();
        return v;
    }
}