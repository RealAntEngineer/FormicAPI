package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.linear.DenseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.DynamicCSRMatrix;
import com.rae.formicapi.foundation.math.operators.linear.HashSparseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.solvers.LeastSquare;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(2)
public class LeastSquareBenchmark {

    private static final float TOL = 1e-6f;

    @Param({"10", "50", "100", "1000"})
    private int n;

    @Param({"5", "25"})
    private int nnzPerRow;

    @Param({"100"})
    private int maxIter;

    private double[]         b;
    private PaddedCSRMatrix  paddedCSR;
    private HashSparseMatrix hash;
    private DynamicCSRMatrix csr;

    @Setup(Level.Trial)
    public void setup() {
        b = randomVector(n);

        double[][] raw = randomSparse(n, n, nnzPerRow);

        paddedCSR = toPaddedCSR(raw, n, n, nnzPerRow);
        hash = toHash(raw, n, n);
        csr = toCSR(raw, n, n);
    }

    // ------------------------------------------------
    // Counters
    // ------------------------------------------------

    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class Counters {

        /**
         * Number of solver iterations performed by the last
         * benchmark invocation.
         *
         * JMH reports this as an auxiliary operation count.
         */
        public long solverIterations;

        @Setup(Level.Invocation)
        public void reset() {
            solverIterations = 0;
        }
    }

    // ------------------------------------------------
    // Benchmarks
    // ------------------------------------------------

    @Benchmark
    public void solvePaddedCSR(Counters counters, Blackhole bh) {
        double[] x = new double[n];

        int iterations = LeastSquare.solve(paddedCSR, b, x,maxIter, TOL);

        counters.solverIterations = iterations;
        bh.consume(x);
    }

    @Benchmark
    public void solveHash(Counters counters, Blackhole bh) {
        double[] x = new double[n];

        int iterations = LeastSquare.solve(hash,  b,x, maxIter, TOL);

        counters.solverIterations = iterations;
        bh.consume(x);
    }

    @Benchmark
    public void solveCSR(Counters counters, Blackhole bh) {
        double[] x = new double[n];

        int iterations = LeastSquare.solve(csr, b, x, maxIter, TOL);

        counters.solverIterations = iterations;
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

            while (chosen.size() < Math.min(nnzPerRow, cols)) {
                chosen.add(rng.nextInt(cols));
            }

            for (int c : chosen) {
                if (c != r) {
                    data[r][c] = rng.nextDouble();
                }
            }
        }

        return data;
    }

    private static PaddedCSRMatrix toPaddedCSR(
            double[][] raw, int rows, int cols, int nnzPerRow) {

        PaddedCSRMatrix m = new PaddedCSRMatrix(rows, cols, nnzPerRow);

        for (int r = 0; r < rows; r++) {
            double[] newValues = new double[nnzPerRow];
            int[]    newCols   = new int[nnzPerRow];
            int      count     = 0;

            for (int c = 0; c < cols; c++) {
                if (raw[r][c] != 0.0) {
                    if (count >= nnzPerRow) {
                        throw new IllegalArgumentException(
                                "Row " + r + " has more than " + nnzPerRow + " non-zero values");
                    }

                    newValues[count] = raw[r][c];
                    newCols[count] = c;
                    count++;
                }
            }

            m.setRow(r, newValues, newCols, count);
        }

        return m;
    }

    private static HashSparseMatrix toHash(double[][] raw, int rows, int cols) {
        HashSparseMatrix m = new HashSparseMatrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (raw[r][c] != 0.0) {
                    m.set(r, c, raw[r][c]);
                }
            }
        }

        return m;
    }

    private static DynamicCSRMatrix toCSR(double[][] raw, int rows, int cols) {
        DynamicCSRMatrix m = new DynamicCSRMatrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (raw[r][c] != 0.0) {
                    m.set(r, c, raw[r][c]);
                }
            }
        }

        // Force/finalize CSR internal representation if required.
        m.multiply(new double[cols], new double[rows]);

        return m;
    }

    private static double[] randomVector(int n) {
        Random rng = new Random(7);
        double[] v = new double[n];

        for (int i = 0; i < n; i++) {
            v[i] = rng.nextDouble();
        }

        return v;
    }
}