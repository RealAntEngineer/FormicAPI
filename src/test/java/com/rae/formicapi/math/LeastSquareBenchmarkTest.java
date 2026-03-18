package com.rae.formicapi.math;

import com.rae.formicapi.math.operators.DenseMatrix;
import com.rae.formicapi.math.operators.DynamicCSRMatrix;
import com.rae.formicapi.math.operators.HashSparseMatrix;
import com.rae.formicapi.math.solvers.LeastSquare;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Benchmarks {@link LeastSquare#solve} performance across matrix implementations:
 * {@link DenseMatrix}, {@link HashSparseMatrix}, and {@link DynamicCSRMatrix}.
 *
 * <p>Tests scaling with problem size n for a fixed sparsity pattern
 * representative of a thermal nodal network (tridiagonal-like).
 *
 * <p>Not a correctness test — see {@link LeastSquareTest} for that.
 */
public class LeastSquareBenchmarkTest {

    private static final int WARMUP_REPS = 5;
    private static final int BENCH_REPS  = 10;
    private static final int MAX_ITER    = 1_000;
    private static final double TOL      = 1e-6;

    // ------------------------------------------------
    // Benchmark — scaling with n
    // ------------------------------------------------

    @Test
    public void benchmarkScalingWithN() {
        int[] sizes     = { 100, 500, 1_000, 2_000, 5_000 };
        int   nnzPerRow = 7;

        System.out.println("\n=== LeastSquare solve — scaling with n (nnzPerRow=" + fmt(nnzPerRow) + ") ===");
        System.out.printf("%-10s %-20s %-20s %-20s%n", "n", "Dense (ns)", "Hash (ns)", "CSR (ns)");
        System.out.println("-".repeat(70));

        for (int n : sizes) {
            double[]         b     = randomVector(n);
            double[]         x0    = new double[n];
            double[][]       raw   = randomSparse(n, n, nnzPerRow);

            DenseMatrix dense = toDense(raw, n, n);
            HashSparseMatrix hash  = toHash(raw, n, n);
            DynamicCSRMatrix csr   = toCSR(raw, n, n);

            long nsDense = benchmark(() -> LeastSquare.solve(dense, x0.clone(), b, MAX_ITER, TOL));
            long nsHash  = benchmark(() -> LeastSquare.solve(hash,  x0.clone(), b, MAX_ITER, TOL));
            long nsCSR   = benchmark(() -> LeastSquare.solve(csr,   x0.clone(), b, MAX_ITER, TOL));

            System.out.printf("%-10s %-20s %-20s %-20s%n",
                    fmt(n), fmt(nsDense), fmt(nsHash), fmt(nsCSR));
        }
    }

    // ------------------------------------------------
    // Benchmark — scaling with sparsity m
    // ------------------------------------------------

    @Test
    public void benchmarkScalingWithSparsity() {
        int   n           = 1_000;
        int[] densities   = { 3, 7, 15, 30, 60 , 120};

        System.out.println("\n=== LeastSquare solve — scaling with sparsity (n=" + fmt(n) + ") ===");
        System.out.printf("%-14s %-20s %-20s %-20s%n", "nnzPerRow", "Dense (ns)", "Hash (ns)", "CSR (ns)");
        System.out.println("-".repeat(74));

        for (int m : densities) {
            double[]         b     = randomVector(n);
            double[]         x0    = new double[n];
            double[][]       raw   = randomSparse(n, n, m);

            DenseMatrix      dense = toDense(raw, n, n);
            HashSparseMatrix hash  = toHash(raw, n, n);
            DynamicCSRMatrix csr   = toCSR(raw, n, n);

            long nsDense = benchmark(() -> LeastSquare.solve(dense, x0.clone(), b, MAX_ITER, TOL));
            long nsHash  = benchmark(() -> LeastSquare.solve(hash,  x0.clone(), b, MAX_ITER, TOL));
            long nsCSR   = benchmark(() -> LeastSquare.solve(csr,   x0.clone(), b, MAX_ITER, TOL));

            System.out.printf("%-14s %-20s %-20s %-20s%n",
                    fmt(m), fmt(nsDense), fmt(nsHash), fmt(nsCSR));
        }
    }

    // ------------------------------------------------
    // Helpers — matrix builders
    // ------------------------------------------------

    /** Generates a random sparse pattern as a dense 2D array */
    private static double[][] randomSparse(int rows, int cols, int nnzPerRow) {
        Random rng  = new Random(42);
        double[][] data = new double[rows][cols];

        for (int r = 0; r < rows; r++) {
            // always put something on the diagonal for conditioning
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
                if (raw[r][c] != 0.0) m.add(r, c, raw[r][c]);
        m.multiply(new double[cols], new double[rows]); // force compile
        return m;
    }

    // ------------------------------------------------
    // Helpers — benchmark runner
    // ------------------------------------------------

    private long benchmark(Runnable task) {
        for (int i = 0; i < WARMUP_REPS; i++) task.run();
        long start = System.nanoTime();
        for (int i = 0; i < BENCH_REPS; i++) task.run();
        return System.nanoTime() - start;
    }

    private static double[] randomVector(int n) {
        Random   rng = new Random(7);
        double[] v   = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble();
        return v;
    }

    // ------------------------------------------------
    // Helpers — formatting
    // ------------------------------------------------

    private static String fmt(long n) {
        return String.format("%,d", n).replace(',', ' ');
    }

    private static String fmt(int n) {
        return String.format("%,d", n).replace(',', ' ');
    }
}