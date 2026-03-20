package com.rae.formicapi.math_tests;

import com.rae.formicapi.fondation.math.operators.CSRMatrix;
import com.rae.formicapi.fondation.math.operators.DynamicCSRMatrix;
import com.rae.formicapi.fondation.math.operators.HashSparseMatrix;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Benchmarks {@link CSRMatrix#transposeMultiply} and {@link CSRMatrix#multiply}
 * scaling with respect to matrix size (n) and non-zeros per row (m).
 *
 * <p>Not a correctness test — validates that wall-clock time grows linearly
 * with nnz = n * m as expected for O(nnz) operations.
 */
public class CSRMatrixScalingTest {

    private static final int WARMUP_REPS = 5;
    private static final int BENCH_REPS = 20;
    private static final int SLACK = 10;

    // ------------------------------------------------
    // Scale with n (rows), fixed sparsity
    // ------------------------------------------------

    @Test
    public void testMultiplyScalesWithN() {
        int[] sizes = {100, 500, 1000, 5000, 10000};
        int nnzPerRow = 10;

        printHeader("multiply — scaling with n", "nnzPerRow", nnzPerRow);
        printColumns("n", "time (ns)", "ns/nnz");

        long prevNs = -1;
        for (int n : sizes) {
            CSRMatrix A = randomCSR(n, n, nnzPerRow);
            double[] x = randomVector(n);
            double[] result = new double[n];

            long ns = benchmark(() -> A.multiply(x, result));
            long nnz = (long) n * nnzPerRow;
            printRow(n, ns, nnz);

            if (prevNs > 0)
                assertTrue(ns < prevNs * SLACK, "multiply time grew faster than expected");
            prevNs = ns;
        }
    }

    private static void printHeader(String title, String paramName, int paramValue) {
        System.out.printf("%n=== %s (%s=%s) ===%n", title, paramName, fmt(paramValue));
    }

    // ------------------------------------------------
    // Scale with m (nnz per row), fixed n
    // ------------------------------------------------

    private static void printColumns(String col1, String col2, String col3) {
        System.out.printf("%-14s %-20s %-18s%n", col1, col2, col3);
        System.out.println("-".repeat(52));
    }

    private static CSRMatrix randomCSR(int rows, int cols, int nnzPerRow) {
        Random rng = new Random(42);
        DynamicCSRMatrix m = new DynamicCSRMatrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            Set<Integer> chosen = new HashSet<>();
            while (chosen.size() < Math.min(nnzPerRow, cols))
                chosen.add(rng.nextInt(cols));
            for (int c : chosen)
                m.add(r, c, rng.nextDouble());
        }

        m.multiply(new double[cols], new double[rows]); // force CSR compile
        return m.toCSR();
    }

    // ------------------------------------------------
    // CSR vs default transposeMultiply comparison
    // ------------------------------------------------

    private static double[] randomVector(int n) {
        Random rng = new Random(7);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble();
        return v;
    }

    // ------------------------------------------------
    // Helpers — printing
    // ------------------------------------------------

    private long benchmark(Runnable task) {
        for (int i = 0; i < WARMUP_REPS; i++) task.run();
        long start = System.nanoTime();
        for (int i = 0; i < BENCH_REPS; i++) task.run();
        return System.nanoTime() - start;
    }

    private static void printRow(int key, long ns, long nnz) {
        System.out.printf("%-14s %-20s %-18s%n",
                fmt(key), fmt(ns), fmtRatio((double) ns / nnz));
    }

    private static String fmt(int n) {
        return String.format("%,d", n).replace(',', ' ');
    }

    private static String fmt(long n) {
        return String.format("%,d", n).replace(',', ' ');
    }

    private static String fmtRatio(double nsPerNnz) {
        return String.format("%,.3f", nsPerNnz).replace(',', ' ');
    }

    @Test
    public void testTransposeMultiplyScalesWithN() {
        int[] sizes = {100, 500, 1000, 5000, 10000};
        int nnzPerRow = 10;

        printHeader("transposeMultiply — scaling with n", "nnzPerRow", nnzPerRow);
        printColumns("n", "time (ns)", "ns/nnz");

        long prevNs = -1;
        for (int n : sizes) {
            CSRMatrix A = randomCSR(n, n, nnzPerRow);
            double[] x = randomVector(n);
            double[] result = new double[n];

            long ns = benchmark(() -> A.transposeMultiply(x, result));
            long nnz = (long) n * nnzPerRow;
            printRow(n, ns, nnz);

            if (prevNs > 0)
                assertTrue(ns < prevNs * SLACK, "transposeMultiply time grew faster than expected");
            prevNs = ns;
        }
    }

    // ------------------------------------------------
    // Helpers — data
    // ------------------------------------------------

    @Test
    public void testMultiplyScalesWithM() {
        int n = 2000;
        int[] densities = {1, 5, 10, 50, 100};

        printHeader("multiply — scaling with m", "n", n);
        printColumns("nnzPerRow", "time (ns)", "ns/nnz");

        long prevNs = -1;
        for (int m : densities) {
            CSRMatrix A = randomCSR(n, n, m);
            double[] x = randomVector(n);
            double[] result = new double[n];

            long ns = benchmark(() -> A.multiply(x, result));
            long nnz = (long) n * m;
            printRow(m, ns, nnz);

            if (prevNs > 0)
                assertTrue(ns < prevNs * SLACK, "multiply time grew faster than expected with m");
            prevNs = ns;
        }
    }

    @Test
    public void testTransposeMultiplyScalesWithM() {
        int n = 2000;
        int[] densities = {1, 5, 10, 50, 100};

        printHeader("transposeMultiply — scaling with m", "n", n);
        printColumns("nnzPerRow", "time (ns)", "ns/nnz");

        long prevNs = -1;
        for (int m : densities) {
            CSRMatrix A = randomCSR(n, n, m);
            double[] x = randomVector(n);
            double[] result = new double[n];

            long ns = benchmark(() -> A.transposeMultiply(x, result));
            long nnz = (long) n * m;
            printRow(m, ns, nnz);

            if (prevNs > 0)
                assertTrue(ns < prevNs * SLACK, "transposeMultiply time grew faster than expected with m");
            prevNs = ns;
        }
    }

    @Test
    public void testCSRFasterThanDefault() {
        int n = 3000;
        int nnzPerRow = 20;

        CSRMatrix csr = randomCSR(n, n, nnzPerRow);
        HashSparseMatrix hash = toHash(csr);
        double[] x = randomVector(n);
        double[] result = new double[n];

        long nsCSR = benchmark(() -> csr.transposeMultiply(x, result));
        long nsHash = benchmark(() -> hash.transposeMultiply(x, result));

        System.out.printf("%n=== transposeMultiply: CSR vs default (n=%d, nnzPerRow=%d) ===%n", n, nnzPerRow);
        System.out.printf("  CSR     : %s ns%n", fmt(nsCSR));
        System.out.printf("  Default : %s ns%n", fmt(nsHash));
        System.out.printf("  Speedup : %.2fx%n", (double) nsHash / nsCSR);

        assertTrue(nsCSR < nsHash, "CSR transposeMultiply should be faster than default");
    }

    private static HashSparseMatrix toHash(CSRMatrix csr) {
        HashSparseMatrix h = new HashSparseMatrix(csr.rows(), csr.cols());
        for (int r = 0; r < csr.rows(); r++)
            for (int c = 0; c < csr.cols(); c++) {
                double v = csr.get(r, c);
                if (v != 0.0) h.set(r, c, v);
            }
        return h;
    }
}
