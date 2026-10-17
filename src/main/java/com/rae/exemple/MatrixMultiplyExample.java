package com.rae.exemple;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
// TODO: add imports for CpuPaddedCSRMatrix, PaddedCSRMatrix, CpuDoubleVector (packages not visible from here)

import java.io.IOException;
import java.util.Random;

/**
 * Serial vs parallel sparse matrix-vector multiply on a 7-point stencil
 * (nx*ny*nz rows), meant to be run from a {@code main} and attached to a profiler.
 *
 * <p>Usage: {@code MatrixMultiplyExample [mode] [threads] [n] [iterations] [rounds] [warmup]}
 * <ul>
 *     <li>{@code mode}       serial | parallel | both (default both). Use a single mode when profiling,
 *                            so the other path doesn't pollute the flame graph.</li>
 *     <li>{@code threads}    CpuExecutor worker count (default 4)</li>
 *     <li>{@code n}          grid edge: rows = n^3 (default 64)</li>
 *     <li>{@code iterations} applies per timed round (default 1000)</li>
 *     <li>{@code rounds}     timed rounds, each printed separately (default 5)</li>
 *     <li>{@code warmup}     untimed applies before measuring, lets the JIT settle (default 2000)</li>
 * </ul>
 *
 * <p>Add {@code -Dprofile.wait=true} to pause before the timed section so a
 * profiler (async-profiler, JFR, VisualVM) can be attached after warmup.
 * Example: {@code java -Dprofile.wait=true ... MatrixMultiplyExample parallel 4 64 5000 10}
 */
public final class MatrixMultiplyExample {

    private static final int ENTRIES_PER_ROW = 7;

    private MatrixMultiplyExample() {
    }

    public static void main(String[] args) throws IOException {
        String mode       = args.length > 0 ? args[0] : "both";
        int    threads    = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        int    n          = args.length > 2 ? Integer.parseInt(args[2]) : 256;
        int    iterations = args.length > 3 ? Integer.parseInt(args[3]) : 100;
        int    rounds     = args.length > 4 ? Integer.parseInt(args[4]) : 5;
        int    warmup     = args.length > 5 ? Integer.parseInt(args[5]) : 200;

        boolean runSerial   = mode.equals("serial") || mode.equals("both");
        boolean runParallel = mode.equals("parallel") || mode.equals("both");
        if (!runSerial && !runParallel)
            throw new IllegalArgumentException("mode must be serial, parallel or both");

        int rows = n * n * n;
        System.out.printf("mode=%s threads=%d rows=%d iterations=%d rounds=%d warmup=%d%n",
                          mode, threads, rows, iterations, rounds, warmup);

        // ---- setup (not part of the measured/profiled section) ----
        CpuPaddedCSRMatrix matrix       = new CpuPaddedCSRMatrix(rows, rows, ENTRIES_PER_ROW);
        PaddedCSRMatrix    matrixSerial = new PaddedCSRMatrix(rows, rows, ENTRIES_PER_ROW);

        Random   random  = new Random(42);
        double[] values  = new double[ENTRIES_PER_ROW];
        int[]    indices = new int[ENTRIES_PER_ROW];

        generateMatrix(n, n, n, indices, values, random, matrix, ENTRIES_PER_ROW, matrixSerial);

        CpuDoubleVector x = new CpuDoubleVector(rows);
        for (int i = 0; i < rows; i++)
            x.array()[i] = random.nextDouble();

        CpuDoubleVector serial   = new CpuDoubleVector(rows);
        CpuDoubleVector parallel = new CpuDoubleVector(rows);

        // ---- serial ----
        if (runSerial) {
            run("serial", rows, iterations, rounds, warmup, () -> matrixSerial.apply(x, serial));
        }

        // ---- parallel ----
        if (runParallel) {
            try (CpuExecutor executor = new CpuExecutor(threads)) {
                matrix.setExecutor(executor);
                run("parallel", rows, iterations, rounds, warmup, () -> matrix.apply(x, parallel));
            }
        }
    }

    /**
     * Warmup, optional profiler pause, then {@code rounds} timed rounds.
     * Per-round output makes JIT/steady-state drift and run-to-run variance visible.
     */
    private static void run(String label, int rows, int iterations, int rounds, int warmup, Runnable apply)
            throws IOException {
        for (int i = 0; i < warmup; i++)
            apply.run();

        if (Boolean.getBoolean("profile.wait")) {
            System.out.println("[" + label + "] warmup done. Attach profiler, then press Enter...");
            System.in.read();
        }

        double best = Double.MAX_VALUE;
        for (int r = 0; r < rounds; r++) {
            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++)
                apply.run();
            long elapsed = System.nanoTime() - start;

            double usPerApply = elapsed / 1_000.0 / iterations;
            double nsPerRow   = (double) elapsed / iterations / rows;
            best = Math.min(best, usPerApply);

            System.out.printf("[%s] round %d: total %.2f ms | %.2f us/apply | %.3f ns/row%n",
                              label, r, elapsed / 1_000_000.0, usPerApply, nsPerRow);
        }
        System.out.printf("[%s] best: %.2f us/apply%n", label, best);
    }

    private static void generateMatrix(int nz, int ny, int nx, int[] indices, double[] values, Random random, CpuPaddedCSRMatrix matrix, int entriesPerRow, PaddedCSRMatrix matrixSerial) {
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {

                    int row = x + nx * (y + ny * z);

                    int count = 0;

                    // center
                    indices[count] = row;
                    values[count++] = random.nextDouble();


                    // x-
                    if (x > 0) {
                        indices[count] = row - 1;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // x+
                    if (x < nx - 1) {
                        indices[count] = row + 1;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // y-
                    if (y > 0) {
                        indices[count] = row - nx;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // y+
                    if (y < ny - 1) {
                        indices[count] = row + nx;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // z-
                    if (z > 0) {
                        indices[count] = row - nx * ny;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // z+
                    if (z < nz - 1) {
                        indices[count] = row + nx * ny;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    matrix.setRow(row, values, indices, entriesPerRow);
                    matrixSerial.setRow(row, values, indices, entriesPerRow);
                }
            }
        }
    }
}