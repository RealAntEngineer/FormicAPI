package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for the CPU-parallel {@link CpuPaddedCSRMatrix}:
 * {@link CpuPaddedCSRMatrix#apply} (Ax) and
 * {@link CpuPaddedCSRMatrix#transposeApply} (Aᵀx).
 *
 * <p>{@code apply} is a straightforward per-row parallel gather, but
 * {@code transposeApply} is a scatter-add: two different row ranges can write
 * the same output column, so {@link CpuPaddedCSRMatrix} routes it through
 * {@link CpuExecutor#parallelForAccumulate}, which pays a fixed
 * {@code O(rows)} clear+reduce cost per call on top of the actual compute —
 * benchmarking both side by side is the point here, not just {@code apply}
 * alone.
 *
 * <p>{@code rows} below {@link CpuExecutor#DEFAULT_PARALLEL_THRESHOLD} (4096)
 * falls back to the same serial loop regardless of {@code threads} — the
 * {@code rows=1000} row is included on purpose to make that crossover
 * visible, not as a bug.
 *
 * <p><b>Every trial's {@link CpuExecutor} is shut down in
 * {@code @TearDown(Level.Trial)}.</b> See
 * {@link CpuPaddedCSR2TensorParallelBenchmark} for why this matters — an
 * un-shut-down executor's worker threads busy-spin (pinning a core each) for
 * the rest of the JVM's life and would skew later trials in the same fork.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class CpuPaddedCSRMatrixParallelBenchmark {

    /**
     * Rows == cols (square matrix). Includes one value below
     * {@link CpuExecutor#DEFAULT_PARALLEL_THRESHOLD} (4096) to exercise the
     * serial-fallback path.
     */
    @Param({"16", "64", "256"})
    public int blocks;

    /** CpuExecutor worker thread count. Tune to the target machine's core count. */
    @Param({"1", "4", "8"})
    public int threads;

    private CpuExecutor executor;
    private CpuPaddedCSRMatrix matrix;
    private CpuDoubleVector    x;
    private CpuDoubleVector    result;

    @Setup(Level.Trial)
    public void setup() {
        Random rnd = new Random(42);

        executor = new CpuExecutor(threads);
        int rows = blocks * 4096;
        matrix = createPaddedMatrix(rnd);
        matrix.setExecutor(executor);

        double[] xArr = new double[rows];
        for (int i = 0; i < rows; i++)
            xArr[i] = rnd.nextDouble();

        x = new CpuDoubleVector(xArr);
        result = new CpuDoubleVector(rows);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        executor.shutdown();
    }

    @Benchmark
    public void apply(Blackhole bh) {
        matrix.apply(x, result);
        bh.consume(result);
    }

    @Benchmark
    public void transposeApply(Blackhole bh) {
        matrix.transposeApply(x, result);
        bh.consume(result);
    }

    private CpuPaddedCSRMatrix createPaddedMatrix(Random random) {

        int rows = blocks * 4096;

        CpuPaddedCSRMatrix matrix = new CpuPaddedCSRMatrix(rows, rows, 7);

        for (int b = 0; b < blocks; b++) {

            int self = b * 4096;

            int xmBlock = b > 0 ? (b - 1) * 4096 : -1;
            int xpBlock = b < blocks - 1 ? (b + 1) * 4096 : -1;


            for (int idx = 0; idx < 4096; idx++) {

                int row = self + idx;

                double[] values = new double[7];
                int[] cols = new int[7];

                values[0] = random.nextDouble();
                cols[0] = row;

                int x = idx & 15;
                int z = (idx >> 4) & 15;
                int y = idx >> 8;

                // -X
                if (x > 0) {
                    cols[1] = row - 1;
                } else {
                    cols[1] = xmBlock != -1 ? xmBlock + idx + 15 : row;
                }

                // +X
                if (x < 15) {
                    cols[2] = row + 1;
                } else {
                    cols[2] = xpBlock != -1 ? xpBlock + idx - 15 : row;
                }

                // -Y
                if (y > 0) {
                    cols[3] = row - 256;
                } else {
                    cols[3] = row; // TODO add the neighbors.
                }

                // +Y
                if (y < 15) {
                    cols[4] = row + 256;
                } else {
                    cols[4] = row;
                }

                // -Z
                if (z > 0) {
                    cols[5] = row - 16;
                } else {
                    cols[5] = row;
                }

                // +Z
                if (z < 15) {
                    cols[6] = row + 16;
                } else {
                    cols[6] = row;
                }

                for (int i = 1; i < 7; i++) {
                    values[i] = random.nextDouble();
                }

                matrix.setRow(row, values, cols, 7);
            }
        }

        return matrix;
    }
}