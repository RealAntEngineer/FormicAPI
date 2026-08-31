package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutable;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuPaddedCSRMatrix;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for the GPU-backed {@link GpuPaddedCSRMatrix}:
 * {@link GpuPaddedCSRMatrix#apply} (Ax, one work-item per row, the
 * {@code csr_matvec} kernel) and {@link GpuPaddedCSRMatrix#transposeApply}
 * (Aᵀx, a scatter-add via {@code csr_matvec_transpose} using a
 * compare-and-swap atomic double-add, since OpenCL has no native one - see
 * {@code GpuExecutor.ATOMIC_KERNEL_SOURCE}).
 *
 * <p>Uses the same 7-point-stencil matrix construction as
 * {@link CpuPaddedCSRMatrixParallelBenchmark} (blocks of 4096 rows,
 * nnzPerRow = 7) so the two benchmarks' numbers are directly comparable
 * row-count for row-count. Unlike the CPU version there is no
 * {@code threads} parameter: {@link GpuExecutor} dispatches onto a single
 * device queue, and (per {@link GpuExecutable}'s javadoc) there is no
 * serial/size-threshold fallback on the GPU path - every row goes through a
 * kernel launch regardless of {@code blocks}.
 *
 * <p><b>{@code executor.finish()} is called at the end of every
 * {@code @Benchmark} method, before {@code Blackhole.consume}.</b> This
 * matters here in a way it doesn't for the CPU benchmark: {@code GpuExecutor}
 * enqueues kernels with {@code clEnqueueNDRangeKernel(..., 0, null, null)} -
 * no wait event - so the launch call returns as soon as the command is
 * queued, not when the GPU finishes executing it. Without an explicit
 * {@code finish()} (or a blocking download), JMH would only be timing
 * enqueue overhead, not the actual matvec.
 *
 * <p>{@code transposeApply} requires a device advertising
 * {@code cl_khr_int64_base_atomics}; {@link #setup()} fails fast with a
 * clear message if the selected device doesn't support it, rather than
 * letting every {@code transposeApply} trial fail individually with
 * {@link UnsupportedOperationException}.
 *
 * <p><b>Every trial's {@link GpuExecutor} is closed in
 * {@code @TearDown(Level.Trial)}</b>, releasing the OpenCL context/queue/
 * program - mirroring the CPU benchmark's executor shutdown, though the
 * failure mode it guards against is different: a GPU context left open
 * doesn't busy-spin a CPU core the way an un-shut-down {@code CpuExecutor}
 * thread pool would, but it does leak device memory and command-queue
 * resources across trials in the same fork.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class GpuPaddedCSRMatrixBenchmark {

    /**
     * Rows == cols (square matrix). Same block sizing as
     * {@link CpuPaddedCSRMatrixParallelBenchmark#blocks} for comparability;
     * there's no serial-fallback crossover to exercise on the GPU path, so
     * unlike the CPU benchmark no row count is included specifically to
     * demonstrate one.
     */
    @Param({"16", "64", "256"})
    public int blocks;

    private GpuExecutor executor;
    private GpuPaddedCSRMatrix matrix;
    private GpuDoubleVector x;
    private GpuDoubleVector result;

    @Setup(Level.Trial)
    public void setup() {
        Random rnd = new Random(42);

        executor = new GpuExecutor();

        if (!executor.supportsScatterAtomics())
            throw new IllegalStateException(
                    "Selected OpenCL device lacks cl_khr_int64_base_atomics; "
                            + "GpuPaddedCSRMatrix.transposeApply(...) cannot run on it. "
                            + "Run this benchmark on a device that supports it, "
                            + "or comment out the transposeApply benchmark method.");

        int rows = blocks * 4096;
        matrix = createPaddedMatrix(rnd, rows, executor);

        double[] xArr = new double[rows];
        for (int i = 0; i < rows; i++)
            xArr[i] = rnd.nextDouble();

        x = new GpuDoubleVector(executor, xArr);
        result = new GpuDoubleVector(executor, rows);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        matrix.close();
        executor.close();
    }

    @Benchmark
    public void apply(Blackhole bh) {
        for (int i = 0; i < 1000; i++) {
            matrix.apply(x, result);
        }
        executor.finish();
        bh.consume(result);
    }

    //@Benchmark
    public void transposeApply(Blackhole bh) {
        for (int i = 0; i < 1000; i++) {
            matrix.transposeApply(x, result);
        }
        executor.finish();
        bh.consume(result);
    }

    private GpuPaddedCSRMatrix createPaddedMatrix(Random random, int rows, GpuExecutor executor) {

        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(rows, rows, 7);
        matrix.setExecutor(executor);

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
                    cols[3] = row;
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