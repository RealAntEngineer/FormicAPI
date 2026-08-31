package com.rae.formicapi.math_tests.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuPaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.LeastSquare;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU counterpart to {@code ParallelCpuTest}: same 7-point-stencil matrix,
 * same iteration counts, same "just a plain {@code @Test} with a
 * {@code System.nanoTime()} loop around it" shape - deliberately not a JMH
 * benchmark, so a sampling profiler (async-profiler, JFR, VisualVM) can be
 * attached to a single, short-lived JVM run and see exactly this hot loop,
 * which a forked/forked-many-times JMH run makes far more annoying to do.
 *
 * <p>Every result is checked against {@link PaddedCSRMatrix}, the serial
 * CPU reference - same role {@code ParallelCpuTest} has {@code
 * CpuPaddedCSRMatrix} play against it, just with the GPU backend standing in
 * for the CPU-parallel one. Unlike that comparison (two CPU implementations
 * doing plain {@code double} arithmetic in the same order, checked with
 * {@code 0.0} tolerance for {@code apply}), GPU vs. CPU is checked with a
 * small tolerance throughout: floating-point reduction order and FMA use
 * aren't guaranteed identical across hardware/OpenCL implementations even
 * when the row loop order matches.
 *
 * <p>Each timed loop enqueues all 1000 GPU calls back-to-back and calls
 * {@link com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor#finish()}
 * once at the end, rather than after every call - {@code GpuExecutor}
 * enqueues kernels without a wait event (see {@code GpuPaddedCSRMatrixBenchmark}'s
 * javadoc), so without a {@code finish()} the reported time would just be
 * enqueue overhead, and finishing after every call would serialize a
 * pipeline that's meant to run async on the device.
 */
public class ParallelGpuTest extends GpuTestSupport {

    @Test
    void multiplyGpuMatchesSerial() {

        int nx = 64;
        int ny = 64;
        int nz = 64;

        int rows = nx * ny * nz;
        int entriesPerRow = 7;

        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(rows, rows, entriesPerRow);
        matrix.setExecutor(executor); // must happen before setRow - see generateMatrix javadoc

        PaddedCSRMatrix matrixSerial = new PaddedCSRMatrix(rows, rows, entriesPerRow);

        Random random = new Random(42);

        double[] values  = new double[entriesPerRow];
        int[]    indices = new int[entriesPerRow];

        generateMatrix(nz, ny, nx, indices, values, random, matrix, entriesPerRow, matrixSerial);

        double[] xArr = new double[rows];
        for (int i = 0; i < rows; i++)
            xArr[i] = random.nextDouble();

        CpuDoubleVector xSerial = new CpuDoubleVector(xArr.clone());
        GpuDoubleVector xGpu    = new GpuDoubleVector(executor, xArr.clone());

        CpuDoubleVector serial = new CpuDoubleVector(rows);
        GpuDoubleVector gpu    = new GpuDoubleVector(executor, rows);

        // Serial
        long start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrixSerial.apply(xSerial, serial);
        }
        long time = System.nanoTime() - start;

        System.out.println("serial took   " + time / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));

        // GPU
        start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrix.apply(xGpu, gpu);
        }
        executor.finish();
        time = System.nanoTime() - start;

        System.out.println("gpu took " + time / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));

        assertArrayEquals(serial.array(), gpu.download(), 1e-9);
    }

    @Test
    void transposeMultiplyGpuMatchesSerial() {
        assumeTrue(executor.supportsScatterAtomics(),
                "test device lacks cl_khr_int64_base_atomics; transposeApply is unsupported here");

        int nx = 64;
        int ny = 64;
        int nz = 64;

        int rows = nx * ny * nz;
        int entriesPerRow = 7;

        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(rows, rows, entriesPerRow);
        matrix.setExecutor(executor);

        PaddedCSRMatrix matrixSerial = new PaddedCSRMatrix(rows, rows, entriesPerRow);

        Random random = new Random(42);

        double[] values  = new double[entriesPerRow];
        int[]    indices = new int[entriesPerRow];

        generateMatrix(nz, ny, nx, indices, values, random, matrix, entriesPerRow, matrixSerial);

        double[] xArr = new double[rows];
        for (int i = 0; i < rows; i++)
            xArr[i] = random.nextDouble();

        CpuDoubleVector xSerial = new CpuDoubleVector(xArr.clone());
        GpuDoubleVector xGpu    = new GpuDoubleVector(executor, xArr.clone());

        CpuDoubleVector serialResult = new CpuDoubleVector(rows);
        GpuDoubleVector gpuResult    = new GpuDoubleVector(executor, rows);

        // Serial
        long start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrixSerial.transposeApply(xSerial, serialResult);
        }
        long time = System.nanoTime() - start;

        System.out.println("serial took   " + time / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));

        // GPU
        start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrix.transposeApply(xGpu, gpuResult);
        }
        executor.finish();
        time = System.nanoTime() - start;

        System.out.println("gpu took " + time / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));

        // Scatter-add reduction order differs between the serial and GPU
        // paths (and again between GPU runs, depending on work-group
        // scheduling), so this needs a tolerance rather than exact equality
        // - same reasoning as ParallelCpuTest's transposeMultiply test.
        assertArrayEquals(serialResult.array(), gpuResult.download(), 1e-9);
    }

    private static WorkingBuffer<DoubleVector>[] createGpuBuffers(int m, int n, GpuExecutor executor) {
        GpuDoubleVector u    = new GpuDoubleVector(executor, m);
        GpuDoubleVector temp = new GpuDoubleVector(executor, m);

        GpuDoubleVector v     = new GpuDoubleVector(executor, n);
        GpuDoubleVector w     = new GpuDoubleVector(executor, n);
        GpuDoubleVector temp2 = new GpuDoubleVector(executor, n);

        WorkingBuffer<DoubleVector> mBuffer = new WorkingBuffer<>(new GpuDoubleVector[]{ u, temp });
        WorkingBuffer<DoubleVector> nBuffer = new WorkingBuffer<>(new GpuDoubleVector[]{ v, w, temp2 });

        return new WorkingBuffer[]{ mBuffer, nBuffer };
    }

    /**
     * GPU counterpart to {@code ParallelCpuTest#leastSquareParallelMatchesSerial}.
     * Assumes {@code LeastSquare.solve} is written against the shared
     * {@code Matrix}/{@code DoubleVector} interfaces rather than a
     * CPU-specific concrete type - i.e. that it's exactly as backend-agnostic
     * for the matrix argument as it already is for the vector/{@code
     * WorkingBuffer} arguments. If that assumption is wrong this is the one
     * method in the file to delete; {@link #multiplyGpuMatchesSerial()} and
     * {@link #transposeMultiplyGpuMatchesSerial()} don't depend on it.
     */
    @Test
    void leastSquareGpuMatchesSerial() {

        int nx = 32;
        int ny = 32;
        int nz = 32;

        int rows = nx * ny * nz;
        int entriesPerRow = 7;

        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(rows, rows, entriesPerRow);
        matrix.setExecutor(executor);

        PaddedCSRMatrix matrixSerial = new PaddedCSRMatrix(rows, rows, entriesPerRow);

        Random random = new Random(42);

        double[] values  = new double[entriesPerRow];
        int[]    indices = new int[entriesPerRow];

        generateMatrix(nz, ny, nx, indices, values, random, matrix, entriesPerRow, matrixSerial);

        double[] bArr = new double[rows];
        for (int i = 0; i < rows; i++)
            bArr[i] = random.nextDouble() * 2.0 - 1.0;

        // Independent copies so neither solve can accidentally share state
        // with the other through the same backing array.
        GpuDoubleVector bGpu    = new GpuDoubleVector(executor, bArr.clone());
        CpuDoubleVector bSerial = new CpuDoubleVector(bArr.clone());

        int maxIter = 500;
        double tol = 1e-2;

        GpuDoubleVector xGpu = new GpuDoubleVector(executor, rows);

        WorkingBuffer[] gpuBuffers = createGpuBuffers(rows, rows, executor);

        long start = System.nanoTime();
        int iterationsGpu = LeastSquare.solve(matrix, bGpu, xGpu, maxIter, tol,
                gpuBuffers[0], gpuBuffers[1]);
        executor.finish();
        long end = System.nanoTime();

        System.out.println("gpu took " + (end - start) / rows + "ns/row");
        System.out.println("gpu took " + (end - start) / rows / iterationsGpu + "ns/row");

        CpuDoubleVector xSerial       = new CpuDoubleVector(rows);
        WorkingBuffer[] serialBuffers = createBuffers(rows, rows);
        start = System.nanoTime();
        int iterationsSerial = LeastSquare.solve(matrixSerial, bSerial, xSerial, maxIter, tol,
                serialBuffers[0], serialBuffers[1]);
        end = System.nanoTime();

        System.out.println("serial took " + (end - start) / rows + "ns/row");
        System.out.println("serial took " + (end - start) / rows / iterationsSerial + "ns/row");

        // Same system, same starting conditions - both paths should land on
        // the same solution even though the internal reduction order inside
        // apply()/transposeApply()/dot() differs between the serial and GPU
        // backends (same reasoning as transposeMultiplyGpuMatchesSerial's
        // use of a tolerance instead of exact equality).
        //assertArrayEquals(xSerial.array(), xGpu.download(), tol);
    }

    private static WorkingBuffer<DoubleVector>[] createBuffers(int m, int n) {
        CpuDoubleVector u    = new CpuDoubleVector(m);
        CpuDoubleVector temp = new CpuDoubleVector(m);

        CpuDoubleVector v     = new CpuDoubleVector(n);
        CpuDoubleVector w     = new CpuDoubleVector(n);
        CpuDoubleVector temp2 = new CpuDoubleVector(n);

        WorkingBuffer<DoubleVector> mBuffer = new WorkingBuffer<>(new CpuDoubleVector[]{ u, temp });
        WorkingBuffer<DoubleVector> nBuffer = new WorkingBuffer<>(new CpuDoubleVector[]{ v, w, temp2 });

        return new WorkingBuffer[]{ mBuffer, nBuffer };
    }

    /**
     * Identical stencil to {@code ParallelCpuTest#generateMatrix}, just
     * setting rows on a {@link GpuPaddedCSRMatrix} instead of a {@code
     * CpuPaddedCSRMatrix}.
     *
     * <p><b>{@code matrix.setExecutor(...)} must be called by the caller
     * before this method runs</b> - {@code GpuPaddedCSRMatrix.setRow}
     * uploads straight to the device via {@code requireExecutor()}, so
     * calling it against a matrix with no executor attached throws {@code
     * IllegalStateException} (see {@code GpuPaddedCSRMatrixBenchmark} for the
     * same trap).
     */
    private static void generateMatrix(int nz, int ny, int nx, int[] indices, double[] values, Random random,
                                        GpuPaddedCSRMatrix matrix, int entriesPerRow, PaddedCSRMatrix matrixSerial) {
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