package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrixD;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.LeastSquare;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * GPU counterpart to {@code ParallelCpuTest}: same 7-point-stencil matrix, same iteration counts, same "just a plain
 * test with a {@code System.nanoTime()} loop around it" shape - deliberately not a JMH benchmark, so a sampling profiler
 * (async-profiler, JFR, VisualVM) can be attached to a single, short-lived JVM run and see exactly this hot loop.
 *
 * <p>Every result is checked against {@link PaddedCSRMatrixD}, the serial CPU reference, which always computes in
 * {@code double}. The multiply tests run for both precisions: for the float flavour the matrix entries and the vector
 * are rounded to {@code float} <em>before</em> they are given to either side, so the two compute on identical numbers
 * and only the GPU arithmetic is measured, with a float-sized tolerance. GPU vs. CPU is always checked with a tolerance:
 * reduction order and FMA use aren't guaranteed identical across hardware even when the row loop order matches.
 *
 * <p>Each timed loop enqueues all 1000 GPU calls back-to-back and calls {@code finish()} once at the end, rather than
 * after every call - the executor enqueues kernels without waiting (see {@code GpuPaddedCSRMatrixBenchmark}'s javadoc),
 * so without a {@code finish()} the reported time would just be enqueue overhead, and finishing after every call would
 * serialize a pipeline that's meant to run async on the device.
 *
 * <p>The setup is written once ({@link #stencilProblem}); it used to be repeated in every test.
 */
public class ParallelGpuTest extends GpuTestSupport {

    private static final int NX = 64, NY = 64, NZ = 64, ENTRIES_PER_ROW = 7;
    private static final int REPEATS = 1000;

    /** The same random stencil system on the GPU and on the serial CPU reference, plus a random right-hand side. */
    private record Problem(MutableMatrix gpu, PaddedCSRMatrixD serial, double[] x, int rows) {
    }

    /**
     * Builds the system for {@code p}. The executor is attached to the GPU matrix <em>before</em> any {@code setRow}:
     * {@code setRow} uploads straight to the device and throws {@code IllegalStateException} without one.
     */
    private Problem stencilProblem(GpuTestPrecision p) {
        int rows = NX * NY * NZ;
        MutableMatrix gpu = p.matrix(executor, rows, rows, ENTRIES_PER_ROW);
        PaddedCSRMatrixD serial = new PaddedCSRMatrixD(rows, rows, ENTRIES_PER_ROW);

        Random random = new Random(42);
        double[] values = new double[ENTRIES_PER_ROW];
        int[] indices = new int[ENTRIES_PER_ROW];
        generateRandomMatrix(p, indices, values, random, gpu, serial);

        double[] x = new double[rows];
        for (int i = 0; i < rows; i++)
            x[i] = p.round(random.nextDouble());
        return new Problem(gpu, serial, x, rows);
    }

    /** One dynamic test per precision. The OpenCL executor only opens on a device with fp64, so double always runs there. */
    private Stream<DynamicTest> forEachPrecision(Consumer<GpuTestPrecision> body) {
        return GpuTestPrecision.forEach(executor, body);
    }

    @TestFactory
    Stream<DynamicTest> multiplyGpuMatchesSerial() {
        return forEachPrecision(p -> {
            Problem pr = stencilProblem(p);

            CpuDoubleVector xSerial = new CpuDoubleVector(pr.x().clone());
            RealVector xGpu = p.vector(executor, pr.x());
            CpuDoubleVector serial = new CpuDoubleVector(pr.rows());
            RealVector gpu = p.vector(executor, pr.rows());

            long start = System.nanoTime();
            for (int i = 0; i < REPEATS; i++)
                pr.serial().apply(xSerial, serial);
            report("serial", System.nanoTime() - start, pr.rows());

            start = System.nanoTime();
            for (int i = 0; i < REPEATS; i++)
                pr.gpu().apply(xGpu, gpu);
            executor.finish();
            report("gpu " + p, System.nanoTime() - start, pr.rows());

            assertArrayEquals(serial.array(), GpuTestPrecision.read(gpu), tolerance(p));
            GpuTestPrecision.close(pr.gpu());
        });
    }

    @TestFactory
    Stream<DynamicTest> transposeMultiplyGpuMatchesSerial() {
        return forEachPrecision(p -> {
            Problem pr = stencilProblem(p);

            CpuDoubleVector xSerial = new CpuDoubleVector(pr.x().clone());
            RealVector xGpu = p.vector(executor, pr.x());
            CpuDoubleVector serialResult = new CpuDoubleVector(pr.rows());
            RealVector gpuResult = p.vector(executor, pr.rows());

            long start = System.nanoTime();
            for (int i = 0; i < REPEATS; i++)
                pr.serial().transposeApply(xSerial, serialResult);
            report("serial", System.nanoTime() - start, pr.rows());

            start = System.nanoTime();
            for (int i = 0; i < REPEATS; i++)
                pr.gpu().transposeApply(xGpu, gpuResult);
            executor.finish();
            report("gpu " + p, System.nanoTime() - start, pr.rows());

            // The scatter-add (double, where the device has 64-bit atomics) sums in an order that differs from the
            // serial path and between runs; the gather (float, or double without atomics) is reproducible but still
            // not in the serial order. Either way this needs a tolerance rather than exact equality.
            assertArrayEquals(serialResult.array(), GpuTestPrecision.read(gpuResult), tolerance(p));
            GpuTestPrecision.close(pr.gpu());
        });
    }

    /** Absolute tolerance: entries are in [0, 1) and a row has 7 of them, so results stay below 7. */
    private static double tolerance(GpuTestPrecision p) {
        return p.isFloat() ? 1e-4 : 1e-9;
    }

    private static void report(String what, long nanos, int rows) {
        System.out.println(what + " took " + nanos / REPEATS);
        System.out.println("effective ns/row :" + ((float) nanos / rows / REPEATS));
    }

    private static WorkingBuffer<RealVector>[] createGpuBuffers(int m, int n, OpenCLGpuExecutor executor) {
        GpuDoubleVector u    = new GpuDoubleVector(executor, m);
        GpuDoubleVector temp = new GpuDoubleVector(executor, m);

        GpuDoubleVector v     = new GpuDoubleVector(executor, n);
        GpuDoubleVector w     = new GpuDoubleVector(executor, n);
        GpuDoubleVector temp2 = new GpuDoubleVector(executor, n);

        WorkingBuffer<RealVector> mBuffer = new WorkingBuffer<>(new GpuDoubleVector[]{ u, temp });
        WorkingBuffer<RealVector> nBuffer = new WorkingBuffer<>(new GpuDoubleVector[]{ v, w, temp2 });

        return new WorkingBuffer[]{ mBuffer, nBuffer };
    }

    /**
     * GPU counterpart to {@code ParallelCpuTest#leastSquareParallelMatchesSerial}. Double precision only: it is a timing
     * comparison with no assertion on the result (the two solvers stop on a loose tolerance and are not expected to agree
     * element for element), so a float copy would add run time without adding a check.
     *
     * <p>Assumes {@code LeastSquare.solve} is written against the shared matrix/vector interfaces rather than a
     * CPU-specific concrete type. If that assumption is wrong this is the one method in the file to delete; the multiply
     * tests don't depend on it.
     */
    @Test
    void leastSquareGpuMatchesSerial() {
        GpuTestPrecision p = GpuTestPrecision.DOUBLE;
        Problem pr = stencilProblem(p);
        int rows = pr.rows();

        Random random = new Random(43);
        double[] bArr = new double[rows];
        for (int i = 0; i < rows; i++)
            bArr[i] = random.nextDouble() * 2.0 - 1.0;

        // Independent copies so neither solve can accidentally share state with the other through the same backing array.
        GpuDoubleVector bGpu    = new GpuDoubleVector(executor, bArr.clone());
        CpuDoubleVector bSerial = new CpuDoubleVector(bArr.clone());

        int maxIter = 500;
        double tol = 1e-2;

        GpuDoubleVector xGpu = new GpuDoubleVector(executor, rows);
        WorkingBuffer[] gpuBuffers = createGpuBuffers(rows, rows, executor);

        long start = System.nanoTime();
        int iterationsGpu = 0;
        for (int i = 0; i < 10; i++) {
            xGpu.clear();
            iterationsGpu += LeastSquare.solve(pr.gpu(), bGpu, xGpu, maxIter, tol, gpuBuffers[0], gpuBuffers[1]);
            executor.finish();
        }
        executor.finish();
        long end = System.nanoTime();

        System.out.println("gpu took " + (end - start) / rows / 10 + "ns/row");
        System.out.println("gpu took " + (end - start) / rows / iterationsGpu + "ns/row");

        CpuDoubleVector xSerial = new CpuDoubleVector(rows);
        WorkingBuffer[] serialBuffers = createBuffers(rows, rows);
        start = System.nanoTime();
        int iterationsSerial = 0;
        for (int i = 0; i < 10; i++) {
            xSerial.clear();
            iterationsSerial += LeastSquare.solve(pr.serial(), bSerial, xSerial, maxIter, tol, serialBuffers[0], serialBuffers[1]);
        }
        end = System.nanoTime();

        System.out.println("serial took " + (end - start) / rows / 10 + "ns/row");
        System.out.println("serial took " + (end - start) / rows / iterationsSerial + "ns/row");

        GpuTestPrecision.close(pr.gpu());
    }

    private static WorkingBuffer<RealVector>[] createBuffers(int m, int n) {
        CpuDoubleVector u    = new CpuDoubleVector(m);
        CpuDoubleVector temp = new CpuDoubleVector(m);

        CpuDoubleVector v     = new CpuDoubleVector(n);
        CpuDoubleVector w     = new CpuDoubleVector(n);
        CpuDoubleVector temp2 = new CpuDoubleVector(n);

        WorkingBuffer<RealVector> mBuffer = new WorkingBuffer<>(new CpuDoubleVector[]{ u, temp });
        WorkingBuffer<RealVector> nBuffer = new WorkingBuffer<>(new CpuDoubleVector[]{ v, w, temp2 });

        return new WorkingBuffer[]{ mBuffer, nBuffer };
    }

    /** Appends one stencil neighbour, or a zero-valued padding entry pointing back at the row when the neighbour is off the grid. */
    private static int addNeighbour(GpuTestPrecision p, boolean inside, int column, int row, int count,
                                    int[] indices, double[] values, Random random) {
        indices[count] = inside ? column : row;
        values[count] = inside ? p.round(random.nextDouble()) : 0.0;
        return count + 1;
    }

    /**
     * Identical stencil to {@code ParallelCpuTest#generateMatrix}, written to both the GPU matrix and the serial one with
     * the same (precision-rounded) values. The neighbour logic is one helper instead of the six copy-pasted branches it used to be;
     * the draw order of the random numbers is unchanged, so the double flavour builds exactly the matrix it always did.
     */
    private static void generateRandomMatrix(GpuTestPrecision p, int[] indices, double[] values, Random random,
                                             MutableMatrix gpu, PaddedCSRMatrixD serial) {
        for (int z = 0; z < NZ; z++) {
            for (int y = 0; y < NY; y++) {
                for (int x = 0; x < NX; x++) {
                    int row = x + NX * (y + NY * z);

                    int count = 0;
                    indices[count] = row; // center
                    values[count++] = p.round(random.nextDouble());

                    count = addNeighbour(p, x > 0, row - 1, row, count, indices, values, random);
                    count = addNeighbour(p, x < NX - 1, row + 1, row, count, indices, values, random);
                    count = addNeighbour(p, y > 0, row - NX, row, count, indices, values, random);
                    count = addNeighbour(p, y < NY - 1, row + NX, row, count, indices, values, random);
                    count = addNeighbour(p, z > 0, row - NX * NY, row, count, indices, values, random);
                    count = addNeighbour(p, z < NZ - 1, row + NX * NY, row, count, indices, values, random);

                    gpu.setRow(row, values, indices, ENTRIES_PER_ROW);
                    serial.setRow(row, values, indices, ENTRIES_PER_ROW);
                }
            }
        }
    }
}