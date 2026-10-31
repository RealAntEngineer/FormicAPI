package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.*; // incl. CpuExecutor
import com.rae.formicapi.foundation.math.operators.backend.gpu.*;
import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import org.junit.jupiter.api.*;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Backend-independent behaviour contract for the GPU vector, matrix and tensor classes -- in <b>both</b>
 * precisions: {@link GpuDoubleVector}/{@link GpuPaddedCSRMatrixD}/... and their float twins
 * {@link GpuFloatVector}/{@link GpuPaddedCSRMatrixF}/... Every result is compared against the CPU implementation of
 * the same interface (or a plain-Java expectation where the CPU class can't express the case), so the suite checks
 * that each backend agrees with the reference semantics.
 *
 * <p><b>Each contract is written once.</b> The bodies below are generic over a {@link Kit} that builds the right
 * flavour of GPU object; {@link #forEachPrecision} turns a body into one dynamic test per precision. For
 * {@link GpuTestPrecision#FLOAT} the CPU reference is a {@code double} implementation fed the <em>float-rounded</em>
 * inputs (see {@link Kit#rand}), so what is measured is the GPU arithmetic, with a float-sized tolerance
 * ({@link GpuTestPrecision#elementTolerance}). The double flavour is skipped, not failed, on a device without fp64;
 * the float flavour needs nothing but the device. Integer and boolean vectors have no precision and run once.
 *
 * <p>Concrete subclasses only say which executor to open; one executor is shared by the whole class.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class GpuContractTest {

    /** @return an open executor, or {@code null} if this backend isn't available (tests are then skipped). */
    abstract GpuExecutor open();

    GpuExecutor ex;

    /**
     * Required by the CPU matrix/tensor reference implementations (their apply methods throw without one).
     * Two threads and a tiny threshold so the reference runs through the parallel path on these small systems;
     * CPU vectors are left without an executor, i.e. plain serial loops.
     */
    CpuExecutor cpuEx;

    @BeforeAll
    void openExecutor() {
        ex = open();
        assumeTrue(ex != null, "backend not available");
        cpuEx = new CpuExecutor(2, 16);
    }

    @AfterAll
    void closeExecutor() throws Exception {
        if (cpuEx != null)
            cpuEx.close();
        if (ex instanceof AutoCloseable c)
            c.close();
    }

    // ------------------------------------------------------------------ precision plumbing

    /** One dynamic test per precision for {@code body}; double is skipped (not failed) where the device has no fp64. */
    Stream<DynamicTest> forEachPrecision(Consumer<Kit> body) {
        return Arrays.stream(GpuTestPrecision.values()).map(p -> DynamicTest.dynamicTest(p.name(), () -> {
            assumeTrue(p.available(ex), "device has no fp64, so the double classes can't run here");
            body.accept(new Kit(p));
        }));
    }

    /** Builds the GPU objects of one precision and reads them back; the only place the two flavours differ. */
    final class Kit {
        final GpuTestPrecision p;

        Kit(GpuTestPrecision p) {
            this.p = p;
        }

        /** values in [0.5, 2.5), already rounded to this precision: never zero (safe divisors), never huge. */
        double[] rand(int n, long seed) {
            return p.round(randomD(n, seed));
        }

        /** A GPU vector holding {@code data} (which must already be rounded to this precision). */
        RealVector vec(double[] data) {
            return p.isFloat() ? new GpuFloatVector(ex, GpuTestPrecision.toFloat(data))
                               : new GpuDoubleVector(ex, data.clone());
        }

        RealVector vec(int n) {
            return p.isFloat() ? new GpuFloatVector(ex, n) : new GpuDoubleVector(ex, n);
        }

        double[] read(RealVector v) {
            if (v instanceof GpuFloatVector f)
                return GpuTestPrecision.widen(f.download());
            return ((GpuDoubleVector) v).download();
        }

        void assertClose(double[] expected, double[] actual) {
            assertEquals(expected.length, actual.length, "length");
            for (int i = 0; i < expected.length; i++)
                assertEquals(expected[i], actual[i], p.tolerance(expected[i]), "element " + i);
        }

        void assertReduction(double expected, double actual, String what) {
            assertEquals(expected, actual, p.reductionTolerance * Math.abs(expected), what);
        }

        void release(RealVector v) {
            try {
                if (v instanceof AutoCloseable c)
                    c.close();
            } catch (Exception e) {
                throw new AssertionError(e);
            }
        }

        // ---- matrices and tensors: the D/F classes share a shape but (mostly) no interface

        Mat matrix(int rows, int cols, int nnzPerRow) {
            if (p.isFloat()) {
                GpuPaddedCSRMatrixF m = new GpuPaddedCSRMatrixF(rows, cols, nnzPerRow);
                m.setExecutor(ex);
                return new Mat(m, m::close);
            }
            GpuPaddedCSRMatrixD m = new GpuPaddedCSRMatrixD(rows, cols, nnzPerRow);
            m.setExecutor(ex);
            return new Mat(m, m::close);
        }

        Quadratic quadratic(int equations, int terms) {
            if (p.isFloat()) {
                GpuPaddedCSR3TensorF t = new GpuPaddedCSR3TensorF(equations, terms);
                t.setExecutor(ex);
                return new Quadratic() {
                    public void setRow(int e, double[] v, int[] v1, int[] v2, int n) { t.setRow(e, v, v1, v2, n); }
                    public void apply(RealVector x, RealVector out) { t.apply(x, out); }
                    public void applyJacobian(RealVector x, RealVector d, RealVector out) { t.multiplyJacobian(x, d, out); }
                    public void close() { t.close(); }
                };
            }
            GpuPaddedCSR3TensorD t = new GpuPaddedCSR3TensorD(equations, terms);
            t.setExecutor(ex);
            return new Quadratic() {
                public void setRow(int e, double[] v, int[] v1, int[] v2, int n) { t.setRow(e, v, v1, v2, n); }
                public void apply(RealVector x, RealVector out) { t.apply(x, out); }
                public void applyJacobian(RealVector x, RealVector d, RealVector out) { t.multiplyJacobian(x, d, out); }
                public void close() { t.close(); }
            };
        }

        OrderN orderN(int order, int equations, int terms) {
            if (p.isFloat()) {
                GpuPaddedCSRTensorF t = new GpuPaddedCSRTensorF(order, equations, terms);
                t.setExecutor(ex);
                return new OrderN() {
                    public void setRow(int e, double[] v, int[][] vars, int n) { t.setRow(e, v, vars, n); }
                    public void apply(RealVector x, RealVector out) { t.apply(x, out); }
                    public void applyJacobian(RealVector x, RealVector d, RealVector out) { t.multiplyJacobian(x, d, out); }
                    public void close() { t.close(); }
                };
            }
            GpuPaddedCSRTensorD t = new GpuPaddedCSRTensorD(order, equations, terms);
            t.setExecutor(ex);
            return new OrderN() {
                public void setRow(int e, double[] v, int[][] vars, int n) { t.setRow(e, v, vars, n); }
                public void apply(RealVector x, RealVector out) { t.apply(x, out); }
                public void applyJacobian(RealVector x, RealVector d, RealVector out) { t.multiplyJacobian(x, d, out); }
                public void close() { t.close(); }
            };
        }
    }

    /** A GPU padded-CSR matrix of either precision plus how to close it ({@code close()} isn't on {@link MutableMatrix}). */
    record Mat(MutableMatrix m, Runnable closer) implements AutoCloseable {
        @Override
        public void close() {
            closer.run();
        }
    }

    interface Quadratic extends AutoCloseable {
        void setRow(int equation, double[] values, int[] var1, int[] var2, int count);

        void apply(RealVector x, RealVector out);

        void applyJacobian(RealVector x, RealVector direction, RealVector out);

        @Override
        void close();
    }

    interface OrderN extends AutoCloseable {
        void setRow(int equation, double[] values, int[][] vars, int count);

        void apply(RealVector x, RealVector out);

        void applyJacobian(RealVector x, RealVector direction, RealVector out);

        @Override
        void close();
    }

    // ------------------------------------------------------------------ helpers

    static final int[] SIZES = {1, 3, 255, 256, 257, 1000, 10_000};

    /** values in [0.5, 2.5): never zero (safe divisors), never huge. Use {@link Kit#rand} when the data goes to a GPU vector. */
    static double[] randomD(int n, long seed) {
        Random r = new Random(seed);
        double[] a = new double[n];
        for (int i = 0; i < n; i++)
            a[i] = 0.5 + 2.0 * r.nextDouble();
        return a;
    }

    /** {@code count} distinct indices in [0, n), in random order. */
    static int[] distinctIndices(int n, int count, long seed) {
        int[] all = new int[n];
        for (int i = 0; i < n; i++) all[i] = i;
        Random r = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = all[i]; all[i] = all[j]; all[j] = t;
        }
        return Arrays.copyOf(all, count);
    }

    GpuIntegerVector gpu(int[] data) {
        return new GpuIntegerVector(ex, data.clone());
    }

    /** Applies {@code op(this, other)} to a CPU pair and a GPU pair holding the same data and compares the results. */
    void compareBinary(Kit k, String name, int n, BiConsumer<RealVector, RealVector> op) {
        double[]        a    = k.rand(n, 1), b = k.rand(n, 2);
        CpuDoubleVector cpuA = new CpuDoubleVector(a.clone());
        op.accept(cpuA, new CpuDoubleVector(b.clone()));
        RealVector gpuA = k.vec(a);
        op.accept(gpuA, k.vec(b));
        try {
            k.assertClose(cpuA.array(), k.read(gpuA));
        } catch (AssertionError e) {
            throw new AssertionError(name + " n=" + n + ": " + e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------------ real vectors: elementwise

    @TestFactory
    Stream<DynamicTest> uploadDownloadRoundTrip() {
        return forEachPrecision(k -> {
            for (int n : new int[]{0, 1, 7, 1000}) {
                double[] data = k.rand(n, 3);
                RealVector v = k.vec(data);
                assertEquals(n, v.size());
                assertArrayEquals(data, k.read(v)); // exact: the data is already representable
                k.release(v);
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> elementwiseOpsMatchCpu() {
        return forEachPrecision(k -> {
            for (int n : SIZES) {
                compareBinary(k, "axpy", n, (a, b) -> a.axpy(2.5, b));
                compareBinary(k, "add(vec)", n, RealVector::add);
                compareBinary(k, "add(scalar)", n, (a, b) -> a.add(1.75));
                compareBinary(k, "subtract", n, RealVector::subtract);
                compareBinary(k, "scale(scalar)", n, (a, b) -> a.scale(-0.5));
                compareBinary(k, "scale(vec)", n, RealVector::scale);
                compareBinary(k, "divide", n, RealVector::divide);
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> chainedOpsKeepOrdering() {
        // many dependent non-blocking dispatches on the same buffers: exercises ordering between them
        return forEachPrecision(k -> {
            double[]        a    = k.rand(5000, 4), b = k.rand(5000, 5);
            CpuDoubleVector cpu  = new CpuDoubleVector(a.clone());
            CpuDoubleVector cpuB = new CpuDoubleVector(b.clone());
            RealVector      g    = k.vec(a), gb = k.vec(b);
            for (int i = 0; i < 50; i++) {
                cpu.axpy(0.1, cpuB).scale(0.99).add(cpuB).subtract(cpuB);
                g.axpy(0.1, gb).scale(0.99).add(gb).subtract(gb);
            }
            k.assertClose(cpu.array(), k.read(g));
        });
    }

    @TestFactory
    Stream<DynamicTest> aliasedOperandIsAllowed() {
        return forEachPrecision(k -> {
            double[] a = k.rand(300, 6);
            RealVector v = k.vec(a);
            v.add(v);
            double[] expected = Arrays.stream(a).map(x -> 2 * x).toArray();
            k.assertClose(expected, k.read(v));
        });
    }

    @TestFactory
    Stream<DynamicTest> singleElementAccessAndSetAdd() {
        return forEachPrecision(k -> {
            RealVector v = k.vec(new double[]{1, 2, 3});
            v.set(10.0, 1);
            assertEquals(10.0, v.get(1));
            v.setAdd(5.5, 1); // regression: setAdd used to ignore its argument
            assertEquals(15.5, v.get(1));
            assertEquals(1.0, v.get(0));
            assertEquals(3.0, v.get(2));

            // the API is double-typed; a float vector must narrow on the way in and widen on the way out
            v.set(0.1, 0);
            assertEquals(k.p.round(0.1), v.get(0));
        });
    }

    @TestFactory
    Stream<DynamicTest> operandsOfWrongKindOrSizeAreRejected() {
        return forEachPrecision(k -> {
            RealVector v = k.vec(k.rand(10, 7));
            assertThrows(UnsupportedOperationException.class, () -> v.add(new CpuDoubleVector(10)));
            assertThrows(IllegalArgumentException.class, () -> v.add(k.vec(k.rand(5, 8))));
            assertThrows(IllegalArgumentException.class, () -> v.dot(k.vec(k.rand(5, 8))));
        });
    }

    // ------------------------------------------------------------------ real vectors: reductions

    @TestFactory
    Stream<DynamicTest> dotMatchesCpuAcrossGroupBoundaries() {
        return forEachPrecision(k -> {
            for (int n : new int[]{1, 2, 255, 256, 257, 511, 512, 513, 1000, 100_000}) {
                double[] a        = k.rand(n, 9), b = k.rand(n, 10);
                double   expected = new CpuDoubleVector(a.clone()).dot(new CpuDoubleVector(b.clone()));
                k.assertReduction(expected, k.vec(a).dot(k.vec(b)), "dot n=" + n);
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> normIsSqrtOfSelfDot() {
        return forEachPrecision(k -> {
            double[] a = k.rand(777, 11);
            k.assertReduction(new CpuDoubleVector(a.clone()).norm(), k.vec(a).norm(), "norm");
        });
    }

    // ------------------------------------------------------------------ real vectors: skipped ops

    private static final boolean[][] FLAGS = {{true, false}, {false, true}, {true, true}};

    interface SkippedOp {
        void apply(RealVector self, RealVector other, IntegerVector idx, boolean t, boolean o);
    }

    void compareSkipped(Kit k, String name, SkippedOp op) {
        int n = 2000;
        int[] idx = distinctIndices(n, 700, 12); // > one work-group, and a strict subset
        for (boolean[] f : FLAGS) {
            double[]        a   = k.rand(n, 13), b = k.rand(n, 14);
            CpuDoubleVector cpu = new CpuDoubleVector(a.clone());
            op.apply(cpu, new CpuDoubleVector(b.clone()), new CpuIntegerVector(idx.clone()), f[0], f[1]);
            RealVector g = k.vec(a);
            op.apply(g, k.vec(b), gpu(idx), f[0], f[1]);
            try {
                k.assertClose(cpu.array(), k.read(g));
            } catch (AssertionError e) {
                throw new AssertionError(name + " thisSkip=" + f[0] + " otherSkip=" + f[1] + ": " + e.getMessage(), e);
            }
        }
    }

    @TestFactory
    Stream<DynamicTest> skippedAddSubtractScaleDivideMatchCpu() {
        return forEachPrecision(k -> {
            compareSkipped(k, "skippedAdd", (s, o, i, t, f) -> s.skippedAdd(o, i, t, f));
            compareSkipped(k, "skippedSubtract", (s, o, i, t, f) -> s.skippedSubtract(o, i, t, f));
            compareSkipped(k, "skippedScale", (s, o, i, t, f) -> s.skippedScale(o, i, t, f));
            compareSkipped(k, "skippedDivide", (s, o, i, t, f) -> s.skippedDivide(o, i, t, f));
        });
    }

    @TestFactory
    Stream<DynamicTest> skippedAxpyHonoursBothFlags() {
        return forEachPrecision(k -> compareSkipped(k, "skippedAxpy", (s, o, i, t, f) -> s.skippedAxpy(1.5, o, i, t, f)));
    }

    @TestFactory
    Stream<DynamicTest> skippedDotHonoursBothFlags() {
        return forEachPrecision(k -> {
            int n = 2000;
            int[] idx = distinctIndices(n, 1000, 15);
            for (boolean[] f : FLAGS) {
                double[] a        = k.rand(n, 16), b = k.rand(n, 17);
                double   expected = new CpuDoubleVector(a.clone()).skippedDot(new CpuDoubleVector(b.clone()), new CpuIntegerVector(idx.clone()), f[0], f[1]);
                k.assertReduction(expected, k.vec(a).skippedDot(k.vec(b), gpu(idx), f[0], f[1]),
                        "thisSkip=" + f[0] + " otherSkip=" + f[1]);
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> skippedOpsRequireAtLeastOneSkipFlag() {
        return forEachPrecision(k -> {
            RealVector       v   = k.vec(k.rand(10, 18)), o = k.vec(k.rand(10, 19));
            GpuIntegerVector idx = gpu(new int[]{1, 2, 3});
            assertThrows(IllegalArgumentException.class, () -> v.skippedAdd(o, idx, false, false));
            assertThrows(IllegalArgumentException.class, () -> v.skippedAxpy(1.0, o, idx, false, false));
            assertThrows(IllegalArgumentException.class, () -> v.skippedDot(o, idx, false, false));
        });
    }

    // ------------------------------------------------------------------ real vectors: gather / resize / copy

    @TestFactory
    Stream<DynamicTest> gatherFromDistinctSource() {
        return forEachPrecision(k -> {
            double[] src = k.rand(1000, 20);
            int[] idx = distinctIndices(1000, 400, 21);
            double[] target = new double[1000];
            Arrays.fill(target, -1);
            RealVector t = k.vec(target);
            t.gather(k.vec(src), gpu(idx));

            double[] expected = target.clone();
            for (int i = 0; i < idx.length; i++) expected[i] = src[idx[i]];
            k.assertClose(expected, k.read(t)); // entries beyond indices.size() untouched
        });
    }

    @TestFactory
    Stream<DynamicTest> gatherAliasedAppliesAnyPermutation() {
        return forEachPrecision(k -> {
            int n = 500;
            double[] a = k.rand(n, 22);
            int[] reverse = new int[n];
            for (int i = 0; i < n; i++) reverse[i] = n - 1 - i; // not safe in place on the CPU; must work here
            RealVector v = k.vec(a);
            v.gather(v, gpu(reverse));

            double[] expected = new double[n];
            for (int i = 0; i < n; i++) expected[i] = a[n - 1 - i];
            k.assertClose(expected, k.read(v));
        });
    }

    @TestFactory
    Stream<DynamicTest> gatherValidatesTargetSize() {
        return forEachPrecision(k -> {
            RealVector small = k.vec(k.rand(2, 23));
            assertThrows(IllegalArgumentException.class, () -> small.gather(k.vec(k.rand(10, 24)), gpu(new int[]{0, 1, 2})));
        });
    }

    @TestFactory
    Stream<DynamicTest> resizeGrowsWithZerosEvenAfterShrink() {
        return forEachPrecision(k -> {
            RealVector v = k.vec(new double[]{1, 2, 3, 4, 5, 6});
            v.resize(2);
            v.resize(5); // back inside capacity: slots 2..4 held 3,4,5 before and must read as zero
            assertArrayEquals(new double[]{1, 2, 0, 0, 0}, k.read(v));

            v.resize(50); // beyond capacity: live elements kept, the rest zero
            double[] expected = new double[50];
            expected[0] = 1; expected[1] = 2;
            assertArrayEquals(expected, k.read(v));
        });
    }

    @TestFactory
    Stream<DynamicTest> copyCloneAndClear() {
        return forEachPrecision(k -> {
            double[] a = k.rand(321, 25);
            RealVector src = k.vec(a);

            RealVector dst = k.vec(3);
            dst.copy(src);
            assertEquals(321, dst.size());
            assertArrayEquals(a, k.read(dst));

            RealVector clone = (RealVector) src.copy();
            src.clear();
            assertArrayEquals(new double[321], k.read(src));
            assertArrayEquals(a, k.read(clone));
        });
    }

    @TestFactory
    Stream<DynamicTest> internalCopyBehavesLikeArraycopyIncludingOverlap() {
        return forEachPrecision(k -> {
            double[] a = k.rand(100, 26);
            int[][] cases = {{0, 50, 30}, {50, 0, 30}, {0, 10, 50}, {10, 0, 50}, {5, 5, 20}, {0, 0, 0}};
            for (int[] c : cases) {
                double[] expected = a.clone();
                System.arraycopy(expected, c[0], expected, c[1], c[2]);
                RealVector v = k.vec(a);
                assertSame(v, v.internalCopy(c[0], c[1], c[2]));
                assertArrayEquals(expected, k.read(v), Arrays.toString(c));
            }
            RealVector v = k.vec(a);
            assertThrows(IndexOutOfBoundsException.class, () -> v.internalCopy(90, 0, 20));
        });
    }

    // ------------------------------------------------------------------ GpuIntegerVector (no precision)

    @Test
    void integerVectorBasics() {
        int[] data = {5, 4, 3, 2, 1, 0};
        GpuIntegerVector v = gpu(data);
        assertArrayEquals(data, v.download());
        v.set(42, 2);
        assertEquals(42, v.get(2));

        v.resize(3);
        v.resize(6); // zero on regrow
        assertArrayEquals(new int[]{5, 4, 42, 0, 0, 0}, v.download());

        v.clear();
        assertArrayEquals(new int[6], v.download());
        assertArrayEquals(new int[6], new GpuIntegerVector(ex, 6).download()); // fresh vectors start at zero
    }

    @Test
    void integerVectorInternalCopy() {
        int[] a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (int[] c : new int[][]{{0, 3, 5}, {3, 0, 5}, {0, 5, 5}}) {
            int[] expected = a.clone();
            System.arraycopy(expected, c[0], expected, c[1], c[2]);
            GpuIntegerVector v = gpu(a);
            v.internalCopy(c[0], c[1], c[2]);
            assertArrayEquals(expected, v.download(), Arrays.toString(c));
        }
    }

    // ------------------------------------------------------------------ GpuBooleanVector (no precision)

    @Test
    void booleanVectorBasics() {
        boolean[] data = {true, false, true, true, false};
        GpuBooleanVector v = new GpuBooleanVector(ex, data);
        assertArrayEquals(data, v.download());
        v.set(true, 1);
        assertTrue(v.get(1));

        v.resize(3);
        v.resize(5);
        assertArrayEquals(new boolean[]{true, true, true, false, false}, v.download());
        v.clear();
        assertArrayEquals(new boolean[5], v.download());
    }

    @Test
    void booleanFillRespectsRangeAndAlignment() {
        // byte granularity matters on backends that fill in 4-byte words: try every offset/length mix
        for (int from = 0; from <= 9; from++) {
            for (int to = from; to <= 13; to++) {
                GpuBooleanVector v = new GpuBooleanVector(ex, 13);
                v.fill(true, from, to);
                boolean[] expected = new boolean[13];
                for (int i = from; i < to; i++) expected[i] = true;
                assertArrayEquals(expected, v.download(), "fill [" + from + "," + to + ")");
            }
        }
        GpuBooleanVector v = new GpuBooleanVector(ex, 5);
        assertThrows(IndexOutOfBoundsException.class, () -> v.fill(true, 0, 6));
        assertThrows(IndexOutOfBoundsException.class, () -> v.fill(true, 3, 2));
    }

    @Test
    void booleanInternalCopy() {
        boolean[] a = {true, false, true, true, false, false, true, false};
        boolean[] expected = a.clone();
        System.arraycopy(expected, 0, expected, 2, 5);
        GpuBooleanVector v = new GpuBooleanVector(ex, a);
        v.internalCopy(0, 2, 5);
        assertArrayEquals(expected, v.download());
    }

    @Test
    void booleanGather() {
        boolean[] src = {true, false, false, true, true, false, true};
        int[] idx = {6, 3, 0, 1};
        GpuBooleanVector t = new GpuBooleanVector(ex, 7);
        try {
            t.gather(new GpuBooleanVector(ex, src), gpu(idx));
        } catch (UnsupportedOperationException e) {
            Assumptions.abort("device lacks 8-bit storage: " + e.getMessage());
        }
        assertArrayEquals(new boolean[]{true, true, true, false, false, false, false}, t.download());

        GpuBooleanVector alias = new GpuBooleanVector(ex, src);
        alias.gather(alias, gpu(new int[]{6, 5, 4, 3, 2, 1, 0}));
        assertArrayEquals(new boolean[]{true, false, true, true, false, false, true}, alias.download());
    }

    // ------------------------------------------------------------------ matrices & tensors

    static final int ROWS = 60, NNZ = 5;

    private void fillMatrices(Kit k, CpuPaddedCSRMatrixD cpu, MutableMatrix gpuM) {
        Random r = new Random(30);
        for (int row = 0; row < ROWS; row++) {
            int count = 1 + r.nextInt(NNZ); // padded rows too
            int[] cols = distinctIndices(ROWS, count, row);
            double[] vals = new double[count];
            for (int i = 0; i < count; i++) vals[i] = k.p.round(r.nextDouble() * 2 - 1);
            cpu.setRow(row, vals, cols, count);
            gpuM.setRow(row, vals, cols, count);
        }
    }

    @TestFactory
    Stream<DynamicTest> csrMatvecMatchesCpu() {
        return forEachPrecision(k -> {
            CpuPaddedCSRMatrixD cpu = new CpuPaddedCSRMatrixD(ROWS, ROWS, NNZ);
            cpu.setExecutor(cpuEx);
            try (Mat g = k.matrix(ROWS, ROWS, NNZ)) {
                fillMatrices(k, cpu, g.m());

                double[] x = k.rand(ROWS, 31);
                CpuDoubleVector cpuOut = new CpuDoubleVector(ROWS);
                cpu.apply(new CpuDoubleVector(x.clone()), cpuOut);

                RealVector gpuOut = k.vec(0);
                g.m().apply(k.vec(x), gpuOut);
                assertEquals(ROWS, gpuOut.size());
                k.assertClose(cpuOut.array(), k.read(gpuOut));
            }
        });
    }

    /**
     * Needs no 64-bit atomics on any device: the float matrix always uses the atomic-free gather, and the double matrix
     * falls back to it where the device has no 64-bit buffer atomics, so there is nothing left to skip here.
     */
    @TestFactory
    Stream<DynamicTest> csrTransposeMatvecMatchesCpu() {
        return forEachPrecision(k -> {
            CpuPaddedCSRMatrixD cpu = new CpuPaddedCSRMatrixD(ROWS, ROWS, NNZ);
            cpu.setExecutor(cpuEx);
            try (Mat g = k.matrix(ROWS, ROWS, NNZ)) {
                fillMatrices(k, cpu, g.m());

                double[] x = k.rand(ROWS, 32);
                CpuDoubleVector cpuOut = new CpuDoubleVector(ROWS);
                cpu.transposeApply(new CpuDoubleVector(x.clone()), cpuOut);

                RealVector gpuOut = k.vec(ROWS);
                g.m().transposeApply(k.vec(x), gpuOut);
                k.assertClose(cpuOut.array(), k.read(gpuOut));
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> csrTransposeMatvecTracksPatternAndValueChanges() {
        // the transpose index is cached: it must be rebuilt when the sparsity pattern changes (setRow) and must NOT
        // go stale when only values change (add) -- so check after every kind of modification, the last one included
        return forEachPrecision(k -> {
            CpuPaddedCSRMatrixD cpu = new CpuPaddedCSRMatrixD(ROWS, ROWS, NNZ);
            cpu.setExecutor(cpuEx);
            try (Mat g = k.matrix(ROWS, ROWS, NNZ)) {
                fillMatrices(k, cpu, g.m());
                double[] x = k.rand(ROWS, 39);

                Runnable check = () -> {
                    CpuDoubleVector cpuOut = new CpuDoubleVector(ROWS);
                    cpu.transposeApply(new CpuDoubleVector(x.clone()), cpuOut);
                    RealVector gpuOut = k.vec(ROWS);
                    g.m().transposeApply(k.vec(x), gpuOut);
                    k.assertClose(cpuOut.array(), k.read(gpuOut));
                };

                check.run();
                for (int round = 0; round < 3; round++) {
                    // values only: the cached index must stay valid and see the new values
                    int row = 7 + round;
                    int col = cpu.getRowCols(row)[0]; // an entry that exists
                    double bump = k.p.round(0.75);
                    cpu.add(row, col, bump);
                    g.m().add(row, col, bump);
                    check.run();

                    // pattern change: another row now points at different columns
                    int[] newCols = distinctIndices(ROWS, 3, 100 + round);
                    double[] newVals = k.p.round(new double[]{0.5, -1.25, 2.0});
                    cpu.setRow(20 + round, newVals, newCols, 3);
                    g.m().setRow(20 + round, newVals, newCols, 3);
                    check.run();
                }
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> csrGetValuesReadsStoredEntries() {
        return forEachPrecision(k -> {
            try (Mat g = k.matrix(4, 4, 2)) {
                g.m().setRow(0, new double[]{1.5, 2.5}, new int[]{0, 3}, 2);
                g.m().setRow(2, new double[]{7.0}, new int[]{2}, 1);

                RealVector out = g.m().getValues(gpu(new int[]{0, 0, 2, 2, 1}), gpu(new int[]{0, 3, 2, 0, 1}));
                assertArrayEquals(new double[]{1.5, 2.5, 7.0, 0.0, 0.0}, k.read(out));
                assertThrows(IllegalArgumentException.class, () -> g.m().getValues(gpu(new int[]{0}), gpu(new int[]{0, 1})));
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> quadraticTensorApplyAndJacobianMatchCpu() {
        return forEachPrecision(k -> {
            final int            eq  = 40, terms = 4;
            CpuPaddedCSR3TensorD cpu = new CpuPaddedCSR3TensorD(eq, terms);
            cpu.setExecutor(cpuEx);
            try (Quadratic g = k.quadratic(eq, terms)) {
                // add() only updates entries that already exist, so structure is defined row by row with setRow
                Random r = new Random(33);
                for (int e = 0; e < eq; e++) {
                    double[] vals = new double[terms];
                    int[] var1 = new int[terms], var2 = new int[terms];
                    for (int t = 0; t < terms; t++) {
                        vals[t] = k.p.round(r.nextDouble() * 2 - 1);
                        var1[t] = r.nextInt(eq);
                        var2[t] = r.nextInt(eq); // var1 == var2 happens: covers the 2*c*x*dx case
                    }
                    var2[0] = var1[0]; // guarantee at least one squared term per equation
                    cpu.setRow(e, vals, var1, var2, terms);
                    g.setRow(e, vals, var1, var2, terms);
                }

                double[]        x      = k.rand(eq, 34), d = k.rand(eq, 35);
                CpuDoubleVector cpuOut = new CpuDoubleVector(eq), cpuJ = new CpuDoubleVector(eq);
                cpu.apply(new CpuDoubleVector(x.clone()), cpuOut);
                cpu.multiplyJacobian(new CpuDoubleVector(x.clone()), new CpuDoubleVector(d.clone()), cpuJ);

                RealVector out = k.vec(eq), jac = k.vec(eq);
                g.apply(k.vec(x), out);
                g.applyJacobian(k.vec(x), k.vec(d), jac);
                k.assertClose(cpuOut.array(), k.read(out));
                k.assertClose(cpuJ.array(), k.read(jac));
            }
        });
    }

    @TestFactory
    Stream<DynamicTest> orderNTensorApplyAndJacobianMatchCpu() {
        return forEachPrecision(k -> {
            final int           order = 4, eq = 30, terms = 3;
            CpuPaddedCSRTensorD cpu   = new CpuPaddedCSRTensorD(order, eq, terms);
            cpu.setExecutor(cpuEx);
            try (OrderN g = k.orderN(order, eq, terms)) {
                Random r = new Random(36);
                for (int e = 0; e < eq; e++) {
                    int count = 1 + r.nextInt(terms); // also exercises padded (inactive) terms
                    double[] vals = new double[count];
                    int[][] vars = new int[count][order - 1]; // [term][order-1]
                    for (int t = 0; t < count; t++) {
                        vals[t] = k.p.round(r.nextDouble() * 2 - 1);
                        for (int d = 0; d < order - 1; d++)
                            vars[t][d] = r.nextInt(eq);
                    }
                    cpu.setRow(e, vals, vars, count);
                    g.setRow(e, vals, vars, count);
                }

                double[]        x      = k.rand(eq, 37), d = k.rand(eq, 38);
                CpuDoubleVector cpuOut = new CpuDoubleVector(eq), cpuJ = new CpuDoubleVector(eq);
                cpu.apply(new CpuDoubleVector(x.clone()), cpuOut);
                cpu.applyJacobian(new CpuDoubleVector(x.clone()), new CpuDoubleVector(d.clone()), cpuJ);

                RealVector out = k.vec(eq), jac = k.vec(eq);
                g.apply(k.vec(x), out);
                g.applyJacobian(k.vec(x), k.vec(d), jac);
                k.assertClose(cpuOut.array(), k.read(out));
                k.assertClose(cpuJ.array(), k.read(jac));
            }
        });
    }
}