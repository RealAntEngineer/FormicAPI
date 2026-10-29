package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.*; // incl. CpuExecutor
import com.rae.formicapi.foundation.math.operators.backend.gpu.*;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import org.junit.jupiter.api.*;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Backend-independent behaviour contract for {@link GpuDoubleVector}, {@link GpuIntegerVector},
 * {@link GpuBooleanVector} and the GPU matrix/tensor classes. Every result is compared against the
 * CPU implementation of the same interface (or a plain-Java expectation where the CPU class can't
 * express the case), so the suite checks that each backend agrees with the reference semantics.
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

    // ------------------------------------------------------------------ helpers

    static final int[] SIZES = {1, 3, 255, 256, 257, 1000, 10_000};

    /** values in [0.5, 2.5): never zero (safe divisors), never huge. */
    static double[] randomD(int n, long seed) {
        Random r = new Random(seed);
        double[] a = new double[n];
        for (int i = 0; i < n; i++)
            a[i] = 0.5 + 2.0 * r.nextDouble();
        return a;
    }

    static float[] randomF(int n, long seed) {
        Random r = new Random(seed);
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = 0.5f + 2.0f * r.nextFloat();
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

    static void assertClose(double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length, "length");
        for (int i = 0; i < expected.length; i++)
            assertEquals(expected[i], actual[i], 1e-12 * Math.max(1.0, Math.abs(expected[i])), "element " + i);
    }

    GpuDoubleVector gpu(double[] data) {
        return new GpuDoubleVector(ex, data.clone());
    }

    GpuFloatVector gpu(float[] data) {
        return new GpuFloatVector(ex, data.clone());
    }

    GpuIntegerVector gpu(int[] data) {
        return new GpuIntegerVector(ex, data.clone());
    }

    /** Applies {@code op(this, other)} to a CPU pair and a GPU pair holding the same data and compares the results. */
    void compareBinary(String name, int n, BiConsumer<RealVector, RealVector> op) {
        double[]        a    = randomD(n, 1), b = randomD(n, 2);
        CpuDoubleVector cpuA = new CpuDoubleVector(a.clone());
        op.accept(cpuA, new CpuDoubleVector(b.clone()));
        GpuDoubleVector gpuA = gpu(a);
        op.accept(gpuA, gpu(b));
        try {
            assertClose(cpuA.array(), gpuA.download());
        } catch (AssertionError e) {
            throw new AssertionError(name + " n=" + n + ": " + e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------------ GpuDoubleVector: elementwise

    @Test
    void uploadDownloadRoundTrip() {
        for (int n : new int[]{0, 1, 7, 1000}) {
            double[] data = randomD(n, 3);
            try (GpuDoubleVector v = gpu(data)) {
                assertEquals(n, v.size());
                assertArrayEquals(data, v.download());
            }
        }
    }

    @Test
    void uploadDownloadFloatRoundTrip() {
        for (int n : new int[]{0, 1, 7, 1000}) {
            float[] data = randomF(n, 3);
            try (GpuFloatVector v = gpu(data)) {
                assertEquals(n, v.size());
                assertArrayEquals(data, v.download());
            }
        }
    }

    @Test
    void elementwiseOpsMatchCpu() {
        for (int n : SIZES) {
            compareBinary("axpy", n, (a, b) -> a.axpy(2.5, b));
            compareBinary("add(vec)", n, RealVector::add);
            compareBinary("add(scalar)", n, (a, b) -> a.add(1.75));
            compareBinary("subtract", n, RealVector::subtract);
            compareBinary("scale(scalar)", n, (a, b) -> a.scale(-0.5));
            compareBinary("scale(vec)", n, RealVector::scale);
            compareBinary("divide", n, RealVector::divide);
        }
    }

    @Test
    void chainedOpsKeepOrdering() {
        // many dependent non-blocking dispatches on the same buffers: exercises ordering between them
        double[]        a   = randomD(5000, 4), b = randomD(5000, 5);
        CpuDoubleVector cpu = new CpuDoubleVector(a.clone());
        CpuDoubleVector cpuB = new CpuDoubleVector(b.clone());
        GpuDoubleVector g = gpu(a), gb = gpu(b);
        for (int i = 0; i < 50; i++) {
            cpu.axpy(0.1, cpuB).scale(0.99).add(cpuB).subtract(cpuB);
            g.axpy(0.1, gb).scale(0.99).add(gb).subtract(gb);
        }
        assertClose(cpu.array(), g.download());
    }

    @Test
    void aliasedOperandIsAllowed() {
        double[] a = randomD(300, 6);
        GpuDoubleVector v = gpu(a);
        v.add(v);
        double[] expected = Arrays.stream(a).map(x -> 2 * x).toArray();
        assertClose(expected, v.download());
    }

    @Test
    void singleElementAccessAndSetAdd() {
        GpuDoubleVector v = gpu(new double[]{1, 2, 3});
        v.set(10.0, 1);
        assertEquals(10.0, v.get(1));
        v.setAdd(5.5, 1); // regression: setAdd used to ignore its argument
        assertEquals(15.5, v.get(1));
        assertEquals(1.0, v.get(0));
        assertEquals(3.0, v.get(2));
    }

    @Test
    void operandsOfWrongKindOrSizeAreRejected() {
        GpuDoubleVector v = gpu(randomD(10, 7));
        assertThrows(UnsupportedOperationException.class, () -> v.add(new CpuDoubleVector(10)));
        assertThrows(IllegalArgumentException.class, () -> v.add(gpu(randomD(5, 8))));
        assertThrows(IllegalArgumentException.class, () -> v.dot(gpu(randomD(5, 8))));
    }

    // ------------------------------------------------------------------ GpuDoubleVector: reductions

    @Test
    void dotMatchesCpuAcrossGroupBoundaries() {
        for (int n : new int[]{1, 2, 255, 256, 257, 511, 512, 513, 1000, 100_000}) {
            double[] a        = randomD(n, 9), b = randomD(n, 10);
            double   expected = new CpuDoubleVector(a.clone()).dot(new CpuDoubleVector(b.clone()));
            double actual = gpu(a).dot(gpu(b));
            assertEquals(expected, actual, 1e-10 * Math.abs(expected), "dot n=" + n);
        }
    }

    @Test
    void normIsSqrtOfSelfDot() {
        double[] a = randomD(777, 11);
        assertEquals(new CpuDoubleVector(a.clone()).norm(), gpu(a).norm(), 1e-10);
    }

    // ------------------------------------------------------------------ GpuDoubleVector: skipped ops

    private static final boolean[][] FLAGS = {{true, false}, {false, true}, {true, true}};

    interface SkippedOp {
        void apply(RealVector self, RealVector other, IntegerVector idx, boolean t, boolean o);
    }

    void compareSkipped(String name, SkippedOp op) {
        int n = 2000;
        int[] idx = distinctIndices(n, 700, 12); // > one work-group, and a strict subset
        for (boolean[] f : FLAGS) {
            double[]        a   = randomD(n, 13), b = randomD(n, 14);
            CpuDoubleVector cpu = new CpuDoubleVector(a.clone());
            op.apply(cpu, new CpuDoubleVector(b.clone()), new CpuIntegerVector(idx.clone()), f[0], f[1]);
            GpuDoubleVector g = gpu(a);
            op.apply(g, gpu(b), gpu(idx), f[0], f[1]);
            try {
                assertClose(cpu.array(), g.download());
            } catch (AssertionError e) {
                throw new AssertionError(name + " thisSkip=" + f[0] + " otherSkip=" + f[1] + ": " + e.getMessage(), e);
            }
        }
    }

    @Test
    void skippedAddSubtractScaleDivideMatchCpu() {
        compareSkipped("skippedAdd", (s, o, i, t, k) -> s.skippedAdd(o, i, t, k));
        compareSkipped("skippedSubtract", (s, o, i, t, k) -> s.skippedSubtract(o, i, t, k));
        compareSkipped("skippedScale", (s, o, i, t, k) -> s.skippedScale(o, i, t, k));
        compareSkipped("skippedDivide", (s, o, i, t, k) -> s.skippedDivide(o, i, t, k));
    }

    @Test
    void skippedAxpyHonoursBothFlags() {
        compareSkipped("skippedAxpy", (s, o, i, t, k) -> s.skippedAxpy(1.5, o, i, t, k));
    }

    @Test
    void skippedDotHonoursBothFlags() {
        int n = 2000;
        int[] idx = distinctIndices(n, 1000, 15);
        for (boolean[] f : FLAGS) {
            double[] a        = randomD(n, 16), b = randomD(n, 17);
            double   expected = new CpuDoubleVector(a.clone()).skippedDot(new CpuDoubleVector(b.clone()), new CpuIntegerVector(idx.clone()), f[0], f[1]);
            double actual = gpu(a).skippedDot(gpu(b), gpu(idx), f[0], f[1]);
            assertEquals(expected, actual, 1e-10 * Math.abs(expected), "thisSkip=" + f[0] + " otherSkip=" + f[1]);
        }
    }

    @Test
    void skippedOpsRequireAtLeastOneSkipFlag() {
        GpuDoubleVector  v   = gpu(randomD(10, 18)), o = gpu(randomD(10, 19));
        GpuIntegerVector idx = gpu(new int[]{1, 2, 3});
        assertThrows(IllegalArgumentException.class, () -> v.skippedAdd(o, idx, false, false));
        assertThrows(IllegalArgumentException.class, () -> v.skippedAxpy(1.0, o, idx, false, false));
        assertThrows(IllegalArgumentException.class, () -> v.skippedDot(o, idx, false, false));
    }

    // ------------------------------------------------------------------ GpuDoubleVector: gather / resize / copy

    @Test
    void gatherFromDistinctSource() {
        double[] src = randomD(1000, 20);
        int[] idx = distinctIndices(1000, 400, 21);
        double[] target = new double[1000];
        Arrays.fill(target, -1);
        GpuDoubleVector t = gpu(target);
        t.gather(gpu(src), gpu(idx));

        double[] expected = target.clone();
        for (int i = 0; i < idx.length; i++) expected[i] = src[idx[i]];
        assertClose(expected, t.download()); // entries beyond indices.size() untouched
    }

    @Test
    void gatherAliasedAppliesAnyPermutation() {
        int n = 500;
        double[] a = randomD(n, 22);
        int[] reverse = new int[n];
        for (int i = 0; i < n; i++) reverse[i] = n - 1 - i; // not safe in place on the CPU; must work here
        GpuDoubleVector v = gpu(a);
        v.gather(v, gpu(reverse));

        double[] expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = a[n - 1 - i];
        assertClose(expected, v.download());
    }

    @Test
    void gatherValidatesTargetSize() {
        GpuDoubleVector small = gpu(randomD(2, 23));
        assertThrows(IllegalArgumentException.class, () -> small.gather(gpu(randomD(10, 24)), gpu(new int[]{0, 1, 2})));
    }

    @Test
    void resizeGrowsWithZerosEvenAfterShrink() {
        GpuDoubleVector v = gpu(new double[]{1, 2, 3, 4, 5, 6});
        v.resize(2);
        v.resize(5); // back inside capacity: slots 2..4 held 3,4,5 before and must read as zero
        assertArrayEquals(new double[]{1, 2, 0, 0, 0}, v.download());

        v.resize(50); // beyond capacity: live elements kept, the rest zero
        double[] expected = new double[50];
        expected[0] = 1; expected[1] = 2;
        assertArrayEquals(expected, v.download());
    }

    @Test
    void copyCloneAndClear() {
        double[] a = randomD(321, 25);
        GpuDoubleVector src = gpu(a);

        GpuDoubleVector dst = new GpuDoubleVector(ex, 3);
        dst.copy(src);
        assertEquals(321, dst.size());
        assertArrayEquals(a, dst.download());

        GpuDoubleVector clone = (GpuDoubleVector) src.copy();
        src.clear();
        assertArrayEquals(new double[321], src.download());
        assertArrayEquals(a, clone.download());
    }

    @Test
    void internalCopyBehavesLikeArraycopyIncludingOverlap() {
        double[] a = randomD(100, 26);
        int[][] cases = {{0, 50, 30}, {50, 0, 30}, {0, 10, 50}, {10, 0, 50}, {5, 5, 20}, {0, 0, 0}};
        for (int[] c : cases) {
            double[] expected = a.clone();
            System.arraycopy(expected, c[0], expected, c[1], c[2]);
            GpuDoubleVector v = gpu(a);
            assertSame(v, v.internalCopy(c[0], c[1], c[2]));
            assertArrayEquals(expected, v.download(), Arrays.toString(c));
        }
        GpuDoubleVector v = gpu(a);
        assertThrows(IndexOutOfBoundsException.class, () -> v.internalCopy(90, 0, 20));
    }

    // ------------------------------------------------------------------ GpuIntegerVector

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

    // ------------------------------------------------------------------ GpuBooleanVector

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

    private void fillMatrices(CpuPaddedCSRMatrixD cpu, GpuPaddedCSRMatrixD gpuM) {
        Random r = new Random(30);
        for (int row = 0; row < ROWS; row++) {
            int count = 1 + r.nextInt(NNZ); // padded rows too
            int[] cols = distinctIndices(ROWS, count, row);
            double[] vals = new double[count];
            for (int i = 0; i < count; i++) vals[i] = r.nextDouble() * 2 - 1;
            cpu.setRow(row, vals, cols, count);
            gpuM.setRow(row, vals, cols, count);
        }
    }

    @Test
    void csrMatvecMatchesCpu() {
        CpuPaddedCSRMatrixD cpu = new CpuPaddedCSRMatrixD(ROWS, ROWS, NNZ);
        cpu.setExecutor(cpuEx);
        GpuPaddedCSRMatrixD g = new GpuPaddedCSRMatrixD(ROWS, ROWS, NNZ);
        g.setExecutor(ex);
        fillMatrices(cpu, g);

        double[] x = randomD(ROWS, 31);
        CpuDoubleVector cpuOut = new CpuDoubleVector(ROWS);
        cpu.apply(new CpuDoubleVector(x.clone()), cpuOut);

        GpuDoubleVector gpuOut = new GpuDoubleVector(ex, 0);
        g.apply(gpu(x), gpuOut);
        assertEquals(ROWS, gpuOut.size());
        assertClose(cpuOut.array(), gpuOut.download());
        g.close();
    }

    @Test
    void csrTransposeMatvecMatchesCpu() {
        CpuPaddedCSRMatrixD cpu = new CpuPaddedCSRMatrixD(ROWS, ROWS, NNZ);
        cpu.setExecutor(cpuEx);
        GpuPaddedCSRMatrixD g = new GpuPaddedCSRMatrixD(ROWS, ROWS, NNZ);
        g.setExecutor(ex);
        fillMatrices(cpu, g);

        double[] x = randomD(ROWS, 32);
        CpuDoubleVector cpuOut = new CpuDoubleVector(ROWS);
        cpu.transposeApply(new CpuDoubleVector(x.clone()), cpuOut);

        GpuDoubleVector gpuOut = new GpuDoubleVector(ex, ROWS);
        try {
            g.transposeApply(gpu(x), gpuOut);
        } catch (UnsupportedOperationException e) {
            Assumptions.abort("device lacks 64-bit buffer atomics: " + e.getMessage());
        }
        assertClose(cpuOut.array(), gpuOut.download());
        g.close();
    }

    @Test
    void csrGetValuesReadsStoredEntries() {
        GpuPaddedCSRMatrixD g = new GpuPaddedCSRMatrixD(4, 4, 2);
        g.setExecutor(ex);
        g.setRow(0, new double[]{1.5, 2.5}, new int[]{0, 3}, 2);
        g.setRow(2, new double[]{7.0}, new int[]{2}, 1);

        GpuDoubleVector out = (GpuDoubleVector) g.getValues(gpu(new int[]{0, 0, 2, 2, 1}), gpu(new int[]{0, 3, 2, 0, 1}));
        assertArrayEquals(new double[]{1.5, 2.5, 7.0, 0.0, 0.0}, out.download());
        assertThrows(IllegalArgumentException.class, () -> g.getValues(gpu(new int[]{0}), gpu(new int[]{0, 1})));
        g.close();
    }

    @Test
    void quadraticTensorApplyAndJacobianMatchCpu() {
        final int            eq  = 40, terms = 4;
        CpuPaddedCSR3TensorD cpu = new CpuPaddedCSR3TensorD(eq, terms);
        cpu.setExecutor(cpuEx);
        GpuPaddedCSR3TensorD g = new GpuPaddedCSR3TensorD(eq, terms);
        g.setExecutor(ex);

        // add() only updates entries that already exist, so structure is defined row by row with setRow
        Random r = new Random(33);
        for (int e = 0; e < eq; e++) {
            double[] vals = new double[terms];
            int[] var1 = new int[terms], var2 = new int[terms];
            for (int t = 0; t < terms; t++) {
                vals[t] = r.nextDouble() * 2 - 1;
                var1[t] = r.nextInt(eq);
                var2[t] = r.nextInt(eq); // var1 == var2 happens: covers the 2*c*x*dx case
            }
            var2[0] = var1[0]; // guarantee at least one squared term per equation
            cpu.setRow(e, vals, var1, var2, terms);
            g.setRow(e, vals, var1, var2, terms);
        }

        double[]        x      = randomD(eq, 34), d = randomD(eq, 35);
        CpuDoubleVector cpuOut = new CpuDoubleVector(eq), cpuJ = new CpuDoubleVector(eq);
        cpu.apply(new CpuDoubleVector(x.clone()), cpuOut);
        cpu.multiplyJacobian(new CpuDoubleVector(x.clone()), new CpuDoubleVector(d.clone()), cpuJ);

        GpuDoubleVector out = new GpuDoubleVector(ex, eq), jac = new GpuDoubleVector(ex, eq);
        g.apply(gpu(x), out);
        g.applyJacobian(gpu(x), gpu(d), jac);
        assertClose(cpuOut.array(), out.download());
        assertClose(cpuJ.array(), jac.download());
        g.close();
    }

    @Test
    void orderNTensorApplyAndJacobianMatchCpu() {
        final int           order = 4, eq = 30, terms = 3;
        CpuPaddedCSRTensorD cpu   = new CpuPaddedCSRTensorD(order, eq, terms);
        cpu.setExecutor(cpuEx);
        GpuPaddedCSRTensorD g = new GpuPaddedCSRTensorD(order, eq, terms);
        g.setExecutor(ex);

        Random r = new Random(36);
        for (int e = 0; e < eq; e++) {
            int count = 1 + r.nextInt(terms); // also exercises padded (inactive) terms
            double[] vals = new double[count];
            int[][] vars = new int[count][order - 1]; // [term][order-1]
            for (int t = 0; t < count; t++) {
                vals[t] = r.nextDouble() * 2 - 1;
                for (int d = 0; d < order - 1; d++)
                    vars[t][d] = r.nextInt(eq);
            }
            cpu.setRow(e, vals, vars, count);
            g.setRow(e, vals, vars, count);
        }

        double[]        x      = randomD(eq, 37), d = randomD(eq, 38);
        CpuDoubleVector cpuOut = new CpuDoubleVector(eq), cpuJ = new CpuDoubleVector(eq);
        cpu.apply(new CpuDoubleVector(x.clone()), cpuOut);
        cpu.applyJacobian(new CpuDoubleVector(x.clone()), new CpuDoubleVector(d.clone()), cpuJ);

        GpuDoubleVector out = new GpuDoubleVector(ex, eq), jac = new GpuDoubleVector(ex, eq);
        g.apply(gpu(x), out);
        g.applyJacobian(gpu(x), gpu(d), jac);
        assertClose(cpuOut.array(), out.download());
        assertClose(cpuJ.array(), jac.download());
        g.close();
    }
}