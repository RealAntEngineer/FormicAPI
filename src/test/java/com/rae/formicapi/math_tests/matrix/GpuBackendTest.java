package com.rae.formicapi.math_tests.matrix;

import com.rae.formicapi.foundation.math.operators.backend.gpu.*;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for the OpenCL-backed GPU vector classes. Deliberately
 * not a benchmark - no timing assertions, just "does the math on the device
 * match the math you'd expect", the same way you'd sanity-check
 * {@code CpuDoubleVector} against a plain array.
 *
 * <p>Requires a real, fp64-capable OpenCL device to actually run - see
 * {@link #setUp()}. On a machine without one (most CI runners, most laptops
 * without a discrete GPU or vendor OpenCL runtime installed), every test in
 * this class is skipped via {@link Assumptions#assumeTrue}, not failed, so
 * this is safe to leave enabled in a build that doesn't always have GPU
 * hardware available.
 */
class GpuBackendTest {

    private static final double EPSILON = 1e-9;

    private GpuExecutor executor;

    @BeforeEach
    void setUp() {
        try {
            executor = new GpuExecutor();
        } catch (RuntimeException e) {
            executor = null;
            Assumptions.assumeTrue(false, "Skipping GPU tests: no fp64-capable OpenCL device available (" + e.getMessage() + ")");
        }
    }

    @AfterEach
    void tearDown() {
        if (executor != null)
            executor.close();
    }

    // ---------------------------------------------------------------- GpuDoubleVector

    @Test
    void uploadDownloadRoundTrip() {
        double[]        data = {1.0, -2.5, 3.25, 0.0, 42.0};
        GpuDoubleVector v    = new GpuDoubleVector(executor, data);

        assertEquals(data.length, v.size());
        assertArrayEquals(data, v.download(), EPSILON);
    }

    @Test
    void axpyMatchesExpected() {
        double[] a = {1.0, 2.0, 3.0, 4.0};
        double[] x = {10.0, 20.0, 30.0, 40.0};

        GpuDoubleVector va = new GpuDoubleVector(executor, a);
        GpuDoubleVector vx = new GpuDoubleVector(executor, x);

        va.axpy(2.0, vx);

        double[] expected = new double[a.length];
        for (int i = 0; i < a.length; i++)
            expected[i] = a[i] + 2.0 * x[i];

        assertArrayEquals(expected, va.download(), EPSILON);
    }

    @Test
    void scaleMatchesExpected() {
        double[] a = {1.0, -2.0, 3.0, -4.0};
        GpuDoubleVector v = new GpuDoubleVector(executor, a);

        v.scale(1.5);

        double[] expected = new double[a.length];
        for (int i = 0; i < a.length; i++)
            expected[i] = a[i] * 1.5;

        assertArrayEquals(expected, v.download(), EPSILON);
    }

    @Test
    void addScalarMatchesExpected() {
        double[] a = {1.0, 2.0, 3.0};
        GpuDoubleVector v = new GpuDoubleVector(executor, a);

        v.add(7.0);

        double[] expected = {8.0, 9.0, 10.0};
        assertArrayEquals(expected, v.download(), EPSILON);
    }

    @Test
    void addVectorMatchesExpected() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {10.0, 20.0, 30.0};

        GpuDoubleVector va = new GpuDoubleVector(executor, a);
        GpuDoubleVector vb = new GpuDoubleVector(executor, b);

        va.add(vb);

        double[] expected = {11.0, 22.0, 33.0};
        assertArrayEquals(expected, va.download(), EPSILON);
    }

    @Test
    void dotMatchesExpected() {
        double[] a = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] b = {5.0, 4.0, 3.0, 2.0, 1.0};

        GpuDoubleVector va = new GpuDoubleVector(executor, a);
        GpuDoubleVector vb = new GpuDoubleVector(executor, b);

        double expected = 0.0;
        for (int i = 0; i < a.length; i++)
            expected += a[i] * b[i];

        assertEquals(expected, va.dot(vb), EPSILON);
    }

    @Test
    void dotOnLargerVectorExercisesMultipleWorkGroups() {
        // Small vectors only ever touch one reduction work-group; this size
        // is picked to be comfortably larger than any reasonable device's
        // work-group size, so the multi-group partial-sum path in
        // GpuExecutor's dot_partial kernel actually gets exercised too.
        int size = 10_000;
        double[] a = new double[size];
        double[] b = new double[size];
        double expected = 0.0;

        for (int i = 0; i < size; i++) {
            a[i] = (i % 7) - 3;
            b[i] = (i % 5) - 2;
            expected += a[i] * b[i];
        }

        GpuDoubleVector va = new GpuDoubleVector(executor, a);
        GpuDoubleVector vb = new GpuDoubleVector(executor, b);

        assertEquals(expected, va.dot(vb), size * EPSILON);
    }

    @Test
    void clearZeroesTheVector() {
        GpuDoubleVector v = new GpuDoubleVector(executor, new double[]{1.0, 2.0, 3.0});
        v.clear();
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, v.download(), EPSILON);
    }

    @Test
    void copyFromVectorCopiesContentsAndIsIndependent() {
        GpuDoubleVector source = new GpuDoubleVector(executor, new double[]{1.0, 2.0, 3.0});
        GpuDoubleVector dest = new GpuDoubleVector(executor, 3);

        dest.copy(source);
        assertArrayEquals(source.download(), dest.download(), EPSILON);

        source.scale(100.0);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, dest.download(), EPSILON,
                "copy(x) should be a snapshot, not a live view - mutating the source afterwards shouldn't affect dest");
    }

    @Test
    void cloneCopyIsIndependent() {
        GpuDoubleVector original = new GpuDoubleVector(executor, new double[]{1.0, 2.0, 3.0});
        GpuDoubleVector clone = (GpuDoubleVector) original.copy();

        assertArrayEquals(original.download(), clone.download(), EPSILON);

        original.scale(100.0);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, clone.download(), EPSILON);
    }

    @Test
    void resizeGrowPreservesExistingElementsAndZeroPadsTheRest() {
        GpuDoubleVector v = new GpuDoubleVector(executor, new double[]{1.0, 2.0, 3.0});
        v.resize(6);

        assertEquals(6, v.size());
        assertArrayEquals(new double[]{1.0, 2.0, 3.0, 0.0, 0.0, 0.0}, v.download(), EPSILON);
    }

    @Test
    void resizeShrinkTruncates() {
        GpuDoubleVector v = new GpuDoubleVector(executor, new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        v.resize(2);

        assertEquals(2, v.size());
        assertArrayEquals(new double[]{1.0, 2.0}, v.download(), EPSILON);
    }

    @Test
    void operationsWithoutAttachedExecutorThrow() {
        GpuDoubleVector v = new GpuDoubleVector(executor, new double[]{1.0, 2.0});
        v.setExecutor(null);

        assertThrows(IllegalStateException.class, v::clear);
    }

    @Test
    void operandsFromDifferentExecutorsAreRejected() {
        try (GpuExecutor otherExecutor = new GpuExecutor()) {
            GpuDoubleVector v1 = new GpuDoubleVector(executor, new double[]{1.0, 2.0});
            GpuDoubleVector v2 = new GpuDoubleVector(otherExecutor, new double[]{3.0, 4.0});

            assertThrows(UnsupportedOperationException.class, () -> v1.axpy(1.0, v2));
        }
    }

    // ---------------------------------------------------------------- GpuIntegerVector

    @Test
    void integerVectorUploadDownloadRoundTrip() {
        int[] data = {5, -3, 0, 42, 17};
        GpuIntegerVector v = new GpuIntegerVector(executor, data);

        assertEquals(data.length, v.size());
        assertArrayEquals(data, v.download());
    }

    @Test
    void integerVectorSingleElementGetSet() {
        GpuIntegerVector v = new GpuIntegerVector(executor, new int[]{1, 2, 3});

        v.set(99, 1);
        assertEquals(99, v.get(1));
        assertEquals(1, v.get(0));
        assertEquals(3, v.get(2));
    }

    @Test
    void integerVectorCopyIsIndependent() {
        GpuIntegerVector original = new GpuIntegerVector(executor, new int[]{1, 2, 3});
        GpuIntegerVector clone = (GpuIntegerVector) original.copy();

        original.set(999, 0);
        assertEquals(1, clone.get(0));
    }

    // ---------------------------------------------------------------- GpuBooleanVector

    @Test
    void booleanVectorUploadDownloadRoundTrip() {
        boolean[] data = {true, false, false, true, true};
        GpuBooleanVector v = new GpuBooleanVector(executor, data);

        assertEquals(data.length, v.size());
        assertArrayEquals(data, v.download());
    }

    @Test
    void booleanVectorSingleElementGetSet() {
        GpuBooleanVector v = new GpuBooleanVector(executor, new boolean[]{false, false, false});

        v.set(true, 1);
        assertTrue(v.get(1));
        assertFalse(v.get(0));
        assertFalse(v.get(2));
    }

    @Test
    void booleanVectorClearResetsToFalse() {
        GpuBooleanVector v = new GpuBooleanVector(executor, new boolean[]{true, true, true});
        v.clear();
        assertArrayEquals(new boolean[]{false, false, false}, v.download());
    }

    // ---------------------------------------------------------------- gather/scatter (DoubleVector + IntegerVector)

    @Test
    void scatterAxpyMatchesExpected() {
        double[] target = {0.0, 0.0, 0.0, 0.0, 0.0};
        double[] source = {1.0, 2.0, 3.0};
        int[]    idx    = {4, 1, 2}; // distinct targets - see GpuExecutor's scatter_axpy kernel doc on duplicate targets

        GpuDoubleVector vTarget = new GpuDoubleVector(executor, target);
        GpuDoubleVector vSource = new GpuDoubleVector(executor, source);
        GpuIntegerVector vIdx = new GpuIntegerVector(executor, idx);

        vTarget.scatterAxpy(2.0, vSource, vIdx);

        double[] expected = target.clone();
        for (int i = 0; i < idx.length; i++)
            expected[idx[i]] += 2.0 * source[i];

        assertArrayEquals(expected, vTarget.download(), EPSILON);
    }

    @Test
    void skippedDotMatchesExpected() {
        double[] a   = {10.0, 20.0, 30.0, 40.0, 50.0};
        double[] b   = {1.0, 2.0, 3.0};
        int[]    idx = {4, 0, 2}; // a[idx[i]] * b[i]

        GpuDoubleVector va = new GpuDoubleVector(executor, a);
        GpuDoubleVector vb = new GpuDoubleVector(executor, b);
        GpuIntegerVector vIdx = new GpuIntegerVector(executor, idx);

        double expected = 0.0;
        for (int i = 0; i < idx.length; i++)
            expected += a[idx[i]] * b[i];

        assertEquals(expected, va.skippedDot(vb, vIdx), EPSILON);
    }

    // ---------------------------------------------------------------- GpuPaddedCSRMatrix

    @Test
    void matrixConstructionAndMetadata() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 4, 2);
        matrix.setExecutor(executor);

        assertEquals(3, matrix.rows());
        assertEquals(4, matrix.cols());
        assertEquals(2, matrix.nnzPerRow());
        assertEquals(6, matrix.getValues().length);
        assertEquals(6, matrix.getColIndex().length);
    }

    @Test
    void matrixSetRowAndGetMatchExpected() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 5, 3);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{2.0, 4.0},
                new int[]{1, 3},
                2
        );

        matrix.setRow(
                1,
                new double[]{5.0, -1.0, 7.0},
                new int[]{0, 2, 4},
                3
        );

        assertEquals(2.0, matrix.get(0, 1), EPSILON);
        assertEquals(4.0, matrix.get(0, 3), EPSILON);
        assertEquals(0.0, matrix.get(0, 0), EPSILON);

        assertEquals(5.0, matrix.get(1, 0), EPSILON);
        assertEquals(-1.0, matrix.get(1, 2), EPSILON);
        assertEquals(7.0, matrix.get(1, 4), EPSILON);
        assertEquals(0.0, matrix.get(1, 1), EPSILON);
    }

    @Test
    void matrixSetRowZeroPadsUnusedEntries() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 5, 3);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{2.0},
                new int[]{4},
                1
        );

        assertArrayEquals(
                new double[]{2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                matrix.getValues(),
                EPSILON
        );

        // The padded column indices are implementation details and should not
        // contribute because their corresponding values are zero.
        assertEquals(0.0, matrix.get(0, 0), EPSILON);
        assertEquals(0.0, matrix.get(0, 1), EPSILON);
        assertEquals(0.0, matrix.get(0, 2), EPSILON);
        assertEquals(0.0, matrix.get(0, 3), EPSILON);
        assertEquals(2.0, matrix.get(0, 4), EPSILON);
    }

    @Test
    void matrixAddAndSetMatchExpected() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 3, 2);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{2.0, 3.0},
                new int[]{0, 2},
                2
        );

        matrix.set(0, 0, 10.0);
        assertEquals(10.0, matrix.get(0, 0), EPSILON);

        matrix.add(0, 0, 5.0);
        assertEquals(15.0, matrix.get(0, 0), EPSILON);

        matrix.add(0, 2, -1.0);
        assertEquals(2.0, matrix.get(0, 2), EPSILON);
    }

    @Test
    void matrixApplyMatchesExpected() {
        /*
         *     [ 2  0  3  0 ]
         * A = [ 0 -1  0  4 ]
         *     [ 5  0  0  6 ]
         */
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 4, 2);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{2.0, 3.0},
                new int[]{0, 2},
                2
        );

        matrix.setRow(
                1,
                new double[]{-1.0, 4.0},
                new int[]{1, 3},
                2
        );

        matrix.setRow(
                2,
                new double[]{5.0, 6.0},
                new int[]{0, 3},
                2
        );

        GpuDoubleVector x = new GpuDoubleVector(
                executor,
                new double[]{1.0, 2.0, 3.0, 4.0}
        );

        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        matrix.apply(x, result);

        assertEquals(3, result.size());
        assertArrayEquals(
                new double[]{
                        2.0 * 1.0 + 3.0 * 3.0,
                        -1.0 * 2.0 + 4.0 * 4.0,
                        5.0 * 1.0 + 6.0 * 4.0
                },
                result.download(),
                EPSILON
        );
    }

    @Test
    void matrixTransposeApplyMatchesExpected() {
        /*
         *     [ 2  0  3  0 ]
         * A = [ 0 -1  0  4 ]
         *     [ 5  0  0  6 ]
         *
         * A^T y =
         * [ 2*y0 + 5*y2,
         *  -y1,
         *  3*y0,
         *  4*y1 + 6*y2 ]
         */
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 4, 2);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{2.0, 3.0},
                new int[]{0, 2},
                2
        );

        matrix.setRow(
                1,
                new double[]{-1.0, 4.0},
                new int[]{1, 3},
                2
        );

        matrix.setRow(
                2,
                new double[]{5.0, 6.0},
                new int[]{0, 3},
                2
        );

        GpuDoubleVector x = new GpuDoubleVector(
                executor,
                new double[]{10.0, 20.0, 30.0}
        );

        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        matrix.transposeApply(x, result);

        assertEquals(4, result.size());
        assertArrayEquals(
                new double[]{
                        2.0 * 10.0 + 5.0 * 30.0,
                        -1.0 * 20.0,
                        3.0 * 10.0,
                        4.0 * 20.0 + 6.0 * 30.0
                },
                result.download(),
                EPSILON
        );
    }

    @Test
    void matrixApplyHandlesPaddedRows() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 4, 4);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{2.0, 3.0},
                new int[]{0, 2},
                2
        );

        matrix.setRow(
                1,
                new double[]{4.0},
                new int[]{3},
                1
        );

        GpuDoubleVector x = new GpuDoubleVector(
                executor,
                new double[]{1.0, 2.0, 3.0, 4.0}
        );

        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        matrix.apply(x, result);

        assertArrayEquals(
                new double[]{
                        2.0 * 1.0 + 3.0 * 3.0,
                        4.0 * 4.0
                },
                result.download(),
                EPSILON
        );
    }

    @Test
    void matrixResizePreservesExistingRows() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 4, 2);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{2.0, 3.0},
                new int[]{0, 2},
                2
        );

        matrix.setRow(
                1,
                new double[]{4.0},
                new int[]{3},
                1
        );

        matrix.resize(4);

        assertEquals(4, matrix.rows());
        assertEquals(4, matrix.cols());
        assertEquals(2, matrix.nnzPerRow());

        assertEquals(2.0, matrix.get(0, 0), EPSILON);
        assertEquals(3.0, matrix.get(0, 2), EPSILON);
        assertEquals(4.0, matrix.get(1, 3), EPSILON);

        assertEquals(0.0, matrix.get(2, 0), EPSILON);
        assertEquals(0.0, matrix.get(3, 0), EPSILON);
    }

    @Test
    void matrixResizeShrinkTruncatesRows() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(4, 3, 2);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{1.0},
                new int[]{0},
                1
        );

        matrix.setRow(
                1,
                new double[]{2.0},
                new int[]{1},
                1
        );

        matrix.setRow(
                2,
                new double[]{3.0},
                new int[]{2},
                1
        );

        matrix.setRow(
                3,
                new double[]{4.0},
                new int[]{0},
                1
        );

        matrix.resize(2);

        assertEquals(2, matrix.rows());
        assertEquals(1.0, matrix.get(0, 0), EPSILON);
        assertEquals(2.0, matrix.get(1, 1), EPSILON);

        assertThrows(
                IndexOutOfBoundsException.class,
                () -> matrix.get(2, 0)
        );
    }

    @Test
    void matrixRejectsInvalidDimensions() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new GpuPaddedCSRMatrix(-1, 3, 2)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new GpuPaddedCSRMatrix(3, -1, 2)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new GpuPaddedCSRMatrix(3, 3, 0)
        );
    }

    @Test
    void matrixRejectsInvalidRowsAndSetRowCounts() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 3, 2);
        matrix.setExecutor(executor);

        assertThrows(
                IndexOutOfBoundsException.class,
                () -> matrix.get(-1, 0)
        );

        assertThrows(
                IndexOutOfBoundsException.class,
                () -> matrix.get(2, 0)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> matrix.setRow(
                        0,
                        new double[]{1.0},
                        new int[]{0},
                        3
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> matrix.setRow(
                        0,
                        new double[]{},
                        new int[]{0},
                        1
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> matrix.setRow(
                        0,
                        new double[]{1.0},
                        new int[]{},
                        1
                )
        );
    }

    @Test
    void matrixApplyRejectsWrongVectorSize() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 3, 2);
        matrix.setExecutor(executor);

        GpuDoubleVector x = new GpuDoubleVector(
                executor,
                new double[]{1.0, 2.0}
        );

        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        assertThrows(
                IllegalArgumentException.class,
                () -> matrix.apply(x, result)
        );
    }

    @Test
    void matrixTransposeApplyRejectsWrongVectorSize() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 3, 2);
        matrix.setExecutor(executor);

        GpuDoubleVector x = new GpuDoubleVector(
                executor,
                new double[]{1.0, 2.0, 3.0}
        );

        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        assertThrows(
                IllegalArgumentException.class,
                () -> matrix.transposeApply(x, result)
        );
    }

    @Test
    void matrixGetReturnsZeroForMissingEntry() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 4, 2);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{5.0},
                new int[]{2},
                1
        );

        assertEquals(5.0, matrix.get(0, 2), EPSILON);
        assertEquals(0.0, matrix.get(0, 0), EPSILON);
        assertEquals(0.0, matrix.get(0, 1), EPSILON);
        assertEquals(0.0, matrix.get(0, 3), EPSILON);
    }

    @Test
    void matrixSetAndAddMissingEntryThrow() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 4, 2);
        matrix.setExecutor(executor);

        matrix.setRow(
                0,
                new double[]{5.0},
                new int[]{2},
                1
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> matrix.set(0, 1, 10.0)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> matrix.add(0, 1, 10.0)
        );
    }
}