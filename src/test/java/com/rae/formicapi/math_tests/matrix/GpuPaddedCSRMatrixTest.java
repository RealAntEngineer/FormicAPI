package com.rae.formicapi.math_tests.matrix;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuPaddedCSRMatrix;
import com.rae.formicapi.math_tests.gpu.GpuTestSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Correctness tests for {@link GpuPaddedCSRMatrix}, mirroring
 * {@code GpuPaddedCSRTensorTest}'s structure for the linear (order-2)
 * counterpart.
 *
 * <p>{@link #applyMatchesExpected()} reuses the same 3x3 SPD example as
 * {@code MatrixConsistencyTest} (dense vs. {@code HashSparseMatrix}) so the
 * three backends - dense, CPU sparse, GPU sparse - are all checked against
 * the same hand-computed {@code Ax}.
 *
 * <p>{@link #transposeApplyMatchesExpected()} and
 * {@link #applyMatchesExpectedRectangular()} instead use a small
 * <em>non-symmetric</em>, non-square (rows != cols) matrix, since a
 * symmetric example can't distinguish a correct {@code transposeApply} from
 * one that accidentally computes {@code apply} again.
 *
 * <p>{@code transposeApply} requires a device with
 * {@code cl_khr_int64_base_atomics} (see {@code GpuExecutor}'s
 * {@code ATOMIC_KERNEL_SOURCE} javadoc); tests exercising it are skipped via
 * {@code assumeTrue} rather than failed when the test device doesn't report
 * that extension.
 */
class GpuPaddedCSRMatrixTest extends GpuTestSupport {

    @Test
    void constructionAndMetadata() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 3, 3);
        matrix.setExecutor(executor);

        assertEquals(3, matrix.rows());
        assertEquals(3, matrix.cols());
        assertEquals(3, matrix.nnzPerRow());
    }

    @Test
    void constructionRejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSRMatrix(-1, 3, 3));
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSRMatrix(3, -1, 3));
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSRMatrix(3, 3, 0));
    }

    @Test
    void setRowAndGetMatchExpected() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 3, 2);
        matrix.setExecutor(executor);

        matrix.setRow(0, new double[]{4.0, -1.0}, new int[]{0, 1}, 2);

        assertEquals(4.0, matrix.get(0, 0), EPSILON);
        assertEquals(-1.0, matrix.get(0, 1), EPSILON);
        assertEquals(0.0, matrix.get(0, 2), EPSILON); // never set
    }

    @Test
    void setRowZeroPadsUnusedEntries() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 3, 3);
        matrix.setExecutor(executor);

        matrix.setRow(1, new double[]{-1.0}, new int[]{0}, 1);

        int base = 1 * matrix.nnzPerRow();
        double[] values = matrix.getValues();
        int[] colIndex = matrix.getColIndex();

        assertArrayEquals(new double[]{-1.0, 0.0, 0.0},
                new double[]{values[base], values[base + 1], values[base + 2]}, EPSILON);

        // Padded entries point at row (row < cols here), matching CSR's
        // "unused slot contributes zero regardless of which valid column it
        // reads" convention.
        assertEquals(0, colIndex[base]);
        assertEquals(1, colIndex[base + 1]);
        assertEquals(1, colIndex[base + 2]);
    }

    @Test
    void setRowRejectsCountOutOfRange() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 3, 2);
        matrix.setExecutor(executor);

        assertThrows(IllegalArgumentException.class,
                () -> matrix.setRow(0, new double[]{1.0, 2.0, 3.0}, new int[]{0, 1, 2}, 3)); // count > nnzPerRow
    }

    @Test
    void addAccumulatesOnExistingEntry() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 2, 1);
        matrix.setExecutor(executor);

        matrix.setRow(0, new double[]{2.0}, new int[]{1}, 1);

        matrix.add(0, 1, 3.0);
        assertEquals(5.0, matrix.get(0, 1), EPSILON);
    }

    @Test
    void addOnMissingEntryThrows() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 2, 1);
        matrix.setExecutor(executor);

        matrix.setRow(0, new double[]{2.0}, new int[]{1}, 1);

        assertThrows(IllegalArgumentException.class, () -> matrix.add(0, 0, 1.0));
    }

    /**
     * Same 3x3 SPD example (a 1-D Poisson stencil) as
     * {@code MatrixConsistencyTest}'s dense-vs-CPU-sparse check:
     * <pre>
     *  [ 4 -1  0 ]   [1]   [2]
     *  [-1  4 -1 ] * [2] = [4]
     *  [ 0 -1  3 ]   [3]   [7]
     * </pre>
     */
    @Test
    void applyMatchesExpected() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(3, 3, 3);
        matrix.setExecutor(executor);

        matrix.setRow(0, new double[]{4.0, -1.0}, new int[]{0, 1}, 2);
        matrix.setRow(1, new double[]{-1.0, 4.0, -1.0}, new int[]{0, 1, 2}, 3);
        matrix.setRow(2, new double[]{-1.0, 3.0}, new int[]{1, 2}, 2);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{1.0, 2.0, 3.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        matrix.apply(x, result);

        assertEquals(3, result.size());
        assertArrayEquals(new double[]{2.0, 4.0, 7.0}, result.download(), EPSILON);
    }

    /**
     * Non-symmetric 2x3 example: {@code A = [[1,2,0],[0,3,4]]}. Used by both
     * {@link #transposeApplyMatchesExpected()} and rectangular {@code apply}
     * coverage below, since a square symmetric matrix can't tell a correct
     * {@code transposeApply} apart from a bugged one that just calls
     * {@code apply} again.
     */
    private GpuPaddedCSRMatrix createRectangularMatrix() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 3, 2);
        matrix.setExecutor(executor);

        matrix.setRow(0, new double[]{1.0, 2.0}, new int[]{0, 1}, 2);
        matrix.setRow(1, new double[]{3.0, 4.0}, new int[]{1, 2}, 2);

        return matrix;
    }

    @Test
    void applyMatchesExpectedRectangular() {
        GpuPaddedCSRMatrix matrix = createRectangularMatrix();

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{1.0, 1.0, 1.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        matrix.apply(x, result);

        assertEquals(2, result.size());
        assertArrayEquals(new double[]{3.0, 7.0}, result.download(), EPSILON);
    }

    @Test
    void transposeApplyMatchesExpected() {
        assumeTrue(executor.supportsScatterAtomics(),
                "test device lacks cl_khr_int64_base_atomics; transposeApply is unsupported here");

        GpuPaddedCSRMatrix matrix = createRectangularMatrix();

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{1.0, 1.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        matrix.transposeApply(x, result);

        assertEquals(3, result.size());
        assertArrayEquals(new double[]{1.0, 5.0, 4.0}, result.download(), EPSILON);
    }

    @Test
    void applyRejectsWrongVectorType() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 2, 1);
        matrix.setExecutor(executor);
        matrix.setRow(0, new double[]{1.0}, new int[]{0}, 1);
        matrix.setRow(1, new double[]{1.0}, new int[]{1}, 1);

        CpuDoubleVector x = new CpuDoubleVector(new double[]{1.0, 1.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        assertThrows(UnsupportedOperationException.class, () -> matrix.apply(x, result));
    }

    @Test
    void applyRejectsMismatchedSize() {
        GpuPaddedCSRMatrix matrix = new GpuPaddedCSRMatrix(2, 3, 1);
        matrix.setExecutor(executor);
        matrix.setRow(0, new double[]{1.0}, new int[]{0}, 1);
        matrix.setRow(1, new double[]{1.0}, new int[]{1}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{1.0, 1.0}); // size 2, cols() == 3
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        assertThrows(IllegalArgumentException.class, () -> matrix.apply(x, result));
    }

    @Test
    void resizeGrowKeepsExistingRowsAndZeroFillsNewOnes() {
        GpuPaddedCSRMatrix matrix = createRectangularMatrix(); // 2 rows x 3 cols

        matrix.resize(3);
        assertEquals(3, matrix.rows());

        // Old rows preserved.
        assertEquals(1.0, matrix.get(0, 0), EPSILON);
        assertEquals(4.0, matrix.get(1, 2), EPSILON);

        // New row has no entries yet.
        assertEquals(0.0, matrix.get(2, 0), EPSILON);
        assertEquals(0.0, matrix.get(2, 1), EPSILON);
        assertEquals(0.0, matrix.get(2, 2), EPSILON);

        matrix.setRow(2, new double[]{5.0}, new int[]{2}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{1.0, 1.0, 1.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);
        matrix.apply(x, result);

        assertArrayEquals(new double[]{3.0, 7.0, 5.0}, result.download(), EPSILON);
    }

    @Test
    void resizeShrinkTruncatesRows() {
        GpuPaddedCSRMatrix matrix = createRectangularMatrix(); // 2 rows x 3 cols

        matrix.resize(1);

        assertEquals(1, matrix.rows());
        assertEquals(1.0, matrix.get(0, 0), EPSILON);
        assertThrows(IndexOutOfBoundsException.class, () -> matrix.get(1, 0));
    }
}