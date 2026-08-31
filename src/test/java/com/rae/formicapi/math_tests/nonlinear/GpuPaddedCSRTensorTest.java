package com.rae.formicapi.math_tests.nonlinear;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuPaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuPaddedCSRTensor;
import com.rae.formicapi.math_tests.gpu.GpuTestSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for {@link GpuPaddedCSRTensor} - the general N-order
 * tensor {@code F_i(x) = sum(c * x[j1] * x[j2] * ... * x[jN-1])}. Uses
 * order 4 (cubic: three x-factors per term) throughout, rather than order 3
 * (which would just be re-testing {@link GpuPaddedCSR3Tensor}'s quadratic
 * case through more general machinery) - and one term with a repeated
 * variable index, to exercise the Jacobian's per-position derivative sum
 * for {@code j == k} in more than two dimensions. Not a benchmark: no
 * timing assertions.
 */
class GpuPaddedCSRTensorTest extends GpuTestSupport {

    private static final int ORDER = 4; // 3 x-factors per term

    @Test
    void constructionAndMetadata() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 3, 2);
        tensor.setExecutor(executor);

        assertEquals(3, tensor.equations());
        assertEquals(2, tensor.termsPerEquation());
        assertEquals(ORDER, tensor.order());
    }

    @Test
    void constructionRejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSRTensor(1, 2, 2)); // order < 2
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSRTensor(ORDER, -1, 2));
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSRTensor(ORDER, 2, 0));
    }

    @Test
    void setRowAndGetRowVarIndicesMatchExpected() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 1, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{1.0, 2.0},
                new int[][]{{0, 1, 2}, {1, 1, 0}}, 2);

        assertArrayEquals(new double[]{1.0, 2.0}, tensor.getRowValues(0), EPSILON);

        int[][] varIndices = tensor.getRowVarIndices(0);
        assertArrayEquals(new int[]{0, 1, 2}, varIndices[0]);
        assertArrayEquals(new int[]{1, 1, 0}, varIndices[1]);
    }

    @Test
    void setRowZeroPadsUnusedEntries() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 1, 3);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{5.0}, new int[][]{{0, 1, 2}}, 1);

        assertArrayEquals(new double[]{5.0, 0.0, 0.0}, tensor.getRowValues(0), EPSILON);
        assertArrayEquals(new int[]{0, 0, 0}, tensor.getRowVarIndices(0)[1]);
        assertArrayEquals(new int[]{0, 0, 0}, tensor.getRowVarIndices(0)[2]);
    }

    @Test
    void setRowRejectsWrongOrderIndices() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 1, 2);
        tensor.setExecutor(executor);

        assertThrows(IllegalArgumentException.class,
                () -> tensor.setRow(0, new double[]{1.0}, new int[][]{{0, 1}}, 1)); // only 2 indices, need 3
    }

    @Test
    void addAccumulatesOnExistingTerm() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 1, 1);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0}, new int[][]{{0, 1, 2}}, 1);

        tensor.add(0, 3.0, 0, 1, 2);
        assertEquals(5.0, tensor.getRowValues(0)[0], EPSILON);
    }

    @Test
    void addOnMissingTermThrows() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 1, 1);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0}, new int[][]{{0, 1, 2}}, 1);

        assertThrows(IllegalStateException.class, () -> tensor.add(0, 1.0, 9, 9, 9));
    }

    @Test
    void addRejectsWrongOrderVarargs() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 1, 1);
        tensor.setExecutor(executor);
        tensor.setRow(0, new double[]{2.0}, new int[][]{{0, 1, 2}}, 1);

        assertThrows(IllegalArgumentException.class, () -> tensor.add(0, 1.0, 0, 1)); // needs 3 indices
    }

    @Test
    void applyMatchesExpected() {
        // eq0: F_0 = 1*x0*x1*x2 + 2*x1*x1*x0
        // eq1: F_1 = 3*x2*x0*x1
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{1.0, 2.0}, new int[][]{{0, 1, 2}, {1, 1, 0}}, 2);
        tensor.setRow(1, new double[]{3.0}, new int[][]{{2, 0, 1}}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{2.0, 3.0, 5.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        tensor.apply(x, result);

        assertEquals(2, result.size());
        assertArrayEquals(new double[]{66.0, 90.0}, result.download(), EPSILON);
    }

    @Test
    void applyJacobianMatchesExpected() {
        // Same structure as applyMatchesExpected.
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{1.0, 2.0}, new int[][]{{0, 1, 2}, {1, 1, 0}}, 2);
        tensor.setRow(1, new double[]{3.0}, new int[][]{{2, 0, 1}}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{2.0, 3.0, 5.0});
        GpuDoubleVector direction = new GpuDoubleVector(executor, new double[]{1.0, 0.0, 0.5});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        tensor.applyJacobian(x, direction, result);

        assertEquals(2, result.size());
        assertArrayEquals(new double[]{36.0, 54.0}, result.download(), EPSILON);
    }

    @Test
    void applyRejectsWrongVectorType() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 1, 1);
        tensor.setExecutor(executor);
        tensor.setRow(0, new double[]{1.0}, new int[][]{{0, 1, 2}}, 1);

        CpuDoubleVector x = new CpuDoubleVector(new double[]{1.0, 1.0, 1.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        assertThrows(UnsupportedOperationException.class, () -> tensor.apply(x, result));
    }

    @Test
    void resizeGrowKeepsExistingRowsAndZeroFillsNewOnes() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{1.0, 2.0}, new int[][]{{0, 1, 2}, {1, 1, 0}}, 2);
        tensor.setRow(1, new double[]{3.0}, new int[][]{{2, 0, 1}}, 1);

        tensor.resize(3);
        assertEquals(3, tensor.equations());
        assertArrayEquals(new double[]{0.0, 0.0}, tensor.getRowValues(2), EPSILON);

        tensor.setRow(2, new double[]{1.0}, new int[][]{{0, 0, 0}}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{2.0, 3.0, 5.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);
        tensor.apply(x, result);

        assertArrayEquals(new double[]{66.0, 90.0, 8.0}, result.download(), EPSILON); // 8 = 2^3
    }

    @Test
    void resizeShrinkTruncatesRows() {
        GpuPaddedCSRTensor tensor = new GpuPaddedCSRTensor(ORDER, 2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{1.0, 2.0}, new int[][]{{0, 1, 2}, {1, 1, 0}}, 2);
        tensor.setRow(1, new double[]{3.0}, new int[][]{{2, 0, 1}}, 1);

        tensor.resize(1);

        assertEquals(1, tensor.equations());
        assertArrayEquals(new double[]{1.0, 2.0}, tensor.getRowValues(0), EPSILON);
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.getRowValues(1));
    }
}