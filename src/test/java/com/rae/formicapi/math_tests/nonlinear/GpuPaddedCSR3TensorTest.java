package com.rae.formicapi.math_tests.nonlinear;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuPaddedCSR3Tensor;
import com.rae.formicapi.math_tests.gpu.GpuTestSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for {@link GpuPaddedCSR3Tensor} - the fixed-structure
 * quadratic tensor {@code F_i(x) = sum(c[i,j,k] * x[j] * x[k])}. Not a
 * benchmark: no timing assertions.
 */
class GpuPaddedCSR3TensorTest extends GpuTestSupport {

    @Test
    void constructionAndMetadata() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(3, 2);
        tensor.setExecutor(executor);

        assertEquals(3, tensor.equations());
        assertEquals(2, tensor.termsPerEquation());
    }

    @Test
    void constructionRejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSR3Tensor(-1, 2));
        assertThrows(IllegalArgumentException.class, () -> new GpuPaddedCSR3Tensor(2, 0));
    }

    @Test
    void setRowAndGetRowMatchExpected() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0, 3.0}, new int[]{0, 1}, new int[]{1, 1}, 2);

        assertArrayEquals(new double[]{2.0, 3.0}, tensor.getRowValues(0), EPSILON);
        assertArrayEquals(new int[]{0, 1}, tensor.getRowVar1(0));
        assertArrayEquals(new int[]{1, 1}, tensor.getRowVar2(0));
    }

    @Test
    void setRowZeroPadsUnusedEntries() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(1, 3);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{5.0}, new int[]{0}, new int[]{0}, 1);

        assertArrayEquals(new double[]{5.0, 0.0, 0.0}, tensor.getRowValues(0), EPSILON);
    }

    @Test
    void setRowRejectsInvalidCounts() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(1, 2);
        tensor.setExecutor(executor);

        assertThrows(IllegalArgumentException.class,
                () -> tensor.setRow(0, new double[]{1.0}, new int[]{0}, new int[]{0}, 3));

        assertThrows(IllegalArgumentException.class,
                () -> tensor.setRow(0, new double[]{}, new int[]{0}, new int[]{0}, 1));
    }

    @Test
    void addAccumulatesOnExistingTerm() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(1, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0}, new int[]{0}, new int[]{1}, 1);

        tensor.add(0, 0, 1, 3.0);
        assertEquals(5.0, tensor.getRowValues(0)[0], EPSILON);
    }

    @Test
    void addOnMissingTermThrows() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(1, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0}, new int[]{0}, new int[]{1}, 1);

        assertThrows(IllegalStateException.class, () -> tensor.add(0, 9, 9, 1.0));
    }

    @Test
    void addWithZeroValueOnMissingTermIsANoOp() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(1, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0}, new int[]{0}, new int[]{1}, 1);

        // value == 0.0 short-circuits before the entry-existence check.
        assertDoesNotThrow(() -> tensor.add(0, 9, 9, 0.0));
    }

    @Test
    void applyMatchesExpected() {
        // eq0: F_0 = 2*x0*x1 + 3*x1*x1
        // eq1: F_1 = 1*x0*x0
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0, 3.0}, new int[]{0, 1}, new int[]{1, 1}, 2);
        tensor.setRow(1, new double[]{1.0}, new int[]{0}, new int[]{0}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{2.0, 3.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        tensor.apply(x, result);

        assertEquals(2, result.size());
        assertArrayEquals(new double[]{39.0, 4.0}, result.download(), EPSILON);
    }

    @Test
    void applyJacobianMatchesExpected() {
        // Same structure as applyMatchesExpected.
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0, 3.0}, new int[]{0, 1}, new int[]{1, 1}, 2);
        tensor.setRow(1, new double[]{1.0}, new int[]{0}, new int[]{0}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{2.0, 3.0});
        GpuDoubleVector direction = new GpuDoubleVector(executor, new double[]{1.0, 0.5});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        tensor.applyJacobian(x, direction, result);

        assertEquals(2, result.size());
        assertArrayEquals(new double[]{17.0, 4.0}, result.download(), EPSILON);
    }

    @Test
    void applyRejectsWrongVectorType() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(1, 1);
        tensor.setExecutor(executor);
        tensor.setRow(0, new double[]{1.0}, new int[]{0}, new int[]{0}, 1);

        CpuDoubleVector x = new CpuDoubleVector(new double[]{1.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);

        assertThrows(UnsupportedOperationException.class, () -> tensor.apply(x, result));
    }

    @Test
    void resizeGrowKeepsExistingRowsAndZeroFillsNewOnes() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0, 3.0}, new int[]{0, 1}, new int[]{1, 1}, 2);
        tensor.setRow(1, new double[]{1.0}, new int[]{0}, new int[]{0}, 1);

        tensor.resize(3);
        assertEquals(3, tensor.equations());
        assertArrayEquals(new double[]{0.0, 0.0}, tensor.getRowValues(2), EPSILON);

        tensor.setRow(2, new double[]{5.0}, new int[]{0}, new int[]{1}, 1);

        GpuDoubleVector x = new GpuDoubleVector(executor, new double[]{2.0, 3.0});
        GpuDoubleVector result = new GpuDoubleVector(executor, 0);
        tensor.apply(x, result);

        assertArrayEquals(new double[]{39.0, 4.0, 30.0}, result.download(), EPSILON);
    }

    @Test
    void resizeShrinkTruncatesRows() {
        GpuPaddedCSR3Tensor tensor = new GpuPaddedCSR3Tensor(2, 2);
        tensor.setExecutor(executor);

        tensor.setRow(0, new double[]{2.0, 3.0}, new int[]{0, 1}, new int[]{1, 1}, 2);
        tensor.setRow(1, new double[]{1.0}, new int[]{0}, new int[]{0}, 1);

        tensor.resize(1);

        assertEquals(1, tensor.equations());
        assertArrayEquals(new double[]{2.0, 3.0}, tensor.getRowValues(0), EPSILON);
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.getRowValues(1));
    }
}