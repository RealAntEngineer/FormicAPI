package com.rae.formicapi.math;

import com.rae.formicapi.math.operators.DynamicCSRMatrix;
import com.rae.formicapi.math.solvers.LeastSquare;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LeastSquare} covering correctness across system types:
 * square, overdetermined, underdetermined, symmetric, and ill-conditioned.
 */
public class LeastSquareTest {

    private static final double TOL = 1e-9;
    private static final int MAX_ITER = 10_000;

    // ------------------------------------------------
    // Helpers
    // ------------------------------------------------

    @Test
    public void testIdentitySystem() {
        double[][] A = {
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        };
        double[] b = {3, 7, 2};
        double[] expected = {3, 7, 2};

        double[] x = LeastSquare.solve(denseToSparce(A), new double[3], b, MAX_ITER, TOL);
        assertVectorEquals(expected, x, 1e-6, "identity system");
    }

    /**
     * Builds a DynamicCSRMatrix from a dense 2D array
     */
    private static DynamicCSRMatrix denseToSparce(double[][] A) {
        int rows = A.length;
        int cols = A[0].length;
        DynamicCSRMatrix m = new DynamicCSRMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (A[r][c] != 0.0)
                    m.add(r, c, A[r][c]);
        return m;
    }

    /**
     * Asserts two vectors are close elementwise
     */
    private static void assertVectorEquals(double[] expected, double[] actual, double tol, String message) {
        assertEquals(expected.length, actual.length, message + " — length mismatch");
        for (int i = 0; i < expected.length; i++)
            assertEquals(expected[i], actual[i], tol, message + " — index " + i);
    }

    @Test
    public void testDiagonalSystem() {
        double[][] A = {
                {2, 0, 0},
                {0, 4, 0},
                {0, 0, 8}
        };
        double[] b = {4, 8, 16};
        double[] expected = {2, 2, 2};

        double[] x = LeastSquare.solve(denseToSparce(A), new double[3], b, MAX_ITER, TOL);
        assertVectorEquals(expected, x, 1e-6, "diagonal system");
    }

    // ------------------------------------------------
    // Square systems
    // ------------------------------------------------

    @Test
    public void testSymmetricPositiveDefinite() {
        // Classic SPD: thermal-like stiffness matrix
        double[][] A = {
                {4, -1, 0},
                {-1, 4, -1},
                {0, -1, 4}
        };
        double[] x_exact = {1, 2, 3};
        double[] b = multiply(A, x_exact);

        double[] x = LeastSquare.solve(denseToSparce(A), new double[3], b, MAX_ITER, TOL);
        assertVectorEquals(x_exact, x, 1e-6, "SPD system");
    }

    /**
     * Computes Ax for a dense 2D array
     */
    private static double[] multiply(double[][] A, double[] x) {
        int rows = A.length;
        int cols = A[0].length;
        double[] result = new double[rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r] += A[r][c] * x[c];
        return result;
    }

    @Test
    public void testGeneralSquare() {
        double[][] A = {
                {2, 1, -1},
                {-3, -1, 2},
                {-2, 1, 2}
        };
        double[] x_exact = {1, 2, 3};
        double[] b = multiply(A, x_exact);

        double[] x = LeastSquare.solve(denseToSparce(A), new double[3], b, MAX_ITER, TOL);
        assertVectorEquals(x_exact, x, 1e-6, "general square system");
    }

    @Test
    public void testOverdeterminedConsistent() {
        // Consistent: exact solution exists, extra row is redundant
        double[][] A = {
                {1, 0},
                {0, 1},
                {1, 1}
        };
        double[] x_exact = {3, 5};
        double[] b = multiply(A, x_exact);

        double[] x = LeastSquare.solve(denseToSparce(A), new double[2], b, MAX_ITER, TOL);
        assertVectorEquals(x_exact, x, 1e-6, "overdetermined consistent");
    }

    // ------------------------------------------------
    // Overdetermined systems (more rows than cols)
    // ------------------------------------------------

    @Test
    public void testOverdeterminedInconsistent() {
        double[][] A = {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        };
        double[] b = { 6, 5, 7, 10 };

        double[] x = LeastSquare.solve(denseToSparce(A), new double[2], b, MAX_ITER, TOL);

        double residual = residualNorm(A, x, b);

        // Verify optimality: perturbing the solution in any direction should not reduce the residual
        Random rng = new Random(42);
        for (int trial = 0; trial < 100; trial++) {
            double[] perturbed = x.clone();
            for (int i = 0; i < perturbed.length; i++)
                perturbed[i] += (rng.nextDouble() - 0.5) * 0.1;

            double perturbedResidual = residualNorm(A, perturbed, b);
            assertTrue(residual <= perturbedResidual + 1e-6,
                    "least-squares solution should be optimal — perturbation trial " + trial);
        }
    }

    /**
     * Computes ||Ax - b||₂
     */
    private static double residualNorm(double[][] A, double[] x, double[] b) {
        double[] Ax = multiply(A, x);
        double sum = 0;
        for (int i = 0; i < b.length; i++) sum += (Ax[i] - b[i]) * (Ax[i] - b[i]);
        return Math.sqrt(sum);
    }

    // ------------------------------------------------
    // Ill-conditioned systems
    // ------------------------------------------------

    @Test
    public void testHilbertMatrix3x3() {
        // Hilbert matrix is notoriously ill-conditioned
        double[][] A = {
                {1.0, 1.0 / 2, 1.0 / 3},
                {1.0 / 2, 1.0 / 3, 1.0 / 4},
                {1.0 / 3, 1.0 / 4, 1.0 / 5}
        };
        double[] x_exact = {1, 1, 1};
        double[] b = multiply(A, x_exact);

        double[] x = LeastSquare.solve(denseToSparce(A), new double[3], b, MAX_ITER, TOL);

        // Loose tolerance — Hilbert matrices amplify errors
        assertVectorEquals(x_exact, x, 1e-4, "Hilbert 3x3");
    }

    // ------------------------------------------------
    // Edge cases
    // ------------------------------------------------

    @Test
    public void testSingleElement() {
        double[][] A = {{5.0}};
        double[] b = {15.0};

        double[] x = LeastSquare.solve(denseToSparce(A), new double[1], b, MAX_ITER, TOL);
        assertVectorEquals(new double[]{3.0}, x, 1e-9, "single element");
    }

    @Test
    public void testZeroRHS() {
        double[][] A = {
                {3, 1},
                {1, 2}
        };
        double[] b = {0, 0};

        double[] x = LeastSquare.solve(denseToSparce(A), new double[2], b, MAX_ITER, TOL);
        assertVectorEquals(new double[]{0, 0}, x, 1e-9, "zero RHS");
    }

    @Test
    public void testAlreadyAtSolution() {
        double[][] A = {
                {2, 1},
                {1, 3}
        };
        double[] x_exact = {2, 1};
        double[] b = multiply(A, x_exact);

        // Pass exact solution as initial guess — should converge in 0 iterations
        double[] x = LeastSquare.solve(denseToSparce(A), x_exact.clone(), b, MAX_ITER, TOL);
        assertVectorEquals(x_exact, x, 1e-9, "initial guess is solution");
    }

    @Test
    public void testWrongRHSLengthThrows() {
        double[][] A = {
                {1, 0},
                {0, 1}
        };
        double[] b = {1, 2, 3}; // wrong length

        assertThrows(IllegalArgumentException.class,
                () -> LeastSquare.solve(denseToSparce(A), new double[2], b, MAX_ITER, TOL),
                "should throw on RHS length mismatch");
    }

    @Test
    public void testWrongInitialGuessLengthThrows() {
        double[][] A = {
                {1, 0},
                {0, 1}
        };
        double[] b = {1, 2};
        double[] x_init = {0, 0, 0}; // wrong length

        assertThrows(IllegalArgumentException.class,
                () -> LeastSquare.solve(denseToSparce(A), x_init, b, MAX_ITER, TOL),
                "should throw on initial guess length mismatch");
    }
}
