package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.foundation.math.operators.DenseMatrix;
import com.rae.formicapi.foundation.math.operators.Matrix;
import com.rae.formicapi.foundation.math.operators.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.PaddedCSR2Tensor;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class NewtonKrylovTest {
    @Test
    void solveSimpleNonlinearSystem() {

        PaddedCSR2Tensor C = new PaddedCSR2Tensor(2, 2, 1);


        C.setRow(0,
                new double[]{1.0},
                new int[]{0},
                new int[]{0},
                1);

        C.setRow(1,
                new double[]{1.0},
                new int[]{1},
                new int[]{1},
                1);

        MutableMatrix A = new DenseMatrix(2, 2);
        A.set(0, 1, 1);
        A.set(1, 0, 1);

        double[] b = {3, 5};

        double[] x = {0, 0};

        double[] Ax = new double[2];
        double[] F = new double[2];
        double[] dx = new double[2];
        double[] r = new double[2];
        double[] rHat0 = new double[2];
        double[] p = new double[2];
        double[] v = new double[2];
        double[] s = new double[2];
        double[] t = new double[2];

        NewtonKrylov.solve(C, A, x, b,
                20, 20,
                1e-12, 1e-12,
                Ax,F, dx, r, rHat0, p, v, s, t);

        assertArrayEquals(new double[]{1.0, 2.0}, x, 1e-8);
    }

    @Test
    void solveCoupledQuadraticSystem() {

        PaddedCSR2Tensor C = new PaddedCSR2Tensor(2, 2, 2);

        /*
         * F0 = x0*x0 + x0*x1
         * F1 = x1*x1 + x0*x1
         */
        C.setRow(0,
                new double[]{1.0, 1.0},
                new int[]{0, 0},
                new int[]{0, 1},
                2);

        C.setRow(1,
                new double[]{1.0, 1.0},
                new int[]{1, 0},
                new int[]{1, 1},
                2);

        MutableMatrix A = new DenseMatrix(2, 2);

        double[] expected = {1.0, 2.0};

        double[] b = new double[2];
        C.multiply(expected, b);

        double[] x = {0.1, 0.1};

        solve(C, A, x, b);

        assertArrayEquals(expected, x, 1e-8);
    }


    @Test
    void solveWithLinearAndQuadraticCoupling() {

        PaddedCSR2Tensor C = new PaddedCSR2Tensor(2, 2, 1);

        /*
         * Quadratic:
         *
         * F0 += x0*x0
         * F1 += x1*x1
         */
        C.setRow(0,
                new double[]{1.0},
                new int[]{0},
                new int[]{0},
                1);

        C.setRow(1,
                new double[]{1.0},
                new int[]{1},
                new int[]{1},
                1);

        MutableMatrix A = new DenseMatrix(2, 2);

        A.set(0, 0, 2.0);
        A.set(0, 1, 1.0);

        A.set(1, 0, 1.0);
        A.set(1, 1, 3.0);

        double[] expected = {2.0, -1.0};

        double[] b = new double[2];

        C.multiply(expected, b);

        double[] Ax = new double[2];
        A.multiply(expected, Ax);

        b[0] += Ax[0];
        b[1] += Ax[1];

        double[] x = {5.0, 5.0};

        solve(C, A, x, b);

        assertArrayEquals(expected, x, 1e-8);
    }


    private static void solve(PaddedCSR2Tensor C, Matrix A, double[] x, double[] b) {

        int n = b.length;

        NewtonKrylov.solve(C, A, x, b,
                50, 50,
                1e-12, 1e-12,
                new double[n],
                new double[n],
                new double[n],
                new double[n],
                new double[n],
                new double[n],
                new double[n],
                new double[n],
                new double[n]);
    }
}
