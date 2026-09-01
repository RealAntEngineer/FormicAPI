package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.DenseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class NewtonKrylovTest {
    @Test
    void solveSimpleNonlinearSystem() {

        PaddedCSR3Tensor C = new PaddedCSR3Tensor(2, 1);


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

        NewtonKrylov.solve(C, A, new CpuDoubleVector(x), new CpuDoubleVector(b),
                20, 20,
                1e-12, 1e-12,null, null, null,
                new WorkingBuffer<>(7, () -> new CpuDoubleVector(A.outputSize())),
                new WorkingBuffer<>(3, () -> new CpuDoubleVector(A.outputSize())));

        assertArrayEquals(new double[]{1.0, 2.0}, x, 1e-8);
    }

    @Test
    void solveCoupledQuadraticSystem() {

        PaddedCSR3Tensor C = new PaddedCSR3Tensor(2, 2);

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

        PaddedCSR3Tensor C = new PaddedCSR3Tensor(2, 1);

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


    private static void solve(PaddedCSR3Tensor C, Matrix A, double[] x, double[] b) {

        int n = b.length;
        CpuDoubleVector xVec = new CpuDoubleVector(x);
        NewtonKrylov.solve(C, A, xVec, new CpuDoubleVector(b),
                50, 50,
                1e-12, 1e-12,null, null, null,
                new WorkingBuffer<>(7, () -> new CpuDoubleVector(n)),
                new WorkingBuffer<>(3, () -> new CpuDoubleVector(n)));
    }
}
