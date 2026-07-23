package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.foundation.math.operators.linear.DenseMatrix;
import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.solvers.BiCGStab;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class BiCGStabTest {

    private static final double EPS = 1e-10;

    @Test
    void solve2x2() {
        MutableMatrix A = new DenseMatrix(2, 2);

        A.set(0,0,4);
        A.set(0,1,1);
        A.set(1,0,2);
        A.set(1,1,3);

        double[] b = {1, 2};

        double[] x = BiCGStab.solve(A, b, 100, 1e-12);

        assertEquals(0.1, x[0], EPS);
        assertEquals(0.6, x[1], EPS);
    }

    @Test
    void solveIdentity() {
        MutableMatrix A = new DenseMatrix(4,4);

        for (int i = 0; i < 4; i++)
            A.set(i, i, 1);

        double[] b = {3, -1, 2, 7};

        double[] x = BiCGStab.solve(A, b, 10, 1e-12);

        assertArrayEquals(b, x, EPS);
    }

    @Test
    void warmStartAlreadySolved() {
        MutableMatrix A = new DenseMatrix(2,2);

        A.set(0,0,5);
        A.set(0,1,1);
        A.set(1,0,1);
        A.set(1,1,4);

        double[] x = {1,2};

        double[] b = new double[2];
        A.multiply(x, b);

        double[] out = BiCGStab.solve(A, x, b, 50, 1e-12);

        assertArrayEquals(new double[]{1,2}, out, EPS);
    }

    @Test
    void solveNonSymmetric() {
        MutableMatrix A = new DenseMatrix(3,3);

        A.set(0,0,3);
        A.set(0,1,2);
        A.set(0,2,-1);

        A.set(1,0,1);
        A.set(1,1,-1);
        A.set(1,2,2);

        A.set(2,0,2);
        A.set(2,1,1);
        A.set(2,2,2);

        double[] expected = {1,2,-1};

        double[] b = new double[3];
        A.multiply(expected, b);

        double[] x = BiCGStab.solve(A, b, 100, 1e-12);

        assertArrayEquals(expected, x, EPS);
    }

    @Test
    void solveNonSymmetricSystem2() {

        MutableMatrix A = new DenseMatrix(3, 3);

        /*
         *  4 1 0
         *  2 3 1
         *  0 4 3
         */
        A.set(0, 0, 4);
        A.set(0, 1, 1);

        A.set(1, 0, 2);
        A.set(1, 1, 3);
        A.set(1, 2, 1);

        A.set(2, 1, 4);
        A.set(2, 2, 3);

        double[] expected = {1, 2, -1};

        double[] b = new double[3];
        A.multiply(expected, b);

        double[] x = BiCGStab.solve(A, b, 100, 1e-12);

        assertArrayEquals(expected, x, 1e-8);
    }
    @Test
    void solveConstrained() {

        MutableMatrix A = new DenseMatrix(2,3);

        A.set(0,0,2);
        A.set(0,1,1);

        A.set(1,0,1);
        A.set(1,1,3);
        A.set(1,2,4);

        double[] x = new double[3];
        x[2] = 2.0;               // fixed value

        boolean[] fixed = {false, false, true};

        double[] b = {4, 13};

        double[] r = new double[2];
        double[] rHat = new double[2];
        double[] p = new double[3];
        double[] v = new double[2];
        double[] s = new double[3];
        double[] t = new double[2];
        int[] unknown = new int[2];

        BiCGStab.solveConstrained(A, x, fixed, b, 100, 1e-12, false,
                r, rHat, p, v, s, t, unknown);


        assertEquals(1.4, x[0], EPS);
        assertEquals(1.2, x[1], EPS);
        assertEquals(2.0, x[2], EPS);

    }

    @Test
    void solveIllConditionedDiagonal() {

        MutableMatrix A = new DenseMatrix(3, 3);

        A.set(0, 0, 1e-6);
        A.set(1, 1, 1);
        A.set(2, 2, 1e6);


        double[] expected = {2, -3, 4};

        double[] b = new double[3];
        A.multiply(expected, b);


        double[] x = BiCGStab.solve(A, b, 500, 1e-12);

        assertArrayEquals(expected, x, 1e-6);
    }
}