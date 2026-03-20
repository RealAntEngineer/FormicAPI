package com.rae.formicapi.math_tests;

import com.rae.formicapi.fondation.math.operators.DenseMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DenseMatrixTest {

    @Test
    public void testMultiply() {

        DenseMatrix m = new DenseMatrix(3,3);

        m.set(0,0,4);
        m.set(0,1,-1);

        m.set(1,0,-1);
        m.set(1,1,4);
        m.set(1,2,-1);

        m.set(2,1,-1);
        m.set(2,2,3);

        double[] x = {1,2,3};
        double[] result = new double[3];

        m.multiply(x, result);

        assertEquals(2, result[0], 1e-9);
        assertEquals(4, result[1], 1e-9);
        assertEquals(7, result[2], 1e-9);
    }
}