package com.rae.formicapi.math;

import com.rae.formicapi.math.operators.DenseMatrix;
import com.rae.formicapi.math.solvers.ConjugateGradient;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ConjugateGradientTest {

    @Test
    public void testSolve() {

        DenseMatrix m = new DenseMatrix(3,3);

        m.set(0,0,4);
        m.set(0,1,-1);

        m.set(1,0,-1);
        m.set(1,1,4);
        m.set(1,2,-1);

        m.set(2,1,-1);
        m.set(2,2,3);

        double[] b = {15,10,10};

        double[] x = ConjugateGradient.solve(m,new double[3],b,1000,1e-10);

        assertEquals(5,x[0],1e-6);
        assertEquals(5,x[1],1e-6);
        assertEquals(5,x[2],1e-6);
    }
}
