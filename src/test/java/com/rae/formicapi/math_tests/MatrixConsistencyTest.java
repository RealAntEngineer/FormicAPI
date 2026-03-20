package com.rae.formicapi.math_tests;

import com.rae.formicapi.fondation.math.operators.DenseMatrix;
import com.rae.formicapi.fondation.math.operators.HashSparseMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MatrixConsistencyTest {

    @Test
    public void denseAndSparseMatch() {

        DenseMatrix dense = new DenseMatrix(3,3);
        HashSparseMatrix sparse = new HashSparseMatrix(3,3);

        dense.set(0,0,4);
        dense.set(0,1,-1);

        dense.set(1,0,-1);
        dense.set(1,1,4);
        dense.set(1,2,-1);

        dense.set(2,1,-1);
        dense.set(2,2,3);

        sparse.add(0,0,4);
        sparse.add(0,1,-1);

        sparse.add(1,0,-1);
        sparse.add(1,1,4);
        sparse.add(1,2,-1);

        sparse.add(2,1,-1);
        sparse.add(2,2,3);

        double[] x = {1,2,3};

        double[] r1 = new double[3];
        double[] r2 = new double[3];

        dense.multiply(x,r1);
        sparse.multiply(x,r2);

        for(int i=0;i<3;i++)
            assertEquals(r1[i],r2[i],1e-9);
    }
}