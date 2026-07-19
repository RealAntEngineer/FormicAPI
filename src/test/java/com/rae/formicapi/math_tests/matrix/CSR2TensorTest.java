package com.rae.formicapi.math_tests.matrix;

import com.rae.formicapi.foundation.math.operators.PaddedCSR2Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CSR2TensorTest {


    @Test
    void multiplyJacobianMatchesFiniteDifference() {

        PaddedCSR2Tensor C = new PaddedCSR2Tensor(2, 2, 2);

        /*
         * F0 = 2*x0² + 3*x0*x1
         * F1 = 5*x0*x1 + 7*x1²
         */
        C.setRow(0,
                new double[]{2.0, 3.0},
                new int[]{0, 0},
                new int[]{0, 1},
                2);

        C.setRow(1,
                new double[]{5.0, 7.0},
                new int[]{0, 1},
                new int[]{1, 1},
                2);

        double[] x = {1.2, -0.7};
        double[] direction = {0.4, -0.8};

        double[] Jd = new double[2];
        C.multiplyJacobian(x, direction, Jd);

        double eps = 1e-8;

        double[] x2 = {
                x[0] + eps * direction[0],
                x[1] + eps * direction[1]
        };

        double[] Fx = new double[2];
        double[] Fx2 = new double[2];

        C.multiply(x, Fx);
        C.multiply(x2, Fx2);

        for (int i = 0; i < 2; i++) {
            double fd = (Fx2[i] - Fx[i]) / eps;
            assertEquals(fd, Jd[i], 1e-6);
        }
    }
}
