package com.rae.formicapi.math_tests.matrix;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuVector;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class ParallelCpuTest {

    @Test
    void multiplyParallelMatchesSerial() {

        int nx = 64;
        int ny = 64;
        int nz = 64;

        int rows = nx * ny * nz;
        int cols = rows;

        int entriesPerRow = 7;

        CpuPaddedCSRMatrix matrix =
                new CpuPaddedCSRMatrix(rows, cols, entriesPerRow);

        PaddedCSRMatrix matrixSerial =
                new PaddedCSRMatrix(rows, cols, entriesPerRow);


        Random random = new Random(42);

        double[] values = new double[entriesPerRow];
        int[] indices = new int[entriesPerRow];


        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {

                    int row = x + nx * (y + ny * z);

                    int count = 0;

                    // center
                    indices[count] = row;
                    values[count++] = random.nextDouble();


                    // x-
                    if (x > 0) {
                        indices[count] = row - 1;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // x+
                    if (x < nx - 1) {
                        indices[count] = row + 1;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // y-
                    if (y > 0) {
                        indices[count] = row - nx;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // y+
                    if (y < ny - 1) {
                        indices[count] = row + nx;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // z-
                    if (z > 0) {
                        indices[count] = row - nx * ny;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    // z+
                    if (z < nz - 1) {
                        indices[count] = row + nx * ny;
                        values[count++] = random.nextDouble();
                    } else {
                        indices[count] = row;
                        values[count++] = 0.0;
                    }


                    matrix.setRow(row, values, indices, entriesPerRow);
                    matrixSerial.setRow(row, values, indices, entriesPerRow);
                }
            }
        }
        CpuVector x = new CpuVector(cols);

        for (int i = 0; i < cols; i++)
            x.array()[i]  = random.nextDouble();

        CpuVector serial = new CpuVector(rows);
        CpuVector parallel = new CpuVector(rows);

        // Serial
        long start = System.nanoTime();
        for (int i = 0; i < 10; i++) {
            //matrixSerial.apply(x, serial);
        }

        System.out.println("serial took   "+ (System.nanoTime() - start)/10);

        double[] xArr = x.array();
        double[] serialArr = serial.array();
        start = System.nanoTime();

        for (int i = 0; i < 10; i++){
            //matrixSerial.multiply(xArr, serialArr);
        }
        System.out.println("serial 2 took "+ (System.nanoTime() - start)/10);

        // Parallel

        //System.out.println("detected "+Runtime.getRuntime().availableProcessors()+ " available processors");
        matrix.setExecutor(new CpuExecutor(4));
        start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrix.apply(x, parallel);
        }
        System.out.println("parallel took "+ (System.nanoTime() - start)/1000);

        assertArrayEquals(
                serial.array(),
                parallel.array(),
                0.0
        );
    }
}
