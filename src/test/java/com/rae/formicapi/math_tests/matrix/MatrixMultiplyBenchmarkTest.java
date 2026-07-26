package com.rae.formicapi.math_tests.matrix;

import com.rae.formicapi.foundation.math.operators.linear.Block7PointMatrix;
import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class MatrixMultiplyBenchmarkTest {

    private static final int BLOCKS = 64;
    private static final int ROWS = BLOCKS * 4096;

    private static final int WARMUP = 20;
    private static final int ITERATIONS = 100;

    @Test
    public void compairMul() {

        Random random = new Random(42);

        double[] x = new double[ROWS];
        double[] result = new double[ROWS];

        for (int i = 0; i < ROWS; i++) {
            x[i] = random.nextDouble();
        }


        Block7PointMatrix block  = createBlockMatrix(random);
        PaddedCSRMatrix   padded = createPaddedMatrix(random);


        benchmark(
                "BlockCSRMatrix",
                block,
                x,
                result
        );


        benchmark(
                "PaddedCSRMatrix",
                padded,
                x,
                result
        );
    }


    private static void benchmark(
            String name,
            MutableMatrix matrix,
            double[] x,
            double[] result
    ) {

        for (int i = 0; i < WARMUP; i++) {
            matrix.multiply(x, result);
        }


        long start = System.nanoTime();

        for (int i = 0; i < ITERATIONS; i++) {
            matrix.multiply(x, result);
        }

        long elapsed = System.nanoTime() - start;


        long rowsProcessed =
                (long) matrix.rows() * ITERATIONS;


        double nsPerRow =
                (double) elapsed / rowsProcessed;


        double rowsPerSecond =
                1e9 / nsPerRow;


        System.out.printf(
                "%-20s %.2f ns/row %.2f M rows/s%n",
                name,
                nsPerRow,
                rowsPerSecond / 1e6
        );
    }


    private static Block7PointMatrix createBlockMatrix(Random random) {

        Block7PointMatrix matrix =
                new Block7PointMatrix(BLOCKS);


        for (int b = 0; b < BLOCKS; b++) {

            int[] neighbours = {
                    b > 0 ? (b - 1) * 4096 : -1,
                    b < BLOCKS - 1 ? (b + 1) * 4096 : -1,
                    -1,
                    -1,
                    -1,
                    -1
            };

            matrix.setNeighbors(b, neighbours);


            for (int i = 0; i < 4096; i++) {

                double[] row = new double[7];

                for (int j = 0; j < 7; j++) {
                    row[j] = random.nextDouble();
                }

                matrix.setRow(
                        b * 4096 + i,
                        row
                );
            }
        }

        return matrix;
    }


    private static PaddedCSRMatrix createPaddedMatrix(Random random) {

        int rows = BLOCKS * 4096;

        PaddedCSRMatrix matrix =
                new PaddedCSRMatrix(rows, rows, 7);


        for (int b = 0; b < BLOCKS; b++) {

            int self = b * 4096;

            int xmBlock = b > 0 ? (b - 1) * 4096 : -1;
            int xpBlock = b < BLOCKS - 1 ? (b + 1) * 4096 : -1;


            for (int idx = 0; idx < 4096; idx++) {

                int row = self + idx;

                double[] values = new double[7];
                int[] cols = new int[7];


                values[0] = random.nextDouble();
                cols[0] = row;


                int x = idx & 15;
                int z = (idx >> 4) & 15;
                int y = idx >> 8;


                // -X
                if (x > 0) {
                    cols[1] = row - 1;
                } else {
                    cols[1] = xmBlock != -1
                            ? xmBlock + idx + 15
                            : row;
                }


                // +X
                if (x < 15) {
                    cols[2] = row + 1;
                } else {
                    cols[2] = xpBlock != -1
                            ? xpBlock + idx - 15
                            : row;
                }


                // -Y
                if (y > 0) {
                    cols[3] = row - 256;
                } else {
                    cols[3] = row; // no neighbour
                }


                // +Y
                if (y < 15) {
                    cols[4] = row + 256;
                } else {
                    cols[4] = row;
                }


                // -Z
                if (z > 0) {
                    cols[5] = row - 16;
                } else {
                    cols[5] = row;
                }


                // +Z
                if (z < 15) {
                    cols[6] = row + 16;
                } else {
                    cols[6] = row;
                }


                for (int i = 1; i < 7; i++) {
                    values[i] = random.nextDouble();
                }


                matrix.setRow(
                        row,
                        values,
                        cols,
                        7
                );
            }
        }

        return matrix;
    }
}