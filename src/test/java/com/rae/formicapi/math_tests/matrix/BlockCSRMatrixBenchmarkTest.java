package com.rae.formicapi.math_tests.matrix;

import com.rae.formicapi.foundation.math.operators.linear.Block7PointMatrix;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class BlockCSRMatrixBenchmarkTest {

    private static final int BLOCKS = 1024;
    private static final int SIZE = BLOCKS * 4096;

    private static final int ITERATIONS = 100;

    static final double rowsProcessed = (double) SIZE * ITERATIONS;


    @Test
    public void testMul() {



        Block7PointMatrix matrix = new Block7PointMatrix(BLOCKS);

        Random random = new Random(42);

        initialize(matrix, random);

        double[] x = new double[SIZE];
        double[] result = new double[SIZE];

        for (int i = 0; i < SIZE; i++) {
            x[i] = random.nextDouble();
        }


        // warmup
        for (int i = 0; i < 20; i++) {
            matrix.multiply(x, result);
        }


        long start = System.nanoTime();

        for (int i = 0; i < ITERATIONS; i++) {
            matrix.multiply(x, result);
        }

        long end = System.nanoTime();


        double seconds = (end - start) / 1e9;


        System.out.println("Blocks: " + BLOCKS);
        System.out.println("Rows:   " + SIZE);

        System.out.println(
                "multiply: " +
                (rowsProcessed / seconds / 1e6) +
                " million rows/sec"
        );

        System.out.println(
                "multiply: " +
                        (seconds * 1e9/rowsProcessed) +
                        " ns / rows"
        );

        start = System.nanoTime();

        /*for (int i = 0; i < ITERATIONS; i++) {
            matrix.multiplyFast(x, result);
        }

        end = System.nanoTime();


        seconds = (end - start) / 1e9;

        System.out.println(
                "fast multiply: " +
                        (rowsProcessed / seconds / 1e6) +
                        " million rows/sec"
        );


        // transpose benchmark

        for (int i = 0; i < ITERATIONS; i++) {
            matrix.transposeMultiplyFast(x, result);
        }

        end = System.nanoTime();


        seconds = (end - start) / 1e9;

        System.out.println(
                "fast transpose: " +
                        (rowsProcessed / seconds / 1e6) +
                        " million rows/sec"
        );

        start = System.nanoTime();

        for (int i = 0; i < ITERATIONS; i++) {
            matrix.transposeMultiply(x, result);
        }

        end = System.nanoTime();


        seconds = (end - start) / 1e9;


        System.out.println(
                "transposeMultiply: " +
                (rowsProcessed / seconds / 1e6) +
                " million rows/sec"
        );*/
    }


    private static void initialize(
            Block7PointMatrix matrix,
            Random random
    ) {

        for (int b = 0; b < BLOCKS; b++) {

            int[] neighbours = new int[6];

            // linear block layout for benchmark
            neighbours[0] = b > 0 ? (b - 1) * 4096 : -1;
            neighbours[1] = b < BLOCKS - 1 ? (b + 1) * 4096 : -1;

            neighbours[2] = -1;
            neighbours[3] = -1;
            neighbours[4] = -1;
            neighbours[5] = -1;

            matrix.setNeighbors(b, neighbours);


            for (int i = 0; i < 4096; i++) {

                double[] row = new double[7];

                row[0] = random.nextDouble();
                row[1] = random.nextDouble();
                row[2] = random.nextDouble();
                row[3] = random.nextDouble();
                row[4] = random.nextDouble();
                row[5] = random.nextDouble();
                row[6] = random.nextDouble();

                matrix.setRow(
                        b * 4096 + i,
                        row
                );
            }
        }
    }
}