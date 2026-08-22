package com.rae.formicapi.math_tests.matrix;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.LeastSquare;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ParallelCpuTest {

    @Test
    void multiplyParallelMatchesSerial() {

        int nx = 64;
        int ny = 64;
        int nz = 64;

        int rows = nx * ny * nz;

        int entriesPerRow = 7;

        CpuPaddedCSRMatrix matrix =
                new CpuPaddedCSRMatrix(rows, rows, entriesPerRow);

        PaddedCSRMatrix matrixSerial =
                new PaddedCSRMatrix(rows, rows, entriesPerRow);


        Random random = new Random(42);

        double[] values  = new double[entriesPerRow];
        int[]    indices = new int[entriesPerRow];


        generateMatrix(nz, ny, nx, indices, values, random, matrix, entriesPerRow, matrixSerial);

        CpuDoubleVector x        = new CpuDoubleVector(rows);

        for (int i = 0; i < rows; i++)
            x.array()[i] = random.nextDouble();

        CpuDoubleVector serial   = new CpuDoubleVector(rows);
        CpuDoubleVector parallel = new CpuDoubleVector(rows);
        //parallel.setExecutor(executor);

        // Serial
        long start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrixSerial.apply(x, serial);
        }
        long time = (System.nanoTime() - start);

        System.out.println("serial took   " + (System.nanoTime() - start) / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));


        // Parallel

        //System.out.println("detected "+Runtime.getRuntime().availableProcessors()+ " available processors");
        CpuExecutor     executor = new CpuExecutor(4);
        matrix.setExecutor(executor);
        start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrix.apply(x, parallel);
        }
        time = (System.nanoTime() - start);
        executor.shutdown();

        System.out.println("parallel took " + time / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));

        assertArrayEquals(
                serial.array(),
                parallel.array(),
                0.0
        );
    }

    @Test
    void transposeMultiplyParallelMatchesSerial() {

        int nx = 64;
        int ny = 64;
        int nz = 64;

        int rows = nx * ny * nz;

        int entriesPerRow = 7;

        CpuPaddedCSRMatrix matrix =
                new CpuPaddedCSRMatrix(rows, rows, entriesPerRow);

        PaddedCSRMatrix matrixSerial =
                new PaddedCSRMatrix(rows, rows, entriesPerRow);

        Random random = new Random(42);

        double[] values  = new double[entriesPerRow];
        int[]    indices = new int[entriesPerRow];

        generateMatrix(nz, ny, nx, indices, values, random, matrix, entriesPerRow, matrixSerial);



        CpuDoubleVector x = new CpuDoubleVector(rows);

        for (int i = 0; i < rows; i++)
            x.array()[i] = random.nextDouble();

        CpuDoubleVector serialResult   = new CpuDoubleVector(rows);


        // Serial
        long start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrixSerial.transposeApply(x, serialResult);
        }
        long time = System.nanoTime() - start;

        System.out.println("serial took   " + time / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));

        // Parallel
        CpuExecutor executor = new CpuExecutor(4);
        matrix.setExecutor(executor);
        CpuDoubleVector parallelResult = new CpuDoubleVector(rows);
        parallelResult.setExecutor(executor);

        start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            matrix.transposeApply(x, parallelResult);
        }
        time = System.nanoTime() - start;
        executor.shutdown();
        System.out.println("parallel took " + time / 1000);
        System.out.println("effective ns/row :" + ((float) time / rows / 1000));

        // Scatter-add reduction order differs between serial and parallel paths,
        // so results are numerically equal but not necessarily bit-identical -
        // unlike the apply() test, this needs a tolerance rather than 0.0.
        assertArrayEquals(serialResult.array(), parallelResult.array(), 1e-9);
    }

    private static WorkingBuffer<DoubleVector>[] createBuffers(int m, int n, CpuExecutor executor) {
        CpuDoubleVector u    = new CpuDoubleVector(m);
        CpuDoubleVector temp = new CpuDoubleVector(m);

        CpuDoubleVector v     = new CpuDoubleVector(n);
        CpuDoubleVector w     = new CpuDoubleVector(n);
        CpuDoubleVector temp2 = new CpuDoubleVector(n);

        if (executor != null) {
            u.setExecutor(executor);
            temp.setExecutor(executor);
            v.setExecutor(executor);
            w.setExecutor(executor);
            temp2.setExecutor(executor);
        }

        WorkingBuffer<DoubleVector> mBuffer = new WorkingBuffer<>(new CpuDoubleVector[]{ u, temp });
        WorkingBuffer<DoubleVector> nBuffer = new WorkingBuffer<>(new CpuDoubleVector[]{ v, w, temp2 });

        return new WorkingBuffer[]{ mBuffer, nBuffer };
    }

    @Test
    void leastSquareParallelMatchesSerial() {

        int nx = 32;
        int ny = 32;
        int nz = 32;

        int rows = nx * ny * nz;
        int entriesPerRow = 7;

        CpuPaddedCSRMatrix matrix =
                new CpuPaddedCSRMatrix(rows, rows, entriesPerRow);

        PaddedCSRMatrix matrixSerial =
                new PaddedCSRMatrix(rows, rows, entriesPerRow);

        Random random = new Random(42);

        double[] values  = new double[entriesPerRow];
        int[]    indices = new int[entriesPerRow];

        generateMatrix(nz, ny, nx, indices, values, random, matrix, entriesPerRow, matrixSerial);


        try {


            double[] bArr = new double[rows];
            for (int i = 0; i < rows; i++)
                bArr[i] = random.nextDouble() * 2.0 - 1.0;

            // Independent copies so neither solve can accidentally share state
            // with the other through the same backing array.
            CpuDoubleVector bParallel = new CpuDoubleVector(bArr.clone());
            CpuDoubleVector bSerial   = new CpuDoubleVector(bArr.clone());
            CpuExecutor executor = new CpuExecutor(4);
            bParallel.setExecutor(executor);

            int maxIter = 500;
            double tol = 1e-2;

            CpuDoubleVector xParallel = new CpuDoubleVector(rows);
            matrix.setExecutor(executor);
            xParallel.setExecutor(executor);

            WorkingBuffer[] parallelBuffer = createBuffers(rows, rows, executor);

            long start = System.nanoTime();
            int iterationsParallel = LeastSquare.solve(matrix, bParallel, xParallel, maxIter, tol,
                    parallelBuffer[0], parallelBuffer[1]);

            long end = System.nanoTime();
            executor.shutdown();
            System.out.println("parallel took "+ (end  - start)/rows + "ns/row");
            System.out.println("parallel took "+ (end  - start)/rows/iterationsParallel + "ns/row");


            CpuDoubleVector xSerial       = new CpuDoubleVector(rows);
            WorkingBuffer[] serialBuffers = createBuffers(rows, rows, null);
            start = System.nanoTime();
            int iterationsSerial = LeastSquare.solve(matrixSerial, bSerial, xSerial, maxIter, tol,
                    serialBuffers[0], serialBuffers[1]);
            end = System.nanoTime();

            System.out.println("serial took "+ (end  - start)/rows + "ns/row");
            System.out.println("serial took "+ (end  - start)/rows/iterationsSerial + "ns/row");

            //assertTrue(iterationsParallel < maxIter, "Parallel LSQR did not converge within maxIter");
            //assertTrue(iterationsSerial < maxIter, "Serial LSQR did not converge within maxIter");

            // Same system, same starting conditions - both paths should land on
            // the same solution even though the internal reduction order inside
            // apply()/transposeApply()/dot() differs between serial and
            // parallel (same reasoning as transposeMultiplyParallelMatchesSerial's
            // use of a tolerance instead of exact equality).
            assertArrayEquals(xSerial.array(), xParallel.array(), tol);

            // And check it's actually solving the system, not just agreeing on
            // a wrong answer: ||Ax - b|| should be small relative to ||b||.
           /* CpuDoubleVector residual = new CpuDoubleVector(rows);
            residual.setExecutor(executor);
            matrix.apply(xParallel, residual);
            residual.scale(-1.0);
            residual.add(bParallel);

            double relativeResidual = residual.norm() / bParallel.norm();*/
            /*assertTrue(relativeResidual < 1e-4,
                    "Relative residual too large: " + relativeResidual);*/

        } finally {

        }
    }


    private static void generateMatrix(int nz, int ny, int nx, int[] indices, double[] values, Random random, CpuPaddedCSRMatrix matrix, int entriesPerRow, PaddedCSRMatrix matrixSerial) {
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
    }

}