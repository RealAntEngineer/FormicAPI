package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.linear.Block7PointMatrix;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH conversion of {@code BlockCSRMatrixBenchmarkTest}.
 *
 * <p>Each block is a fixed 16x16x16 = 4096-row unit; {@code blocks} is the
 * number of such units chained in a line (block {@code b} neighbours
 * {@code b-1} and {@code b+1} only — matches the original test's linear
 * layout).
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(2)
public class Block7PointMatrixBenchmark {

    private static final int BLOCK_SIZE = 4096;

    @Param({"64", "256", "1024"})
    private int blocks;

    private Block7PointMatrix matrix;
    private double[] x;
    private double[] result;

    @Setup(Level.Trial)
    public void setup() {
        int size = blocks * BLOCK_SIZE;

        matrix = new Block7PointMatrix(blocks);
        Random random = new Random(42);
        initialize(matrix, blocks, random);

        x = new double[size];
        for (int i = 0; i < size; i++)
            x[i] = random.nextDouble();

        result = new double[size];
    }

    @Benchmark
    public void multiply(Blackhole bh) {
        matrix.multiply(x, result);
        bh.consume(result);
    }

    // NOTE: the original test also exercised multiplyFast, transposeMultiplyFast,
    // and transposeMultiply, but had them commented out. Uncomment/add analogous
    // @Benchmark methods here once those code paths are ready to measure:
    //
    // @Benchmark
    // public void transposeMultiply(Blackhole bh) {
    //     matrix.transposeMultiply(x, result);
    //     bh.consume(result);
    // }

    private static void initialize(Block7PointMatrix matrix, int blocks, Random random) {

        for (int b = 0; b < blocks; b++) {

            int[] neighbours = new int[6];

            // linear block layout for benchmark
            neighbours[0] = b > 0 ? (b - 1) * BLOCK_SIZE : -1;
            neighbours[1] = b < blocks - 1 ? (b + 1) * BLOCK_SIZE : -1;

            neighbours[2] = -1;
            neighbours[3] = -1;
            neighbours[4] = -1;
            neighbours[5] = -1;

            matrix.setNeighbors(b, neighbours);

            for (int i = 0; i < BLOCK_SIZE; i++) {

                double[] row = new double[7];

                row[0] = random.nextDouble();
                row[1] = random.nextDouble();
                row[2] = random.nextDouble();
                row[3] = random.nextDouble();
                row[4] = random.nextDouble();
                row[5] = random.nextDouble();
                row[6] = random.nextDouble();

                matrix.setRow(b * BLOCK_SIZE + i, row);
            }
        }
    }
}