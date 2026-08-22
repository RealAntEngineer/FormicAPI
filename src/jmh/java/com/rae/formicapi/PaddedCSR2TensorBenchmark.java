package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for {@link PaddedCSR3Tensor#apply} (F(x) = C:x:x) and
 * {@link PaddedCSR3Tensor#multiplyJacobian} (J(x)*direction).
 *
 * <p>Both are the per-Newton-iteration / per-BiCGSTAB-iteration hot paths in
 * {@link com.rae.formicapi.foundation.math.solvers.NewtonKrylov}, so this
 * benchmark is meant to catch regressions in either op independently of the
 * solver's convergence behavior.
 *
 * <p>{@code equations} doubles as the variable-space size: {@code x} and
 * {@code direction} are sized {@code equations} and each term's two variable
 * indices are drawn from {@code [0, equations)}, matching the square
 * (variables == equations) case {@code NewtonKrylov.solve} assumes.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class PaddedCSR2TensorBenchmark {

    /** Number of equations (== number of variables, square case). */
    @Param({"1000", "100000"})
    public int equations;

    /** Fixed nonzero quadratic terms per equation (sparsity level). */
    @Param({"4", "16"})
    public int termsPerEquation;

    @Param({"RANDOM", "SEQUENTIAL"})
    public String indexPattern;

    private PaddedCSR3Tensor tensor;
    private DoubleVector     x;
    private DoubleVector direction;
    private DoubleVector result;

    @Setup(Level.Trial)
    public void setup() {
        Random rnd = new Random(42);

        tensor = new PaddedCSR3Tensor(equations, termsPerEquation);

        double[] rowValues = new double[termsPerEquation];
        int[] rowVar1 = new int[termsPerEquation];
        int[] rowVar2 = new int[termsPerEquation];

        for (int eq = 0; eq < equations; eq++) {
            for (int t = 0; t < termsPerEquation; t++) {
                rowValues[t] = rnd.nextDouble() * 2.0 - 1.0;

                rowVar1[t] = index(eq, t, 0, rnd);
                rowVar2[t] = index(eq, t, 1, rnd);
            }
            tensor.setRow(eq, rowValues, rowVar1, rowVar2, termsPerEquation);
        }

        double[] xArr = new double[equations];
        double[] dArr = new double[equations];
        for (int i = 0; i < equations; i++) {
            xArr[i] = rnd.nextDouble();
            dArr[i] = rnd.nextDouble();
        }

        x = new CpuDoubleVector(xArr);
        direction = new CpuDoubleVector(dArr);
        result = new CpuDoubleVector(equations);
    }

    @Benchmark
    public void apply(Blackhole bh) {
        tensor.apply(x, result);
        bh.consume(result);
    }

    @Benchmark
    public void multiplyJacobian(Blackhole bh) {
        tensor.multiplyJacobian(x, direction, result);
        bh.consume(result);
    }


    private int index(int eq, int term, int which, Random rnd) {
        return switch (indexPattern) {

            case "SEQUENTIAL" ->
                    (eq * termsPerEquation + term + which) % equations;

            case "RANDOM" ->
                    rnd.nextInt(equations);

            default ->
                    throw new IllegalStateException(indexPattern);
        };
    }
}