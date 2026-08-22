package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSRTensor;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(3)
public class TensorSpecializationBenchmark {

    @Param({"1000", "100000"})
    public int equations;

    @Param({"4", "16"})
    public int termsPerEquation;

    @Param({"RANDOM", "SEQUENTIAL"})
    public String indexPattern;

    private PaddedCSR3Tensor specialized;
    private PaddedCSRTensor  generic;

    private double[] x;
    private double[] direction;

    private double[] resultSpecialized;
    private double[] resultGeneric;

    @Setup(Level.Trial)
    public void setup() {

        Random rnd = new Random(42);

        specialized =
                new PaddedCSR3Tensor(
                        equations,
                        termsPerEquation
                );

        // order = 2 => quadratic / 3rd-order tensor
        generic =
                new PaddedCSRTensor(
                        2,
                        equations,
                        termsPerEquation
                );

        double[] values = new double[termsPerEquation];

        int[] var1 = new int[termsPerEquation];
        int[] var2 = new int[termsPerEquation];

        int[][] genericIndices = new int[termsPerEquation][2];

        for (int eq = 0; eq < equations; eq++) {

            for (int t = 0; t < termsPerEquation; t++) {

                values[t] = rnd.nextDouble() * 2.0 - 1.0;

                int j = index(eq, t, 0, rnd);
                int k = index(eq, t, 1, rnd);

                var1[t] = j;
                var2[t] = k;

                genericIndices[t][0] = j;
                genericIndices[t][1] = k;
            }

            specialized.setRow(
                    eq,
                    values,
                    var1,
                    var2,
                    termsPerEquation
            );

            generic.setRow(
                    eq,
                    values,
                    genericIndices,
                    termsPerEquation
            );
        }

        x = new double[equations];
        direction = new double[equations];

        for (int i = 0; i < equations; i++) {
            x[i] = rnd.nextDouble();
            direction[i] = rnd.nextDouble();
        }

        resultSpecialized = new double[equations];
        resultGeneric = new double[equations];
    }

    private int index(int equation, int term, int which, Random rnd) {

        return switch (indexPattern) {

            case "RANDOM" ->
                    rnd.nextInt(equations);

            case "SEQUENTIAL" ->
                    (equation * termsPerEquation
                            + term * 2
                            + which)
                            % equations;

            default ->
                    throw new IllegalStateException(
                            "Unknown index pattern: " + indexPattern
                    );
        };
    }

    // ------------------------------------------------------------
    // F(x)
    // ------------------------------------------------------------

    @Benchmark
    public void specializedApply(Blackhole bh) {

        specialized.multiply(
                x,
                resultSpecialized
        );

        bh.consume(resultSpecialized);
    }

    @Benchmark
    public void genericApply(Blackhole bh) {

        generic.multiply(
                x,
                resultGeneric
        );

        bh.consume(resultGeneric);
    }

    // ------------------------------------------------------------
    // J(x) * direction
    // ------------------------------------------------------------

    @Benchmark
    public void specializedJacobian(Blackhole bh) {

        specialized.multiplyJacobian(
                x,
                direction,
                resultSpecialized
        );

        bh.consume(resultSpecialized);
    }

    @Benchmark
    public void genericJacobian(Blackhole bh) {

        generic.multiplyJacobian(
                x,
                direction,
                resultGeneric
        );

        bh.consume(resultGeneric);
    }
}