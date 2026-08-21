package com.rae.formicapi;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSR2Tensor;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for the CPU-parallel {@link CpuPaddedCSR2Tensor}:
 * {@link CpuPaddedCSR2Tensor#apply} (F(x) = C:x:x) and
 * {@link CpuPaddedCSR2Tensor#applyJacobian} (J(x)*direction).
 *
 * <p>Compare against the plain (serial) {@code PaddedCSR2TensorBenchmark} to
 * see whether/where parallel dispatch actually pays for itself.
 * {@code equations} below {@link CpuExecutor#DEFAULT_PARALLEL_THRESHOLD}
 * (4096) falls back to the exact same serial loop regardless of
 * {@code threads} — the {@code equations=1000} row is included on purpose to
 * make that crossover visible in the results (expect it to look like pure
 * dispatch overhead with zero benefit, not a bug).
 *
 * <p>{@code equations} doubles as the variable-space size, same convention as
 * {@code PaddedCSR2TensorBenchmark}: {@code x}/{@code direction} are sized
 * {@code equations} and each term's two variable indices are drawn from
 * {@code [0, equations)}.
 *
 * <p><b>Every trial's {@link CpuExecutor} is shut down in
 * {@code @TearDown(Level.Trial)}.</b> Skipping this leaves its worker threads
 * busy-spinning (pinning a core each) for the rest of the JVM's life — see
 * {@link CpuExecutor#shutdown()} — which would silently contend with and
 * skew every later trial in the same fork.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class CpuPaddedCSR2TensorParallelBenchmark {

    /**
     * Number of equations (== number of variables, square case). Includes one
     * value below {@link CpuExecutor#DEFAULT_PARALLEL_THRESHOLD} (4096) to
     * exercise the serial-fallback path.
     */
    @Param({"1000", "10000", "1000000"})
    public int equations;

    /** Fixed nonzero quadratic terms per equation (sparsity level). */
    @Param({"4", "16"})
    public int termsPerEquation;

    /** CpuExecutor worker thread count. Tune to the target machine's core count. */
    @Param({"1", "4", "8"})
    public int threads;

    private CpuExecutor executor;
    private CpuPaddedCSR2Tensor tensor;
    private CpuDoubleVector     x;
    private CpuDoubleVector     direction;
    private CpuDoubleVector result;

    @Setup(Level.Trial)
    public void setup() {
        Random rnd = new Random(42);

        executor = new CpuExecutor(threads);

        tensor = new CpuPaddedCSR2Tensor(equations, termsPerEquation);
        tensor.setExecutor(executor);

        double[] rowValues = new double[termsPerEquation];
        int[] rowVar1 = new int[termsPerEquation];
        int[] rowVar2 = new int[termsPerEquation];

        for (int eq = 0; eq < equations; eq++) {
            for (int t = 0; t < termsPerEquation; t++) {
                rowValues[t] = rnd.nextDouble() * 2.0 - 1.0;
                rowVar1[t] = rnd.nextInt(equations);
                rowVar2[t] = rnd.nextInt(equations);
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

    @TearDown(Level.Trial)
    public void tearDown() {
        executor.shutdown();
    }

    @Benchmark
    public void apply(Blackhole bh) {
        tensor.apply(x, result);
        bh.consume(result);
    }

    @Benchmark
    public void applyJacobian(Blackhole bh) {
        tensor.applyJacobian(x, direction, result);
        bh.consume(result);
    }
}