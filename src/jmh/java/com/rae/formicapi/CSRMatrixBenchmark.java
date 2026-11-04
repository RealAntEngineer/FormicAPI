
package com.rae.formicapi;


import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.CSRMatrixD;
import com.rae.formicapi.foundation.math.operators.linear.DynamicCSRMatrix;
import com.rae.formicapi.foundation.math.operators.linear.HashSparseMatrix;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.*;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(2)
public class CSRMatrixBenchmark {

    @State(Scope.Thread)
    public static class MatrixState {

        @Param({"100", "500", "1000", "2000", "3000"})
        public int n;

        @Param({"1", "5", "10", "20", "50", "100"})
        public int nnzPerRow;

        CSRMatrixD       csr;
        HashSparseMatrix hash;
        double[]         x;
        double[] result;
        CpuDoubleVector vectorX;
        CpuDoubleVector vectorResult;

        @Setup(Level.Trial)
        public void setup() {
            csr = randomCSR(n, n, nnzPerRow);
            hash = toHash(csr);

            x = randomVector(n);
            result = new double[n];

            vectorX = new CpuDoubleVector(x);
            vectorResult = new CpuDoubleVector(result);
        }
    }

    @Benchmark
    public void multiply(MatrixState s, Blackhole bh) {
        s.csr.multiply(s.x, s.result);
        bh.consume(s.result);
    }

    @Benchmark
    public void transposeApply(MatrixState s, Blackhole bh) {
        s.csr.transposeApply(s.vectorX, s.vectorResult);
        bh.consume(s.result);
    }

    @Benchmark
    public void hashTransposeApply(MatrixState s, Blackhole bh) {
        s.hash.transposeApply(s.vectorX, s.vectorResult);
        bh.consume(s.result);
    }

    @Benchmark
    public void defaultTransposeMultiply(MatrixState s, Blackhole bh) {
        Arrays.fill(s.result, 0.0);

        for (int r = 0; r < s.csr.rows(); r++) {
            for (int c = 0; c < s.csr.cols(); c++) {
                s.result[c] += s.csr.get(r, c) * s.x[r];
            }
        }

        bh.consume(s.result);
    }

    private static CSRMatrixD randomCSR(int rows, int cols, int nnzPerRow) {
        Random rng = new Random(42);
        DynamicCSRMatrix m = new DynamicCSRMatrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            Set<Integer> chosen = new HashSet<>();
            while (chosen.size() < Math.min(nnzPerRow, cols))
                chosen.add(rng.nextInt(cols));

            for (int c : chosen)
                m.add(r, c, rng.nextDouble());
        }

        m.multiply(new double[cols], new double[rows]);
        return m.toCSR();
    }

    private static HashSparseMatrix toHash(CSRMatrixD csr) {
        HashSparseMatrix h = new HashSparseMatrix(csr.rows(), csr.cols());

        for (int r = 0; r < csr.rows(); r++)
            for (int c = 0; c < csr.cols(); c++) {
                double v = csr.get(r, c);
                if (v != 0.0) h.set(r, c, v);
            }

        return h;
    }

    private static double[] randomVector(int n) {
        Random rng = new Random(7);
        double[] v = new double[n];

        for (int i = 0; i < n; i++)
            v[i] = rng.nextDouble();

        return v;
    }
}
